# SPDX-License-Identifier: Apache-2.0
# Body release packaging, 2026-11-02; numerical solver retained from the verified snapshot.
"""CPU body converter with exact attachment Jacobians and non-nested inverse.

The inverse solves the stationarity equations of the *same* forward fit at the
observed SMPL-X pose. This is a local inverse, not a proof of a global inverse.
No source-body sidecar, temporal smoothing, or GT lookup is used.
"""
from __future__ import annotations

import numpy as np
import torch

from . import _lmfit as reference
from ._fastskin import FastSoma
from . import meshfit as MF


def global_rotations(model, rotation, root):
    """Rotation-only FK avoids recomputing mesh pose blendshapes."""
    result = []
    for j, parent in enumerate(model.parents):
        parent = int(parent)
        result.append(rotation[:, j] if parent < 0 else result[parent] @ rotation[:, j])
    return torch.stack(result, 1)


class BodySMPLX(reference.SmplxModel):
    """Exact SMPL-X with neutral non-body joints, not a reduced mesh proxy.

    Neutral descendant joints have the same skinning transform as their last
    body ancestor. Sum their weights once; all vertices and body pose
    correctives are retained. A vertex subset only avoids unused Jacobian rows.
    """
    def __init__(self, sx, shape, vertices=None):
        super().__init__(sx, shape)
        weights = torch.zeros(sx.W.shape[0], 22, dtype=sx.W.dtype)
        for j in range(55):
            ancestor = j
            while ancestor >= 22:
                ancestor = int(sx.parents[ancestor])
            weights[:, ancestor] += sx.W[:, j]
        self.W = weights
        self.P = self.P[:, :, :21]
        self.parents = sx.parents[:22]
        self.D = reference.subtree(list(self.parents))
        self.nJ = 22
        if vertices is not None:
            self.v0 = self.v0[vertices]
            self.P = self.P[vertices]
            self.W = self.W[vertices]

    def parts(self, rotation, translation):
        batch = len(rotation)
        vp = self.v0[None] + torch.einsum('vcjab,tjab->tvc', self.P, rotation[:, 1:22] - torch.eye(3))
        gr, gt = [], []
        for j, parent in enumerate(self.parents):
            parent = int(parent)
            gr.append(rotation[:, j] if parent < 0 else gr[parent] @ rotation[:, j])
            gt.append(self.J0[j].expand(batch, 3) if parent < 0 else
                      (gr[parent] @ (self.J0[j] - self.J0[parent])[:, None])[..., 0] + gt[parent])
        gr, gt = torch.stack(gr, 1), torch.stack(gt, 1) + translation[:, None]
        at = gt - (gr @ self.J0[:22, :, None])[..., 0]
        return vp, gr, at, gt


class Attachment:
    def __init__(self, faces, att, select=None):
        tri, bary, offset = att
        ids = np.arange(len(tri)) if select is None else np.asarray(select)
        self.faces = torch.as_tensor(faces[tri[ids]], dtype=torch.long)
        self.bary = torch.as_tensor(bary[ids])
        self.offset = torch.as_tensor(offset[ids])

    def evaluate(self, vertices, jacobian=None):
        a, b, c = vertices[:, self.faces].unbind(2)
        e1, e2 = b - a, c - a
        q = torch.cross(e1, e2, dim=-1)
        length = q.norm(dim=-1, keepdim=True)
        if torch.any(length < 1e-12):
            raise ValueError("Degenerate posed attachment triangle")
        n = q / length
        point = (vertices[:, self.faces] * self.bary.to(vertices)[None, :, :, None]).sum(2)
        point = point + self.offset.to(vertices)[None, :, None] * n
        if jacobian is None:
            return point
        ja, jb, jc = jacobian[:, self.faces].unbind(2)
        dq = -reference.skew(e2) @ (jb - ja) + reference.skew(e1) @ (jc - ja)
        dn = (dq - n[..., None] * (n[..., None] * dq).sum(-2, keepdim=True)) / length[..., None]
        jp = (jacobian[:, self.faces] * self.bary.to(vertices)[None, :, :, None, None]).sum(2)
        return point, jp + self.offset.to(vertices)[None, :, None, None] * dn


def global_rotation_jac(model, rotation, root, opt, joints, right=None):
    g = global_rotations(model, rotation, root)
    gj = g[:, joints] if right is None else g[:, joints] @ right
    # Every optimized ancestor rotates the global joint about its world axes.
    axes = g[:, opt].transpose(-1, -2)
    j = reference.skew(axes[:, None]) @ gj[:, :, None, None]
    ancestor = model.D[opt][:, joints].T.to(j)
    j = j * ancestor[None, :, :, None, None, None]
    j = j.permute(0, 1, 4, 5, 2, 3).reshape(len(rotation), 9 * len(joints), 3 * len(opt))
    return torch.cat([j, j.new_zeros(len(rotation), j.shape[1], 3)], -1)


def solve_lm(initial, apply, residual, jacobian, max_iter=16):
    """Per-frame LM; no silent failed solves or coupled batch acceptance."""
    state = tuple(x.clone() for x in initial)
    r = residual(state)
    if not torch.isfinite(r).all():
        raise ValueError("Non-finite initial residual")
    cost = r.double().square().sum(-1)
    damping = torch.full_like(cost, 1e-3)
    done = torch.zeros(len(cost), dtype=torch.bool)
    accepted = torch.zeros(len(cost), dtype=torch.int64)
    for _ in range(max_iter):
        j = jacobian(state).double()
        h = j.transpose(1, 2) @ j
        gradient = (j.transpose(1, 2) @ r.double()[..., None])[..., 0]
        diag = h.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        # Root/rotation units differ; scale the optimality check by curvature.
        done |= (gradient.abs() / diag.sqrt()).amax(-1) < 2e-7
        if done.all():
            break
        pending = ~done
        for _ in range(8):
            system = h + torch.diag_embed(damping[:, None] * diag)
            delta = -torch.linalg.solve(system, gradient[..., None])[..., 0]
            delta[~pending] = 0
            trial = apply(state, delta.to(state[0]))
            rr = residual(trial)
            cc = rr.double().square().sum(-1)
            if not torch.isfinite(cc).all():
                raise ValueError("Non-finite trial residual")
            ok = pending & (cc < cost)
            improvement = cost - cc
            for i in range(len(state)):
                shape = (-1,) + (1,) * (state[i].ndim - 1)
                state = state[:i] + (torch.where(ok.view(shape), trial[i], state[i]),) + state[i + 1:]
            r = torch.where(ok[:, None], rr, r)
            cost = torch.where(ok, cc, cost)
            accepted += ok
            done |= ok & (improvement < 1e-10 * cost.clamp_min(1e-8))
            damping = torch.where(ok, damping / 3, torch.where(pending, damping * 5, damping)).clamp(1e-10, 1e9)
            pending &= ~ok
            if not pending.any():
                break
        done |= damping >= 1e9
        if done.all():
            break
    return state, {"cost": cost.tolist(), "accepted_steps": accepted.tolist(), "iterations_limit": max_iter}


class BodyConverter:
    def __init__(self, smplx_model, rest_fit, forward_iterations=16, inverse_iterations=16):
        self.base = MF.MeshConverter(smplx_model, rest_fit)
        skeleton, skin = MF.soma_skin()
        self.fs = FastSoma(skin, skeleton)
        self.soma = reference.SomaModel(skin)
        self.smplx = BodySMPLX(self.base.sx, self.base.shape)
        self.xopt = list(range(22))
        # The 22 anatomical body joints, including wrists and toes; Neck2 and
        # face/finger slots are not free variables in the SOMA30 body contract.
        self.sopt = self.fs.slice30[MF.analytic.M[:22]].tolist()
        self.sj = self.fs.slice30[torch.tensor([1, 2, 3, 4])].tolist()
        self.xj = [3, 6, 9, 12]
        self.crest = global_rotations(self.smplx, MF.rodrigues(self.base.rest_pose[None]), torch.zeros(1, 3))[0, self.xj]
        self.body_x = torch.where(self.base.sx.W.argmax(1) < 22)[0].numpy()
        self.smplx_body = BodySMPLX(self.base.sx, self.base.shape, self.body_x)
        owner_s = skin.lbs_indices.gather(1, skin.lbs_weights.argmax(1, keepdim=True))[:, 0]
        self.body_s = torch.where(torch.isin(owner_s, torch.tensor(self.sopt)))[0].numpy()
        self.attachment = Attachment(self.base.SF, self.base.ax, self.body_x)
        self.back_attachment = Attachment(self.base.sx.faces, self.base.as_, self.body_s)
        self.forward_iterations = forward_iterations
        self.inverse_iterations = inverse_iterations
        self.last_diagnostics = {}

    def _soma_state(self, local, root):
        local = torch.as_tensor(local, dtype=torch.float32)
        root = torch.as_tensor(root, dtype=torch.float32)
        if local.ndim != 4 or local.shape[1:] != (30, 3, 3) or root.shape != (len(local), 3):
            raise ValueError("Expected SOMA local [T,30,3,3], root [T,3]")
        if not torch.isfinite(local).all() or not torch.isfinite(root).all():
            raise ValueError("Non-finite SOMA input")
        if ((local.transpose(-1, -2) @ local - torch.eye(3)).abs().max() > 2e-4
                or torch.linalg.det(local).min() < .9998):
            raise ValueError("SOMA input contains non-SO(3) rotation matrices")
        if (local[:, 5] - torch.eye(3)).abs().max() > 1e-5:
            raise ValueError("Non-neutral SOMA Neck2 is outside this frozen body contract")
        # Keep the native relaxed hand template. Set only unused SOMA30 slots
        # neutral, avoiding the frozen-initializer bug in the earlier solver.
        full = self.fs.to77(local)
        unused = [int(j) for j in self.fs.slice30 if int(j) not in self.sopt]
        full[:, unused] = torch.eye(3)
        return full, root

    def _smplx_state(self, poses, translation):
        poses = torch.as_tensor(poses, dtype=torch.float32)
        translation = torch.as_tensor(translation, dtype=torch.float32)
        if poses.shape[1:] == (165,):
            poses = poses.reshape(-1, 55, 3)
        if poses.ndim != 3 or poses.shape[1:] != (55, 3) or translation.shape != (len(poses), 3):
            raise ValueError("Expected SMPL-X poses [T,55,3] and translation [T,3]")
        if not torch.isfinite(poses).all() or not torch.isfinite(translation).all():
            raise ValueError("Non-finite SMPL-X input")
        rotation = MF.rodrigues(poses)
        rotation[:, 22:] = torch.eye(3)
        return rotation, translation

    def _forward(self, state, initial=None):
        local, root = state
        target = self.attachment.evaluate(self.soma.verts(local, root))
        target_g = global_rotations(self.soma, local, root)[:, self.sj] @ self.crest
        if initial is None:
            r25, tr, _ = self.base.cv.soma_to_smplx(local[:, self.fs.slice30].numpy(), root.numpy())
            rotation = torch.eye(3).repeat(len(local), 55, 1, 1)
            rotation[:, :22] = torch.tensor(r25[:, :22]).float()
            trans = torch.tensor(tr + self.base.cv.sx[0] - self.base.J[0].numpy()).float()
            initial = rotation, trans
        def residual(x):
            v = self.smplx_body.verts(*x) - target
            g = global_rotations(self.smplx, *x)[:, self.xj] - target_g
            return torch.cat([v.flatten(1), g.flatten(1)], -1)
        def jac(x):
            _, j = self.smplx_body.jac(*x, self.xopt)
            return torch.cat([j.flatten(1, 2), global_rotation_jac(self.smplx, *x, self.xopt, self.xj)], 1)
        result, self.last_diagnostics = solve_lm(initial, lambda x, d: self.smplx.apply(*x, self.xopt, d), residual, jac, self.forward_iterations)
        return result

    def _initial_inverse(self, x):
        r, tr = x
        tr0 = tr.numpy() + self.base.J[0].numpy() - self.base.cv.sx[0]
        local, root = self.base.cv.smplx_to_soma(r[:, :25].numpy().astype(np.float64), tr0.astype(np.float64))
        initial = self._soma_state(local, root)
        target = self.back_attachment.evaluate(self.smplx.verts(*x))
        target_g = global_rotations(self.smplx, *x)[:, self.xj] @ self.crest.transpose(-1, -2)
        def residual(s):
            v = self.soma.verts(*s)[:, self.body_s] - target
            g = global_rotations(self.soma, *s)[:, self.sj] - target_g
            return torch.cat([v.flatten(1), g.flatten(1)], -1)
        def jac(s):
            _, j = self.soma.jac(*s, self.sopt)
            return torch.cat([j[:, self.body_s].flatten(1, 2), global_rotation_jac(self.soma, *s, self.sopt, self.sj)], 1)
        state, _ = solve_lm(initial, lambda s, d: self.soma.apply(*s, self.sopt, d), residual, jac, 12)
        return state

    def _inverse(self, x, direct=False):
        initial = self._initial_inverse(x)
        if direct:
            return initial
        r, tr = x
        vertices, j = self.smplx_body.jac(r, tr, self.xopt)
        jx = j.flatten(1, 2)
        jg = global_rotation_jac(self.smplx, r, tr, self.xopt, self.xj)
        jp = torch.cat([jx, jg], 1).double()
        # A fixed projection of the exact forward stationarity residual.
        # Whiten by J'J; no repeated forward fit and no approximation of dF/dθ.
        h = jp.transpose(1, 2) @ jp
        chol = torch.linalg.cholesky(h)
        projection = torch.linalg.solve_triangular(chol, jp.transpose(1, 2), upper=False)
        target_g = global_rotations(self.smplx, r, tr)[:, self.xj]
        def residual(s):
            v = vertices - self.attachment.evaluate(self.soma.verts(*s))
            g = target_g - global_rotations(self.soma, *s)[:, self.sj] @ self.crest
            return (projection @ torch.cat([v.flatten(1), g.flatten(1)], -1).double()[..., None])[..., 0]
        def jac(s):
            v, j = self.soma.jac(*s, self.sopt)
            _, jt = self.attachment.evaluate(v, j)
            jg = global_rotation_jac(self.soma, *s, self.sopt, self.sj, self.crest)
            return -(projection @ torch.cat([jt.flatten(1, 2), jg], 1).double())
        result, self.last_diagnostics = solve_lm(initial, lambda s, d: self.soma.apply(*s, self.sopt, d), residual, jac, self.inverse_iterations)
        return result

    @torch.no_grad()
    def soma_to_smplx(self, local, root, hands=None, chunk=2):
        state = self._soma_state(local, root)
        rr, tt = [], []
        for start in range(0, len(root), chunk):
            r, t = self._forward(tuple(x[start:start + chunk] for x in state))
            rr.append(r); tt.append(t)
        poses = MF.rotvec(torch.cat(rr).numpy()).astype(np.float32)
        if hands is not None:
            hands = np.asarray(hands)
            if hands.shape != (len(poses), 30, 3) or not np.isfinite(hands).all():
                raise ValueError("Hand passthrough must be finite [T,30,3]")
            poses = poses.astype(np.result_type(poses.dtype, hands.dtype), copy=False)
            poses[:, 25:] = hands
        return poses, torch.cat(tt).numpy()

    @torch.no_grad()
    def smplx_to_soma(self, poses, translation, chunk=2, direct=False):
        state = self._smplx_state(poses, translation)
        ll, rr = [], []
        for start in range(0, len(translation), chunk):
            l, r = self._inverse(tuple(x[start:start + chunk] for x in state), direct=direct)
            ll.append(l[:, self.fs.slice30]); rr.append(r)
        return torch.cat(ll).numpy(), torch.cat(rr).numpy()
