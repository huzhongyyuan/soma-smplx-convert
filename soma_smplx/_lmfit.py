# SPDX-License-Identifier: Apache-2.0
# Body release packaging, 2026-11-02; numerical solver retained from the verified snapshot.
"""Per-frame Levenberg-Marquardt fit of a skinned mesh to target points, with exact analytic Jacobians.
Parameters per frame: right-multiplied local rotation increments of the chosen joints (L_k <- L_k exp(eps_k)) + root translation.
Joint k's world-frame increment delta_k = G_k eps_k rotates its whole subtree about p_k: dv = delta_k x (y_vk - W_vk p_k),
y_vk = sum over influences j in subtree(k) of w_vj (T_j b_v), W_vk = sum of those weights.  SMPL-X adds the pose-corrective term."""
import numpy as np, torch
from ardy.skeleton.kinematics import batch_rigid_transform


def skew(a):
    z = torch.zeros_like(a[..., 0])
    return torch.stack([z, -a[..., 2], a[..., 1], a[..., 2], z, -a[..., 0], -a[..., 1], a[..., 0], z], -1).reshape(a.shape + (3,))


def expm(e):
    th = e.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    K = skew(e / th); th = th[..., None]
    return torch.eye(3, dtype=e.dtype) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def subtree(parents):
    n = len(parents); D = torch.zeros(n, n)
    for j in range(n):
        k = j
        while k >= 0: D[k, j] = 1; k = int(parents[k])
    return D


class SomaModel:
    """SOMA77 LBS (identical to ardy SOMASkin.skin for local rotations)."""
    def __init__(self, skin):
        s = skin.skeleton_skin
        self.parents = s.joint_parents.long(); self.root_idx = s.root_idx; self.nj = s.neutral_joints.float()
        self.bind_inv = skin.bind_rig_transform_inv.float(); self.bv = skin.bind_vertices.float()
        self.idx = skin.lbs_indices.long(); self.w = skin.lbs_weights.float(); self.D = subtree(self.parents.tolist())
        self.nJ = 77

    def joints(self, L, root):
        jp, G = batch_rigid_transform(L, self.nj[None].expand(L.shape[0], -1, 3), self.parents, self.root_idx)
        return jp + root[:, None], G

    def parts(self, L, root):   # u [T,V,8,3] = T_j b_v for each influence, G, p
        p, G = self.joints(L, root)
        A = G @ self.bind_inv[:, :3, :3]; t = (G @ self.bind_inv[:, :3, 3:])[..., 0] + p
        u = torch.einsum('tviab,vb->tvia', A[:, self.idx], self.bv) + t[:, self.idx]
        return u, G, p

    def verts(self, L, root):
        u, _, _ = self.parts(L, root); return (u * self.w[None, ..., None]).sum(2)

    def jac(self, L, root, opt):  # -> V [T,V,3], J [T,V,3,3K+3]
        u, G, p = self.parts(L, root); V = (u * self.w[None, ..., None]).sum(2)
        Dw = self.D[opt][:, self.idx] * self.w[None]                    # [K,V,8]
        y = torch.einsum('kvi,tvia->tvka', Dw, u); Wk = Dw.sum(-1).T      # [V,K]
        a = y - Wk[None, ..., None] * p[:, opt][:, None]
        Jr = -skew(a) @ G[:, opt][:, None]                               # [T,V,K,3,3]
        T, Vn, K = a.shape[:3]
        J = torch.cat([Jr.permute(0, 1, 3, 2, 4).reshape(T, Vn, 3, 3 * K), torch.eye(3).expand(T, Vn, 3, 3)], -1)
        return V, J

    def apply(self, L, root, opt, step):
        K = len(opt); e = step[:, :3 * K].reshape(-1, K, 3)
        L = L.clone(); L[:, opt] = L[:, opt] @ expm(e); return L, root + step[:, 3 * K:]


class SmplxModel:
    """SMPL-X LBS with fixed shape; same maths as soma_smplx.meshfit.SMPLX."""
    def __init__(self, sx, shape):
        self.sx = sx; self.v0, self.J0 = sx.shaped(shape); self.W = sx.W; self.parents = sx.parents
        self.P = sx.pdirs.reshape(-1, 3, 54, 3, 3)                         # [V,3,joint-1,3,3]
        self.D = subtree(list(self.parents)); self.nJ = 55

    def parts(self, R, transl):
        T = R.shape[0]; I = torch.eye(3)
        vp = self.v0[None] + torch.einsum('vcjab,tjab->tvc', self.P, R[:, 1:] - I)
        Gr, Gt = [None] * 55, [None] * 55
        for j, q in enumerate(self.parents):
            if q < 0: Gr[j], Gt[j] = R[:, j], self.J0[j].expand(T, 3)
            else: Gr[j] = Gr[q] @ R[:, j]; Gt[j] = (Gr[q] @ (self.J0[j] - self.J0[q])[:, None])[..., 0] + Gt[q]
        Gr = torch.stack(Gr, 1); Gt = torch.stack(Gt, 1) + transl[:, None]
        At = Gt - (Gr @ self.J0[..., None])[..., 0]
        return vp, Gr, At, Gt

    def verts(self, R, transl):
        vp, Gr, At, _ = self.parts(R, transl)
        Mr = torch.einsum('vj,tjab->tvab', self.W, Gr); Mt = torch.einsum('vj,tja->tva', self.W, At)
        return (Mr @ vp[..., None])[..., 0] + Mt

    def jac(self, R, transl, opt):
        vp, Gr, At, p = self.parts(R, transl)
        u = torch.einsum('tjab,tvb->tvja', Gr, vp) + At[:, None]            # [T,V,55,3]
        V = (u * self.W[None, ..., None]).sum(2)
        DW = self.D[opt][:, None, :] * self.W[None]                       # [K,V,55]
        y = torch.einsum('kvj,tvja->tvka', DW, u); Wk = DW.sum(-1).T
        a = y - Wk[None, ..., None] * p[:, opt][:, None]
        Jr = -skew(a) @ Gr[:, opt][:, None]                              # [T,V,K,3,3]
        # pose correctives: d(R_k - I)/d eps_e = R_k [e]x ; vertex moves by Mr_v @ d vp
        Mr = torch.einsum('vj,tjab->tvab', self.W, Gr)
        E = skew(torch.eye(3))                                           # [3(e),3,3]
        for i, k in enumerate(opt):
            if k == 0: continue
            dR = R[:, k][:, None] @ E[None]                               # [T,3,3,3]
            dvp = torch.einsum('vcab,teab->tvce', self.P[:, :, k - 1], dR)  # [T,V,3,3]
            Jr[:, :, i] += Mr @ dvp
        T, Vn, K = a.shape[:3]
        J = torch.cat([Jr.permute(0, 1, 3, 2, 4).reshape(T, Vn, 3, 3 * K), torch.eye(3).expand(T, Vn, 3, 3)], -1)
        return V, J

    def apply(self, R, transl, opt, step):
        K = len(opt); e = step[:, :3 * K].reshape(-1, K, 3)
        R = R.clone(); R[:, opt] = R[:, opt] @ expm(e); return R, transl + step[:, 3 * K:]


def glob_rots(model, R, root):
    return model.joints(R, root)[1] if hasattr(model, "bind_inv") else model.parts(R, root)[1]


def lm_fit(model, R, root, tgt, opt, iters=50, tol=1e-12, reg=None, wvert=None, chunk=8, log=None, rot=None):
    """Minimise sum_v w_v |V_v - tgt_v|^2 (+ reg: (weight, joint ids, target local rots) keeps those joints near a pose).
    rot: (weight, joint ids, target GLOBAL rots [T,n,3,3]) adds weight * |G_j - target|_F^2 (anatomical correspondence term).
    Returns R, root, per-frame final loss, per-frame iterations used."""
    outR, outr, outl, outn = [], [], [], []
    w = None if wvert is None else torch.as_tensor(wvert, dtype=torch.float32)[None, :, None]
    for s in range(0, len(R), chunk):
        Rc, rc, tc = R[s:s + chunk].clone(), root[s:s + chunk].clone(), tgt[s:s + chunk]
        T = len(Rc); lam = torch.full((T,), 1e-3); done = torch.zeros(T, dtype=torch.bool); nit = torch.zeros(T, dtype=torch.long)

        def loss_of(Rx, rx):
            r = model.verts(Rx, rx) - tc
            l = ((r ** 2) * (1 if w is None else w)).sum((1, 2))
            if reg is not None:
                rw, rj, rt = reg; l = l + rw * ((Rx[:, rj] - rt[s:s + chunk]) ** 2).sum((1, 2, 3))
            if rot is not None:
                ow, oj, ot = rot; l = l + ow * ((glob_rots(model, Rx, rx)[:, oj] - ot[s:s + chunk]) ** 2).sum((1, 2, 3))
            return l
        cur = loss_of(Rc, rc)
        for it in range(iters):
            V, J = model.jac(Rc, rc, opt); r = V - tc
            if w is not None: J = J * w[..., None].sqrt(); r = r * w.sqrt()
            Jf = J.reshape(T, -1, J.shape[-1]); H = Jf.transpose(1, 2) @ Jf; g = Jf.transpose(1, 2) @ r.reshape(T, -1, 1)
            if reg is not None:   # Gauss-Newton term for |R_j - R_t|^2 with R_j <- R_j exp(eps): dR = R_j [e]x
                rw, rj, rt = reg; E = skew(torch.eye(3)); K = len(opt)
                for jj in rj:
                    i = opt.index(jj); Rj = Rc[:, jj]; dR = (Rj[:, None] @ E[None]).reshape(T, 3, 9).transpose(1, 2)  # [T,9,3]
                    res = (Rj - rt[s:s + chunk][:, rj.index(jj)]).reshape(T, 9, 1)
                    H[:, 3 * i:3 * i + 3, 3 * i:3 * i + 3] += rw * dR.transpose(1, 2) @ dR
                    g[:, 3 * i:3 * i + 3] += rw * dR.transpose(1, 2) @ res
            if rot is not None:   # dG_j / d eps_a = [G_a e_c]x G_j for every optimised ancestor a of j
                ow, oj, ot = rot; Gg = glob_rots(model, Rc, rc); K = len(opt); E = skew(torch.eye(3))
                for n, jj in enumerate(oj):
                    Jn = torch.zeros(T, 9, J.shape[-1]); Gj = Gg[:, jj]
                    for i, a in enumerate(opt):
                        if model.D[a, jj] == 0: continue
                        ax = torch.einsum("tab,cb->tca", Gg[:, a], torch.eye(3))            # world axes G_a e_c  [T,3,3]
                        Jn[:, :, 3 * i:3 * i + 3] = (skew(ax) @ Gj[:, None]).reshape(T, 3, 9).transpose(1, 2)
                    res = (Gj - ot[s:s + chunk][:, n]).reshape(T, 9, 1)
                    H = H + ow * Jn.transpose(1, 2) @ Jn; g = g + ow * Jn.transpose(1, 2) @ res
            dg = torch.diagonal(H, dim1=1, dim2=2)
            for _ in range(10):
                A = (H + torch.diag_embed(lam[:, None] * (dg + 1e-9))).double()
                step = -torch.cholesky_solve(g.double(), torch.linalg.cholesky(A))[..., 0].float()
                step[done] = 0
                Rn, rn = model.apply(Rc, rc, opt, step); new = loss_of(Rn, rn)
                ok = (new < cur) & ~done
                Rc[ok], rc[ok] = Rn[ok], rn[ok]
                rel = (cur - new) / cur.clamp_min(1e-20)
                done = done | (ok & (rel < tol))
                cur = torch.where(ok, new, cur); nit += (~done).long()
                lam = torch.where(ok, lam / 3, lam * 5).clamp(1e-9, 1e9)
                if ok.any() or done.all(): break
            done = done | (lam >= 1e9)
            if done.all(): break
        outR.append(Rc); outr.append(rc); outl.append(cur); outn.append(nit)
        if log: log(f'chunk {s}: loss {cur.mean():.3e} iters {nit.float().mean():.1f}')
    return torch.cat(outR), torch.cat(outr), torch.cat(outl), torch.cat(outn)
