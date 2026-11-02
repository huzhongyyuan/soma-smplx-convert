"""SOMA30 <-> SMPL-X conversion by mesh-surface fitting.

SOMA and SMPL-X place their joints at different anatomical points (hip width ~8 cm apart, different spine
heights), so copying bone directions (soma_smplx.analytic) leaves several cm of hand/shoulder offset.
Here the SMPL-X *surface* is fitted to the SOMA surface instead:
  1. rest: fit SMPL-X betas/expression/pose to the SOMA rest mesh (ICP, point-to-plane), then attach every
     SMPL-X vertex to its closest point on the SOMA rest surface (triangle, barycentric, normal offset)
     and every SOMA vertex to the SMPL-X rest surface.  Saved once with fit_rest().
  2. per frame SOMA->SMPL-X: carry the attachments onto the posed SOMA mesh and solve SMPL-X global orient,
     body pose, hand pose and translation (betas fixed) by L-BFGS, initialised from the analytic converter.
  3. per frame SMPL-X->SOMA: same with the reverse attachments, solving SOMA30 local rotations and root.
Error is reported as vertex-to-surface distance; the two meshes are different shapes, so it is never zero.
"""
import json
import numpy as np
import torch
from . import analytic

N_BETAS, N_EXPR = 20, 10
DT = torch.float32


def rodrigues(r):
    th = r.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    k = r / th; th = th[..., None]
    z = torch.zeros_like(k[..., 0])
    K = torch.stack([z, -k[..., 2], k[..., 1], k[..., 2], z, -k[..., 0], -k[..., 1], k[..., 0], z], -1).reshape(r.shape[:-1] + (3, 3))
    I = torch.eye(3, dtype=r.dtype).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K


def rotvec(R):
    from scipy.spatial.transform import Rotation
    sh = R.shape[:-2]
    return Rotation.from_matrix(np.asarray(R, np.float64).reshape(-1, 3, 3)).as_rotvec().reshape(sh + (3,))


class SMPLX:
    """Full 55-joint SMPL-X LBS in torch (shape + expression + pose blendshapes)."""
    def __init__(self, smplx_npz):
        z = np.load(smplx_npz, allow_pickle=True)
        t = lambda a: torch.tensor(np.asarray(a, np.float64), dtype=DT)
        self.vt = t(z['v_template']); self.faces = np.asarray(z['f'], np.int64)
        sd = np.asarray(z['shapedirs'])
        self.sdirs = t(np.concatenate([sd[..., :N_BETAS], sd[..., 300:300 + N_EXPR]], -1))
        self.Jreg = t(z['J_regressor']); self.W = t(z['weights'])
        self.pdirs = t(np.asarray(z['posedirs']).reshape(-1, 486))
        p = np.asarray(z['kintree_table'])[0].astype(np.int64); p[0] = -1; self.parents = p

    def shaped(self, shape):                       # shape [N_BETAS+N_EXPR]
        v = self.vt + self.sdirs @ shape
        return v, self.Jreg @ v

    def __call__(self, shape, R, transl):            # R [T,55,3,3] local
        v0, J = self.shaped(shape); T = R.shape[0]
        I = torch.eye(3, dtype=DT)
        v = v0[None] + ((R[:, 1:] - I).reshape(T, -1) @ self.pdirs.T).reshape(T, -1, 3)
        Gr, Gt = [None] * 55, [None] * 55
        for j, p in enumerate(self.parents):
            if p < 0: Gr[j], Gt[j] = R[:, j], J[j].expand(T, 3)
            else:
                Gr[j] = Gr[p] @ R[:, j]; Gt[j] = (Gr[p] @ (J[j] - J[p])[:, None])[..., 0] + Gt[p]
        Gr = torch.stack(Gr, 1); Gt = torch.stack(Gt, 1)
        At = Gt - (Gr @ J[..., None])[..., 0]
        Mr = torch.einsum('vj,tjab->tvab', self.W, Gr); Mt = torch.einsum('vj,tja->tva', self.W, At)
        verts = (Mr @ v[..., None])[..., 0] + Mt + transl[:, None]
        return verts, Gt + transl[:, None]


_SOMA = {}
def soma_skin():
    """(SOMASkeleton30, SOMASkin) from the ARDY package (https://github.com/nv-tlabs/ardy)."""
    if not _SOMA:
        from ardy.skeleton import SOMASkeleton30
        from ardy.viz.soma_skin import SOMASkin
        sk = SOMASkeleton30().float()
        _SOMA['sk'], _SOMA['skin'] = sk, SOMASkin(sk)
    return _SOMA['sk'], _SOMA['skin']


def soma_faces():
    return soma_skin()[1].faces.numpy()


def soma_verts_t(local_t, root_t):
    """differentiable SOMA mesh; local_t [T,30,3,3], root_t [T,3]"""
    jp = torch.zeros(local_t.shape[0], 30, 3, dtype=DT); jp[:, 0] = root_t
    return soma_skin()[1].skin(local_t, jp)


def soma_verts(local, root):
    with torch.no_grad():
        return soma_verts_t(torch.as_tensor(np.asarray(local), dtype=DT), torch.as_tensor(np.asarray(root), dtype=DT)).numpy()


def face_normals(v, f):
    a, b, c = v[..., f[:, 0], :], v[..., f[:, 1], :], v[..., f[:, 2], :]
    n = torch.cross(b - a, c - a, dim=-1)
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def closest_on_mesh(points, verts, faces, k=24):
    """exact closest point among faces touching the k nearest vertices (no rtree dependency)"""
    import trimesh
    from scipy.spatial import cKDTree
    vf = [[] for _ in range(len(verts))]
    for fi, f in enumerate(faces):
        for v in f: vf[v].append(fi)
    L = max(len(x) for x in vf); VF = np.full((len(verts), L), -1, np.int64)
    for v, x in enumerate(vf): VF[v, :len(x)] = x
    _, nn = cKDTree(verts).query(points, k=k)
    cand = VF[nn].reshape(len(points), -1)
    best_d = np.full(len(points), np.inf); best_p = np.zeros_like(points); best_f = np.zeros(len(points), np.int64)
    tris = verts[faces]
    for c in range(cand.shape[1]):
        f = cand[:, c]; ok = f >= 0
        cp = trimesh.triangles.closest_point(tris[f[ok]], points[ok])
        d = np.linalg.norm(cp - points[ok], axis=1)
        idx = np.where(ok)[0]; better = d < best_d[idx]
        best_d[idx[better]] = d[better]; best_p[idx[better]] = cp[better]; best_f[idx[better]] = f[ok][better]
    return best_p, best_d, best_f


def surface_dist(points, verts, faces):
    return closest_on_mesh(np.asarray(points, np.float64), np.asarray(verts, np.float64), faces)[1]


def attach(points, verts, faces):
    """closest point on mesh -> (triangle id, barycentric, normal offset)"""
    import trimesh
    points = np.asarray(points, np.float64); verts = np.asarray(verts, np.float64)
    cp, dist, tri = closest_on_mesh(points, verts, faces)
    bary = trimesh.triangles.points_to_barycentric(verts[faces[tri]], cp)
    a, b, c = verts[faces[tri, 0]], verts[faces[tri, 1]], verts[faces[tri, 2]]
    n = np.cross(b - a, c - a); n /= np.linalg.norm(n, axis=1, keepdims=True).clip(1e-12)
    off = ((points - cp) * n).sum(1)
    return tri.astype(np.int64), bary.astype(np.float32), off.astype(np.float32)


def carry(verts_t, faces, att):
    """attachment points on posed mesh(es); verts_t [T,V,3] torch"""
    tri, bary, off = att
    f = torch.as_tensor(faces[tri]); b = torch.as_tensor(bary); o = torch.as_tensor(off)
    p = (verts_t[:, f] * b[None, ..., None]).sum(2)
    n = face_normals(verts_t, faces[tri])
    return p + o[None, :, None] * n


def fit_rest(smplx_npz, out_npz, log=print):
    """Fit SMPL-X to the SOMA rest mesh and save shape + attachments to out_npz (run once, a few minutes on CPU)."""
    from scipy.spatial import cKDTree
    sk, _ = soma_skin()
    S = soma_verts_t(torch.eye(3, dtype=DT).repeat(1, 30, 1, 1), sk.neutral_joints[0:1].to(DT))[0].detach()
    SF = soma_faces()
    fn = face_normals(S[None], SF)[0].numpy(); vn = np.zeros_like(S.numpy())
    for k in range(3): np.add.at(vn, SF[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True).clip(1e-12); vn = torch.tensor(vn)
    tree = cKDTree(S.numpy())
    sx = SMPLX(smplx_npz)
    shape = torch.zeros(N_BETAS + N_EXPR, dtype=DT, requires_grad=True)
    pose = torch.zeros(55, 3, dtype=DT); pose[0, 0] = 1e-4; pose = pose.requires_grad_(True)
    _, J0 = sx.shaped(shape.detach())
    transl = (-J0[0]).clone().requires_grad_(True)   # SOMA Hips at origin
    opt = torch.optim.Adam([shape, pose, transl], lr=0.01)
    for outer in range(60):
        with torch.no_grad():
            V, _ = sx(shape, rodrigues(pose)[None], transl[None]); V = V[0].numpy()
        _, nn = tree.query(V); nn = torch.as_tensor(nn)
        _, nn2 = cKDTree(V).query(S.numpy()); nn2 = torch.as_tensor(nn2)
        for it in range(40):
            V, _ = sx(shape, rodrigues(pose)[None], transl[None]); V = V[0]
            d = V - S[nn]
            l_plane = ((d * vn[nn]).sum(-1) ** 2).mean()
            l_pt = (d ** 2).sum(-1).mean()
            l_rev = ((S - V[nn2]) ** 2).sum(-1).mean()
            reg = 1e-5 * (shape ** 2).sum() + 1e-5 * (pose[1:] ** 2).sum()
            loss = l_plane + 0.1 * l_pt + l_rev + reg
            opt.zero_grad(); loss.backward(); opt.step()
        if outer % 10 == 9:
            log(f'rest ICP {outer+1}: plane {l_plane.sqrt()*1000:.1f} mm  pt {l_pt.sqrt()*1000:.1f}  rev {l_rev.sqrt()*1000:.1f}  |b| {shape[:N_BETAS].norm():.2f}', flush=True)
    with torch.no_grad():
        V, _ = sx(shape, rodrigues(pose)[None], transl[None]); V = V[0].numpy()
    a_x = attach(V, S.numpy(), SF)        # SMPL-X vertex -> SOMA surface
    a_s = attach(S.numpy(), V, sx.faces)  # SOMA vertex -> SMPL-X surface
    np.savez(out_npz, shape=shape.detach().numpy(), rest_pose=pose.detach().numpy(), rest_transl=transl.detach().numpy(),
             smplx_rest_verts=V, soma_rest_verts=S.numpy(),
             ax_tri=a_x[0], ax_bary=a_x[1], ax_off=a_x[2], as_tri=a_s[0], as_bary=a_s[1], as_off=a_s[2])
    mm = lambda a: dict(mean=round(float(np.abs(a).mean() * 1000), 2), p95=round(float(np.percentile(np.abs(a), 95) * 1000), 2),
                        max=round(float(np.abs(a).max() * 1000), 2))
    st = dict(betas_norm=float(shape[:N_BETAS].norm()), smplx_to_soma_surface_mm=mm(a_x[2]), soma_to_smplx_surface_mm=mm(a_s[2]))
    log(json.dumps(st)); return st


class MeshConverter:
    def __init__(self, smplx_npz, rest_npz):
        """smplx_npz: SMPLX_NEUTRAL_2020.npz; rest_npz: output of fit_rest()."""
        z = np.load(rest_npz)
        self.shape = torch.tensor(z['shape'], dtype=DT); self.rest_pose = torch.tensor(z['rest_pose'], dtype=DT)
        self.ax = (z['ax_tri'], z['ax_bary'], z['ax_off']); self.as_ = (z['as_tri'], z['as_bary'], z['as_off'])
        self.sx = SMPLX(smplx_npz); self.SF = soma_faces()
        _, self.J = self.sx.shaped(self.shape)
        self.cv = analytic.Converter(smplx_npz)

    def smplx_verts(self, pose55, transl):
        with torch.no_grad():
            return self.sx(self.shape, rodrigues(torch.as_tensor(pose55, dtype=DT)), torch.as_tensor(transl, dtype=DT))[0].numpy()

    def soma_to_smplx(self, local, root, chunk=64, iters=60):
        """local [T,30,3,3], root [T,3] -> pose55 axis-angle [T,55,3], transl [T,3] (SMPL-X shape = self.shape)"""
        rot25, transl0, _ = self.cv.soma_to_smplx(local, root)
        pose_init = np.zeros((len(local), 55, 3)); pose_init[:, :25] = rotvec(rot25)
        pose_init[:, 25:55] = self.rest_pose[25:55].numpy()
        tr_init = transl0 + self.cv.sx[0] - self.J[0].numpy()
        out_p, out_t = [], []
        for s in range(0, len(local), chunk):
            with torch.no_grad():
                tgt = carry(soma_verts_t(torch.as_tensor(local[s:s+chunk], dtype=DT), torch.as_tensor(root[s:s+chunk], dtype=DT)), self.SF, self.ax)
            p = torch.tensor(pose_init[s:s+chunk], dtype=DT, requires_grad=True)
            t = torch.tensor(tr_init[s:s+chunk], dtype=DT, requires_grad=True)
            opt = torch.optim.LBFGS([p, t], lr=1, max_iter=iters, line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
            def closure():
                opt.zero_grad()
                V, _ = self.sx(self.shape, rodrigues(p), t)
                loss = ((V - tgt) ** 2).sum(-1).mean() * 1e4
                loss.backward(); return loss
            opt.step(closure); opt.step(closure)
            out_p.append(p.detach().numpy()); out_t.append(t.detach().numpy())
        return np.concatenate(out_p), np.concatenate(out_t)

    def smplx_to_soma(self, pose55, transl, chunk=64, iters=60):
        """pose55 [T,55,3] axis-angle, transl [T,3] -> local [T,30,3,3], root [T,3]"""
        Rx = rodrigues(torch.as_tensor(pose55, dtype=DT)).numpy()
        tr_v2 = transl + self.J[0].numpy() - self.cv.sx[0]
        local0, root0 = self.cv.smplx_to_soma(Rx[:, :25].astype(np.float64), tr_v2.astype(np.float64))
        out_l, out_r = [], []
        for s in range(0, len(pose55), chunk):
            with torch.no_grad():
                V, _ = self.sx(self.shape, torch.as_tensor(Rx[s:s+chunk]), torch.as_tensor(transl[s:s+chunk], dtype=DT))
                tgt = carry(V, self.sx.faces, self.as_)
            q = torch.tensor(rotvec(local0[s:s+chunk]), dtype=DT, requires_grad=True)
            r = torch.tensor(root0[s:s+chunk], dtype=DT, requires_grad=True)
            opt = torch.optim.LBFGS([q, r], lr=1, max_iter=iters, line_search_fn='strong_wolfe', tolerance_grad=1e-9, tolerance_change=1e-12)
            def closure():
                opt.zero_grad()
                Vs = soma_verts_t(rodrigues(q), r)
                loss = ((Vs - tgt) ** 2).sum(-1).mean() * 1e4
                loss.backward(); return loss
            opt.step(closure); opt.step(closure)
            out_l.append(rodrigues(q.detach()).numpy()); out_r.append(r.detach().numpy())
        return np.concatenate(out_l), np.concatenate(out_r)
