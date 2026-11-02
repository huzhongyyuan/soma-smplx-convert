"""Analytic SOMA30 <-> SMPL-X body conversion (bone-direction transfer).

Per SMPL-X joint j with SOMA counterpart k=M[j]:  G_smplx[j] = G_soma[k] @ C[j].
C[j] is a constant rest-pose correction that turns the SMPL-X rest bone direction(s)
of j onto the SOMA rest bone direction(s) (single child: minimal swing; several
children: Kabsch); leaves inherit their parent's C so the rest pose maps to rest.
The SMPL-X neck takes SOMA Neck1 (same height above the pelvis); SOMA Neck2 has no
SMPL-X joint, so the neck gets an extra per-frame swing D that puts the SMPL-X
neck->head bone on the posed SOMA Neck1->Head chord (head lands where SOMA's is).

Shape: Converter fits SMPL-X shape betas (Converter.betas) so the limb lengths, hip /
shoulder widths, pelvis->neck, neck->head and eye spacing equal the SOMA ones.  The
SOMA spine joints sit at other heights than SMPL-X spine1-3, so those three joints
are not position-matched (their bone directions still are).

Sidecar (SOMA local rotations of Neck1, Neck2 and the four hand end joints) makes
SOMA->SMPL-X->SOMA exact; SMPL-X->SOMA->SMPL-X is exact without it (Neck2 = identity).

This converter only transfers rotations; joint positions of the two skeletons still
differ by several cm (different anatomical joint placement).  It is used to
initialise the mesh-fit converter in soma_smplx.meshfit.
"""
from __future__ import annotations
import numpy as np

SOMA_PARENTS = [-1, 0, 1, 2, 3, 4, 5, 6, 6, 6, 3, 10, 11, 12, 13, 13, 3, 16, 17, 18, 19, 19, 0, 22, 23, 24, 0, 26, 27, 28]
SMPLX_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15]
# SMPL-X body/face joint -> SOMA30 joint (neck = Neck1)
M = np.array([0, 22, 26, 1, 23, 27, 2, 24, 28, 3, 25, 29, 4, 10, 16, 6, 11, 17, 12, 18, 13, 19, 7, 8, 9])
NECK, HEAD, S_NECK1, S_NECK2, S_HEAD = 12, 15, 4, 5, 6
SIDECAR = [4, 5, 14, 15, 20, 21]     # SOMA Neck1, Neck2, L thumb end, L middle end, R thumb end, R middle end
NO_ALIGN = {22}                      # SMPL-X jaw sits on the head joint: not a bone direction
# alignment children: SMPL-X child joints and the SOMA joints they correspond to
# (wrists use the middle finger root 28/43 <-> SOMA HandMiddleEnd 15/21)
EXTRA = {20: [(28, 15)], 21: [(43, 21)]}
N_BETAS = 16
BETA_REG = 3e-3
# position-matched landmark pairs (pose-invariant in both bodies): limbs, widths, trunk height, head
SHAPE_PAIRS = [(0, 1), (0, 2), (1, 2), (1, 4), (2, 5), (4, 7), (5, 8), (7, 10), (8, 11), (0, 12), (12, 16), (12, 17),
               (16, 17), (16, 18), (17, 19), (18, 20), (19, 21), (12, 15), (15, 23), (15, 24), (23, 24)]
LANDMARKS = [0, 1, 2, 4, 5, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 23, 24]   # all but spine1-3, collars, jaw


def _rot_between(a, b):
    a = a/np.linalg.norm(a); b = b/np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0: return np.eye(3)
        raise ValueError('antiparallel rest bones')
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K@K/(1+c)


def _swing(a, b):
    """Batched minimal rotation a[T,3] -> b[T,3]."""
    a = a/np.linalg.norm(a, axis=-1, keepdims=True); b = b/np.linalg.norm(b, axis=-1, keepdims=True)
    v = np.cross(a, b); c = (a*b).sum(-1)
    K = np.zeros(a.shape[:-1]+(3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -v[..., 2], v[..., 1], -v[..., 0]
    K = K - K.swapaxes(-1, -2)
    return np.eye(3) + K + K@K/(1+c)[..., None, None]


def _kabsch(A, Bv):
    """R with R @ A[i] ~ Bv[i] (unit directions)."""
    A = A/np.linalg.norm(A, axis=1, keepdims=True); Bv = Bv/np.linalg.norm(Bv, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd(Bv.T @ A)
    D = np.diag([1, 1, np.sign(np.linalg.det(U@Vt))])
    return U@D@Vt


def smplx_shape(model, betas=None):
    """(v_shaped [V,3], joints [55,3]) of the SMPL-X body for shape betas (None = 0); model = np.load(SMPLX npz)."""
    v = model['v_template'].astype(np.float64)
    if betas is not None:
        v = v + model['shapedirs'][:, :, :len(betas)].astype(np.float64) @ np.asarray(betas, np.float64)
    return v, model['J_regressor'].astype(np.float64) @ v


def fit_betas(model, so, n=N_BETAS, reg=BETA_REG):
    """Shape betas whose SHAPE_PAIRS distances equal the SOMA rest ones (least squares + ridge); returns betas, max err m."""
    from scipy.optimize import least_squares
    Jr = model['J_regressor'].astype(np.float64)[:25]
    J0 = Jr @ model['v_template'].astype(np.float64)
    JD = np.einsum('jv,vcb->jcb', Jr, model['shapedirs'][:, :, :n].astype(np.float64))
    ia, ib = np.array(SHAPE_PAIRS).T
    target = np.linalg.norm(so[M[ia]] - so[M[ib]], axis=1)
    dist = lambda b: np.linalg.norm((J0 + JD @ b)[ia] - (J0 + JD @ b)[ib], axis=1)
    sol = least_squares(lambda b: np.concatenate([dist(b) - target, reg*b]), np.zeros(n), method='lm', max_nfev=20000)
    return sol.x, float(np.abs(dist(sol.x) - target).max())


def soma_rest_joints():
    from ardy.skeleton import SOMASkeleton30
    return SOMASkeleton30().neutral_joints.detach().cpu().numpy().astype(np.float64)


def corrections(sx, so):
    """Rest correction per SMPL-X joint; leaves inherit the parent's; neck aligns with the Neck1->Head chord."""
    C = np.repeat(np.eye(3)[None], 25, 0)
    for j in range(25):                    # parents precede children in SMPL-X order
        kids = [(c, M[c]) for c in range(25) if SMPLX_PARENTS[c] == j and c not in NO_ALIGN] + EXTRA.get(j, [])
        if not kids:
            C[j] = C[SMPLX_PARENTS[j]]
            continue
        A = np.stack([sx[c]-sx[j] for c, _ in kids]); Bv = np.stack([so[k]-so[M[j]] for _, k in kids])
        C[j] = _rot_between(A[0], Bv[0]) if len(kids) == 1 else _kabsch(A, Bv)
    return C


def _globals(local, parents):
    g = np.empty_like(local)
    for j, p in enumerate(parents): g[:, j] = local[:, j] if p < 0 else g[:, p] @ local[:, j]
    return g


def _locals(g, parents):
    l = np.empty_like(g)
    for j, p in enumerate(parents): l[:, j] = g[:, j] if p < 0 else g[:, p].swapaxes(-1, -2) @ g[:, j]
    return l


class Converter:
    def __init__(self, smplx_npz, fit_shape=True):
        """smplx_npz: path to SMPLX_NEUTRAL_2020.npz.
        fit_shape=True: SMPL-X uses self.betas (fitted to the SOMA skeleton); False: betas=0."""
        model = np.load(smplx_npz, allow_pickle=True)
        self.so = soma_rest_joints()
        self.betas, self.fit_err_m = fit_betas(model, self.so) if fit_shape else (np.zeros(N_BETAS), None)
        self.sx = smplx_shape(model, self.betas)[1]
        self.C = corrections(self.sx, self.so)

    def soma_to_smplx(self, local30, root):
        """local30 [T,30,3,3] SOMA local rotations, root [T,3] SOMA Hips world position.
        Returns SMPL-X local rotations [T,25,3,3] (0 = global_orient, 1..21 body, 22 jaw, 23/24 eyes),
        transl [T,3] (for shape self.betas) and sidecar [T,6,3,3] SOMA local rotations of SIDECAR joints."""
        local30 = np.asarray(local30, np.float64)
        gs = _globals(local30, SOMA_PARENTS)
        gx = gs[:, M] @ self.C
        # neck swing: SMPL-X neck->head bone onto the posed SOMA Neck1->Head chord
        chord = (np.einsum('tab,b->ta', gs[:, S_NECK1], self.so[S_NECK2]-self.so[S_NECK1])
                 + np.einsum('tab,b->ta', gs[:, S_NECK2], self.so[S_HEAD]-self.so[S_NECK2]))
        gx[:, NECK] = _swing(np.einsum('tab,b->ta', gx[:, NECK], self.sx[HEAD]-self.sx[NECK]), chord) @ gx[:, NECK]
        transl = np.asarray(root, np.float64) - self.sx[0]
        return _locals(gx, SMPLX_PARENTS), transl, local30[:, SIDECAR].copy()

    def smplx_to_soma(self, local25, transl, sidecar=None):
        local25 = np.asarray(local25, np.float64); T = len(local25)
        gx = _globals(local25, SMPLX_PARENTS)
        gs = np.empty((T, 30, 3, 3))
        side = np.repeat(np.eye(3)[None, None], T, 0).repeat(len(SIDECAR), 1) if sidecar is None else np.asarray(sidecar, np.float64)
        known = dict(zip(M.tolist(), list((gx @ self.C.swapaxes(-1, -2)).swapaxes(0, 1))))
        if sidecar is not None: known.pop(S_NECK1)            # exact Neck1 from the sidecar (neck swing is not invertible alone)
        for k, p in enumerate(SOMA_PARENTS):     # parents precede children in SOMA order
            if k in known: gs[:, k] = known[k]
            else: gs[:, k] = gs[:, p] @ side[:, SIDECAR.index(k)]
        return _locals(gs, SOMA_PARENTS), np.asarray(transl, np.float64) + self.sx[0]
