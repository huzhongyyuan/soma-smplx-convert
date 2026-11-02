"""Round-trip test of the mesh-fit converter on one clip, with optional 4-panel video / GIF.
  --soma  x.npz : SOMA -> SMPL-X -> SOMA      (local_rot_mats, root_positions)
  --smplx x.npz : SMPL-X -> SOMA -> SMPL-X    (poses [T,165], trans; evaluated with the fitted shape, not the file's betas)
Metrics (mm): surface distance between converted and source mesh (every 10th frame), round-trip vertex / root error,
round-trip rotation error, per-vertex acceleration (jitter) of input / converted / round-trip, seconds per frame.
Usage: python scripts/evaluate.py --soma clip.npz --start 200 --frames 120 --name FineDance_003 --out results --render --gif"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soma_smplx import meshfit as MF, render as RD


def mm(a):
    a = np.abs(np.asarray(a)) * 1000
    return dict(mean=round(float(a.mean()), 2), p95=round(float(np.percentile(a, 95)), 2), max=round(float(a.max()), 2))


def acc(v):  # [T,V,3] -> mean |second difference| per frame, mm
    return round(float(np.linalg.norm(v[2:] - 2 * v[1:-1] + v[:-2], axis=-1).mean() * 1000), 3)


def surf(points_seq, verts_seq, faces, every=10):
    return mm(np.concatenate([MF.surface_dist(p, v, faces) for p, v in zip(points_seq[::every], verts_seq[::every])]))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True); g.add_argument('--soma'); g.add_argument('--smplx')
    ap.add_argument('--start', type=int, default=0); ap.add_argument('--frames', type=int, default=120)
    ap.add_argument('--name'); ap.add_argument('--out', default='results')
    ap.add_argument('--smplx-model', required=True); ap.add_argument('--rest', default='rest_fit.npz')
    ap.add_argument('--render', action='store_true'); ap.add_argument('--gif', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    C = MF.MeshConverter(a.smplx_model, a.rest); s, n = a.start, a.frames
    name = a.name or os.path.splitext(os.path.basename(a.soma or a.smplx))[0]
    if a.soma:
        z = np.load(a.soma); local = z['local_rot_mats'][s:s+n].astype(np.float32); root = z['root_positions'][s:s+n].astype(np.float32)
        v0 = MF.soma_verts(local, root)
        t0 = time.time(); p55, tr = C.soma_to_smplx(local, root); t1 = time.time()
        v1 = C.smplx_verts(p55, tr)
        l2, r2 = C.smplx_to_soma(p55, tr); t2 = time.time()
        v2 = MF.soma_verts(l2, r2)
        rot = np.degrees(np.arccos(np.clip((np.einsum('tjab,tjab->tj', local, l2) - 1) / 2, -1, 1)))
        stem = f'soma_{name}'
        res = dict(case='soma_input', name=name, start=s, frames=len(local),
                   smplx_to_soma_surface_mm=surf(v1, v0, C.SF), soma_to_smplx_surface_mm=surf(v0, v1, C.sx.faces),
                   roundtrip_vertex_mm=mm(np.linalg.norm(v2 - v0, axis=-1)), roundtrip_root_mm=mm(np.linalg.norm(r2 - root, axis=-1)),
                   roundtrip_local_rot_deg=dict(mean=round(float(rot.mean()), 3), max=round(float(rot.max()), 3)),
                   accel_mm_per_frame2=dict(input=acc(v0), converted=acc(v1), roundtrip=acc(v2)),
                   sec_per_frame=dict(forward=round((t1 - t0) / len(local), 3), backward=round((t2 - t1) / len(local), 3)))
        np.savez_compressed(f'{a.out}/{stem}.npz', smplx_pose55=p55, smplx_transl=tr, soma_local_back=l2, soma_root_back=r2)
        F0, F1 = C.SF, C.sx.faces; cen = root[:, [0, 2]]
        names = ['SOMA input', 'SMPL-X (converted)', 'SOMA back (SOMA->SMPL-X->SOMA)', 'round-trip vertex error']
        title = f'only SOMA given | {name} frames {s}-{s+n}'
    else:
        z = np.load(a.smplx, allow_pickle=True)
        p55 = z['poses'][s:s+n].reshape(-1, 55, 3).astype(np.float32); tr = z['trans'][s:s+n].astype(np.float32)
        v0 = C.smplx_verts(p55, tr)
        t0 = time.time(); l, r = C.smplx_to_soma(p55, tr); t1 = time.time()
        v1 = MF.soma_verts(l, r)
        p2, tr2 = C.soma_to_smplx(l, r); t2 = time.time()
        v2 = C.smplx_verts(p2, tr2)
        rd = lambda x, y: np.degrees(np.linalg.norm(x - y, axis=-1))   # rough axis-angle distance
        stem = f'smplx_{name}'
        res = dict(case='smplx_input', name=name, start=s, frames=len(p55), note='SMPL-X shape = fitted rest shape, not the file betas',
                   soma_to_smplx_surface_mm=surf(v1, v0, C.sx.faces), smplx_to_soma_surface_mm=surf(v0, v1, C.SF),
                   roundtrip_vertex_mm=mm(np.linalg.norm(v2 - v0, axis=-1)), roundtrip_root_mm=mm(np.linalg.norm(tr2 - tr, axis=-1)),
                   roundtrip_pose_axisangle_deg=dict(body_mean=round(float(rd(p2[:, :22], p55[:, :22]).mean()), 3),
                                                     hands_mean=round(float(rd(p2[:, 25:55], p55[:, 25:55]).mean()), 3)),
                   accel_mm_per_frame2=dict(input=acc(v0), converted=acc(v1), roundtrip=acc(v2)),
                   sec_per_frame=dict(forward=round((t1 - t0) / len(p55), 3), backward=round((t2 - t1) / len(p55), 3)))
        np.savez_compressed(f'{a.out}/{stem}.npz', soma_local=l, soma_root=r, smplx_pose55_back=p2, smplx_transl_back=tr2)
        F0, F1 = C.sx.faces, C.SF; cen = v0.mean(1)[:, [0, 2]]
        names = ['SMPL-X input', 'SOMA (converted)', 'SMPL-X back (SMPL-X->SOMA->SMPL-X)', 'round-trip vertex error']
        title = f'only SMPL-X given | {name} frames {s}-{s+n} (shape = fitted rest shape)'
    if a.render:
        sx_rgb = RD.smplx_face_rgb(a.smplx_model, C.sx.faces); so_rgb = RD.soma_face_rgb(MF.soma_skin()[1], C.SF)
        R0, R1 = (so_rgb, sx_rgb) if a.soma else (sx_rgb, so_rgb)
        err = np.linalg.norm(v2 - v0, axis=-1) * 1000
        fw = RD.comparison_video(f'{a.out}/{stem}.mp4', title, names, [(v0, F0, R0), (v1, F1, R1), (v2, F0, R0)], F0, err, cen)
        res['worst_frame'] = fw
        if a.gif: RD.mp4_to_gif(f'{a.out}/{stem}.mp4', f'{a.out}/{stem}.gif')
    open(f'{a.out}/{stem}.json', 'w').write(json.dumps(res, indent=1) + '\n'); print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
