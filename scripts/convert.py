"""Convert a motion file between SOMA30 and SMPL-X (mesh fit). Both formats: y-up, metres.
SOMA npz:   local_rot_mats [T,30,3,3], root_positions [T,3]
SMPL-X npz: poses [T,165] axis-angle (55 joints), trans [T,3], betas [30] (fitted shape: 20 betas + 10 expression)
Usage: python scripts/convert.py soma2smplx in.npz out.npz --smplx-model ... --rest rest_fit.npz
       python scripts/convert.py smplx2soma in.npz out.npz --smplx-model ... --rest rest_fit.npz
smplx2soma assumes the input was made with the fitted shape (betas in the input are ignored)."""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from soma_smplx.meshfit import MeshConverter

ap = argparse.ArgumentParser()
ap.add_argument('direction', choices=['soma2smplx', 'smplx2soma']); ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--smplx-model', required=True); ap.add_argument('--rest', default='rest_fit.npz')
a = ap.parse_args()
C = MeshConverter(a.smplx_model, a.rest); z = np.load(a.inp, allow_pickle=True)
if a.direction == 'soma2smplx':
    p55, tr = C.soma_to_smplx(z['local_rot_mats'].astype(np.float32), z['root_positions'].astype(np.float32))
    np.savez(a.out, poses=p55.reshape(len(p55), -1), trans=tr, betas=C.shape.numpy(), gender='neutral', mocap_frame_rate=30)
else:
    l, r = C.smplx_to_soma(z['poses'].reshape(len(z['poses']), 55, 3).astype(np.float32), z['trans'].astype(np.float32))
    np.savez(a.out, local_rot_mats=l, root_positions=r)
print('wrote', a.out)
