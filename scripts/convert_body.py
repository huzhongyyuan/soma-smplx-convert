# SPDX-License-Identifier: Apache-2.0
# Body release packaging, 2026-11-02; numerical solver retained from the verified snapshot.
"""Explicit, non-resampling file interface for the frozen-shape body converter."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soma_smplx.body import BodyConverter


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('direction', choices=['soma2smplx', 'smplx2soma'])
    p.add_argument('input'); p.add_argument('output')
    p.add_argument('--smplx-model', required=True); p.add_argument('--rest', required=True)
    p.add_argument('--fps', type=float, default=30, help='Declared input fps when the archive has no fps field. No resampling.')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--hands-from', help='SMPL-X npz with aligned poses[T,165] or hands[T,30,3]; passthrough only')
    p.add_argument('--canonicalize-shape', action='store_true', help='Explicitly normalize input SMPL-X shape to the frozen rest shape, preserving body local rotations and world pelvis before retargeting')
    args = p.parse_args()
    if torch.cuda.is_available(): raise RuntimeError('Run with CUDA_VISIBLE_DEVICES empty; this implementation is CPU-only')
    if not 1 <= args.threads <= 2: raise ValueError('This candidate is validated with at most two math threads')
    torch.set_num_threads(args.threads); torch.set_num_interop_threads(args.threads)
    source, destination = Path(args.input), Path(args.output)
    if destination.exists(): raise FileExistsError(destination)
    z = np.load(source, allow_pickle=False)
    if args.direction == 'soma2smplx' and 'conversion_version' in z.files:
        from soma_smplx.native_transport import PRODUCERS
        if str(z['conversion_version']) in PRODUCERS:
            raise ValueError('Native rotation-transport input requires scripts/convert_native.py; surface fitting is a different inverse')
    fps = float(z['fps']) if 'fps' in z.files else args.fps
    if not np.isfinite(fps) or fps <= 0: raise ValueError('Invalid frame rate')
    c = BodyConverter(args.smplx_model, args.rest)
    meta = {'direction': args.direction, 'input': str(source.resolve()), 'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'converter_sha256': hashlib.sha256((Path(__file__).resolve().parents[1] / 'soma_smplx' / 'body.py').read_bytes()).hexdigest(),
            'rest_sha256': hashlib.sha256(Path(args.rest).read_bytes()).hexdigest(), 'fps': fps,
            'shape_policy': 'fixed fitted rest shape', 'smoothing': False, 'resampling': False, 'body_joints_smplx': list(range(22))}
    if args.direction == 'soma2smplx':
        hands = z['smplx_hand_pose'] if 'smplx_hand_pose' in z.files else None
        if args.hands_from:
            h = np.load(args.hands_from, allow_pickle=False)
            hands = h['hands'] if 'hands' in h.files else h['poses'].reshape(-1, 55, 3)[:, 25:]
        poses, trans = c.soma_to_smplx(z['local_rot_mats'], z['root_positions'], hands=hands)
        output = dict(poses=poses.reshape(-1, 165), trans=trans, betas=c.base.shape[:20].numpy(), expression=c.base.shape[20:].numpy())
        meta['hands'] = 'exact local-rotation passthrough' if hands is not None else 'neutral default'
    else:
        poses = np.asarray(z['poses']).reshape(-1, 55, 3)
        trans = np.asarray(z['trans']).copy()
        if 'betas' not in z.files:
            raise ValueError('SMPL-X input must declare betas; shape cannot be guessed')
        betas = np.asarray(z['betas']).reshape(-1)
        if len(betas) > 20 and np.any(np.abs(betas[20:]) > 1e-8):
            raise ValueError('Source uses nonzero shape coefficients beyond this model adapter\'s 20 supported coefficients')
        shape = torch.zeros(30)
        shape[:min(20, len(betas))] = torch.from_numpy(betas[:20]).float()
        if 'expression' in z.files:
            expression = np.asarray(z['expression'])
            if expression.ndim != 1 or len(expression) > 10:
                raise ValueError('Per-frame or >10 expression coefficients are outside the body-only shape contract')
            shape[20:20 + len(expression)] = torch.from_numpy(expression).float()
        if not torch.allclose(shape, c.base.shape, atol=1e-5, rtol=0):
            if not args.canonicalize_shape:
                raise ValueError('Source shape differs from the frozen rest fit. Use a matching rest artifact, or explicitly --canonicalize-shape; no silent subject-shape replacement.')
            source_j0 = c.base.sx.shaped(shape)[1][0].numpy()
            trans += source_j0 - c.base.J[0].numpy()
            meta['shape_policy'] = 'explicit source-to-rest shape normalization; world pelvis preserved before fitting'
            meta['original_betas'] = betas.tolist()
        local, root = c.smplx_to_soma(poses, trans)
        output = dict(local_rot_mats=local, root_positions=root, smplx_hand_pose=poses[:, 25:].copy())
        meta['hands'] = 'stored unchanged as SMPL-X local rotations; not fitted through SOMA'
    for field in ['timestamps', 'frame_times']:
        if field in z.files:
            if len(z[field]) != len(next(iter(output.values()))): raise ValueError('Time array length mismatch')
            output[field] = z[field]
    output['fps'] = np.asarray(fps)
    output['conversion_metadata_json'] = np.asarray(json.dumps(meta, ensure_ascii=False))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream: np.savez(stream, **output)
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__': main()
