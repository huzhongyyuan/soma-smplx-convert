# SPDX-License-Identifier: Apache-2.0
"""Inverse declared SOMA rotation transport; CPU only, no GT or model lookup."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from soma_smplx.native_transport import PRODUCERS, VERSION, soma_to_smplx, handrep_fingers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('input',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--producer',choices=PRODUCERS,required=True)
    p.add_argument('--root-scale',type=float,help='Required unless recorded in input; use 1 for unscaled transport')
    p.add_argument('--hands-from',type=Path,help='Aligned finger_rotvec or smplx_hand_pose sidecar; optional betas')
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    with np.load(a.input,allow_pickle=False) as z:
        if 'units' in z.files and str(z['units']) not in ('m','meter'):
            raise ValueError('Native transport requires meters')
        if 'up_axis' in z.files and str(z['up_axis']) not in ('Y','Y_UP'):
            raise ValueError('Native transport requires Y-up')
        if 'conversion_version' in z.files and str(z['conversion_version']) != a.producer:
            raise ValueError('Declared producer conflicts with input conversion_version')
        scale = float(z['root_scale']) if 'root_scale' in z.files else a.root_scale
        if scale is None: raise ValueError('Producer root scale is required, including explicit 1')
        if a.root_scale is not None and scale != a.root_scale:
            raise ValueError('Root scale conflicts with producer metadata')
        hands = z['smplx_hand_pose'] if 'smplx_hand_pose' in z.files else None
        if 'hand_rep' in z.files:
            decoded = Rotation.from_matrix(handrep_fingers(z['hand_rep']).reshape(-1,3,3)).as_rotvec().reshape(-1,30,3)
            if hands is not None:
                left = Rotation.from_rotvec(hands.reshape(-1,3)).as_matrix()
                right = Rotation.from_rotvec(decoded.reshape(-1,3)).as_matrix()
                np.testing.assert_allclose(left,right,atol=2e-5,rtol=0)
            hands = decoded
        betas = np.zeros(20)
        if a.hands_from:
            with np.load(a.hands_from,allow_pickle=False) as h:
                hands = h['finger_rotvec'].reshape(-1,30,3) if 'finger_rotvec' in h.files else h['smplx_hand_pose']
                if 'betas' in h.files:
                    b = np.asarray(h['betas']).reshape(-1)
                    if len(b) not in (10,20,300) or not np.isfinite(b).all(): raise ValueError('Invalid betas sidecar')
                    betas = b.copy()  # retain all BEAT2 coefficients; do not truncate shape
        # Raw decoder rotations retain Neck2. A converter-only neutralized copy
        # is not a valid input to this matched inverse.
        local = z['body_local_rotations_raw'] if 'body_local_rotations_raw' in z.files else z['local_rot_mats']
        poses,trans = soma_to_smplx(local,z['root_positions'],producer=a.producer,root_scale=scale,hands=hands)
        fps = float(z['fps'])
        if not np.isfinite(fps) or fps<=0: raise ValueError('Invalid FPS')
        output = dict(poses=poses,trans=trans,betas=betas,expression=np.zeros(10),fps=np.asarray(fps))
        for key in ('timestamps','frame_times','times_seconds','source_frame30'):
            if key in z.files:
                if len(z[key]) != len(poses): raise ValueError('Time array length mismatch')
                output[key] = z[key]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    metadata = dict(converter=VERSION,producer=a.producer,root_scale=scale,
        source_sha256=sha(a.input),code_sha256=sha(Path(__file__).resolve().parents[1]/'soma_smplx/native_transport.py'),
        hands_sha256=sha(a.hands_from) if a.hands_from else None,
        shape_policy='hand-sidecar betas' if a.hands_from else 'explicit neutral zero shape',
        root_policy='producer world coordinates divided by scale; fixed offset retained',
        neck2_policy='compose into head; raw reconstruction retained',smoothing=False,resampling=False,gt_lookup=False)
    output['conversion_metadata_json'] = np.asarray(json.dumps(metadata))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('xb') as f: np.savez_compressed(f,**output)
    print(json.dumps(metadata))


if __name__=='__main__': main()
