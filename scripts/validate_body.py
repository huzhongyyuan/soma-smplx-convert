"""Check the installed body converter with separately licensed local assets.

Reports finite geometry, full SMPL-X equivalence and a real CLI round trip.
This is a technical smoke check, not a dataset quality threshold.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from soma_smplx.body import BodyConverter, reference


def stats(value):
    value = np.asarray(value)*1000
    return dict(mean=float(value.mean()),p95=float(np.percentile(value,95)),max=float(value.max()))


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument('--smplx-model',required=True)
    ap.add_argument('--rest',required=True)
    ap.add_argument('--input',help='Optional SOMA npz; only its first frame is used.')
    ap.add_argument('--out',required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    c = BodyConverter(args.smplx_model,args.rest)
    if args.input:
        z = np.load(args.input,allow_pickle=False)
        local = z['local_rot_mats'][:1].astype(np.float32)
        root = z['root_positions'][:1].astype(np.float32)
        input_hash = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    else:
        local = np.tile(np.eye(3,dtype=np.float32),(1,30,1,1))
        root = np.array([[.2,1.1,-.3]],dtype=np.float32)
        input_hash = None
    pose, translation = c.soma_to_smplx(local,root)
    state = c._smplx_state(pose,translation)
    full = reference.SmplxModel(c.base.sx,c.base.shape).verts(*state)
    collapsed = c.smplx.verts(*state)
    equivalence = float((full-collapsed).abs().max())
    assert equivalence<2e-6, equivalence
    back, backroot = c.smplx_to_soma(pose,translation)
    original = c.soma.verts(*c._soma_state(local,root))[:,c.body_s].numpy()
    returned = c.soma.verts(*c._soma_state(back,backroot))[:,c.body_s].numpy()
    assert np.isfinite(returned).all()
    assert np.max(np.abs(back.swapaxes(-1,-2)@back-np.eye(3)))<2e-4
    with tempfile.TemporaryDirectory(prefix='soma-body-check-') as tmp:
        folder = Path(tmp)
        times = np.array([12.5],dtype=np.float64)
        np.savez(folder/'input.npz',local_rot_mats=local,root_positions=root,fps=30,timestamps=times)
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        for direction,source,target in [('soma2smplx','input.npz','x.npz'),('smplx2soma','x.npz','back.npz')]:
            cmd=[sys.executable,'-B',str(Path(__file__).with_name('convert_body.py')),direction,str(folder/source),str(folder/target),
                 '--smplx-model',args.smplx_model,'--rest',args.rest,'--threads','1']
            run=subprocess.run(cmd,env=env,text=True,capture_output=True,timeout=120)
            if run.returncode: raise RuntimeError(run.stderr)
        x=np.load(folder/'x.npz',allow_pickle=False)
        b=np.load(folder/'back.npz',allow_pickle=False)
        np.testing.assert_array_equal(x['poses'].reshape(-1,55,3)[:,22:],0)
        np.testing.assert_array_equal(x['poses'].reshape(-1,55,3),pose)
        np.testing.assert_array_equal(x['trans'],translation)
        np.testing.assert_array_equal(b['timestamps'],times)
        np.testing.assert_array_equal(b['local_rot_mats'],back)
        np.testing.assert_array_equal(b['root_positions'],backroot)
        assert float(b['fps'])==30
    result={'scope':'One-frame technical smoke; no quality acceptance inferred.',
            'input_sha256':input_hash,'rest_sha256':hashlib.sha256(Path(args.rest).read_bytes()).hexdigest(),
            'smplx_model_sha256':hashlib.sha256(Path(args.smplx_model).read_bytes()).hexdigest(),
            'full_smplx_equivalence_max_coordinate_m':equivalence,'cli_roundtrip':'passed',
            'neutral_nonbody':'passed','timestamps_and_fps':'passed',
            'soma_body_roundtrip_mm':stats(np.linalg.norm(returned-original,axis=-1)),
            'root_roundtrip_mm':stats(np.linalg.norm(backroot-root,axis=-1))}
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__': main()
