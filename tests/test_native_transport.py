import unittest
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from soma_smplx.native_transport import (BODY_FROM_SOMA, SOMA_PARENTS,
    SMPLX_PARENTS, PRODUCERS, HAND_ORDER, soma_to_smplx, handrep_fingers)


def global_rotations(local,parents):
    result = []
    for j,p in enumerate(parents):
        result.append(local[:,j] if p<0 else result[p] @ local[:,j])
    return np.stack(result,axis=1)


class NativeTransportTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(8023)
        self.local = Rotation.random(3*30,random_state=self.rng).as_matrix().reshape(3,30,3,3)
        self.root = self.rng.normal(size=(3,3)) + [10,2,-7]

    def convert(self,**kwargs):
        options=dict(producer=PRODUCERS[0],root_scale=1.09); options.update(kwargs)
        return soma_to_smplx(self.local,self.root,**options)

    def test_arbitrary_neck2_preserves_all_selected_global_orientations(self):
        poses,trans = self.convert()
        local = Rotation.from_rotvec(poses[:,:66].reshape(-1,3)).as_matrix().reshape(3,22,3,3)
        expected = global_rotations(self.local,SOMA_PARENTS)[:,BODY_FROM_SOMA]
        np.testing.assert_allclose(global_rotations(local,SMPLX_PARENTS),expected,atol=2e-15,rtol=0)
        np.testing.assert_array_equal(trans,self.root/1.09)
        self.assertGreater(np.linalg.norm(trans[0]),1)  # no root recentering
        self.assertFalse(poses[:,66:].any())

    def test_exact_finger_passthrough_and_handrep_column_order(self):
        fingers = Rotation.random(3*2*15,random_state=self.rng).as_matrix().reshape(3,2,15,3,3)
        thumb_first = fingers[:,:,HAND_ORDER]
        six = np.concatenate((thumb_first[...,0],thumb_first[...,1]),axis=-1)
        hand = np.zeros((3,2,96)); hand[...,6:] = six.reshape(3,2,90)
        recovered = handrep_fingers(hand)
        np.testing.assert_allclose(recovered.reshape(fingers.shape),fingers,atol=6e-16,rtol=0)
        rv = Rotation.from_matrix(recovered.reshape(-1,3,3)).as_rotvec().reshape(3,30,3)
        poses,_ = self.convert(hands=rv)
        np.testing.assert_array_equal(poses[:,75:],rv.reshape(3,90))

    def test_reject_wrong_producer_scale_rotation_and_timeline(self):
        for options in ({'producer':'meshfit'},{'root_scale':0},{'root_scale':np.nan},
                        {'hands':np.zeros((2,30,3))}):
            with self.subTest(options=options),self.assertRaises(ValueError): self.convert(**options)
        bad = self.local.copy(); bad[0,2] *= 2
        with self.assertRaises(ValueError): soma_to_smplx(bad,self.root,producer=PRODUCERS[0],root_scale=1)
        hand = np.zeros((3,2,96))
        with self.assertRaises(ValueError): handrep_fingers(hand)

    def test_file_api_preserves_time_raw_neck2_and_all300_betas(self):
        script=Path(__file__).resolve().parents[1]/'scripts/convert_native.py'
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);source=d/'input.npz';sidecar=d/'hands.npz';out=d/'output.npz'
            neutral=np.broadcast_to(np.eye(3),(3,30,3,3)).copy()
            np.savez(source,local_rot_mats=neutral,body_local_rotations_raw=self.local,
                     root_positions=self.root,root_scale=np.asarray(1.09),fps=np.asarray(30.),
                     times_seconds=np.arange(3)/30,units=np.asarray('meter'),up_axis=np.asarray('Y'))
            betas=self.rng.normal(size=300);hands=np.zeros((3,90))
            np.savez(sidecar,betas=betas,finger_rotvec=hands)
            command=[sys.executable,str(script),str(source),str(out),'--producer',PRODUCERS[0],'--hands-from',str(sidecar)]
            done=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            with np.load(out,allow_pickle=False) as z:
                expected,_=self.convert()
                np.testing.assert_array_equal(z['poses'],expected)
                np.testing.assert_array_equal(z['betas'],betas)
                np.testing.assert_array_equal(z['times_seconds'],np.arange(3)/30)
                self.assertFalse(json.loads(str(z['conversion_metadata_json']))['gt_lookup'])
            self.assertNotEqual(subprocess.run(command,capture_output=True).returncode,0)


if __name__=='__main__': unittest.main()
