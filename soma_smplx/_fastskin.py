# SPDX-License-Identifier: Apache-2.0
# Body release packaging, 2026-11-02; numerical solver retained from the verified snapshot.
"""Candidate fast SOMA77 LBS (dense weights + einsum), checked against ardy SOMASkin.skin. 2026-11-02"""
import numpy as np, torch
from ardy.skeleton.kinematics import batch_rigid_transform


class FastSoma:
    def __init__(self, skin, sk30):
        s77 = skin.skeleton_skin
        self.parents = s77.joint_parents.long(); self.root_idx = s77.root_idx
        self.nj = s77.neutral_joints.float()
        self.bind_inv = skin.bind_rig_transform_inv.float()          # [77,4,4]
        V = skin.bind_vertices.shape[0]
        W = torch.zeros(V, 77)
        W.scatter_add_(1, skin.lbs_indices, skin.lbs_weights.float())
        self.W = W; self.bv = skin.bind_vertices.float()
        self.slice30 = torch.as_tensor(sk30.get_skel_slice(s77)).long()
        self.relaxed = s77.relaxed_hands_rest_pose.float()           # [77,3,3]

    def to77(self, local30):
        L = self.relaxed.expand(local30.shape[0], 77, 3, 3).clone(); L[:, self.slice30] = local30; return L

    def verts(self, local77, root):
        T = local77.shape[0]
        jp, R = batch_rigid_transform(local77, self.nj[None].expand(T, 77, 3), self.parents, self.root_idx)
        jp = jp + root[:, None]
        A = (R @ self.bind_inv[:, :3, :3]); t = (R @ self.bind_inv[:, :3, 3:])[..., 0] + jp     # [T,77,3,3], [T,77,3]
        Mr = torch.einsum('vj,tjab->tvab', self.W, A); Mt = torch.einsum('vj,tja->tva', self.W, t)
        return (Mr @ self.bv[:, :, None])[..., 0] + Mt
