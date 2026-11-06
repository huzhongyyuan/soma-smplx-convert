# SPDX-License-Identifier: Apache-2.0
"""Matched inverse of declared local/global rotation transport into SOMA30.

This preserves mapped global orientations, including a nonneutral Neck2.
It is not an inverse of surface fitting and does not preserve SOMA bone lengths.
Root coordinates retain the producer's fixed world offset; no GT is consulted.
"""
import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 'native-transport-inverse-v1'
PRODUCERS = (
    'smplx_official_full_to_ardy_somaskel30_30fps_global_transport_v1',
    'soma_native55_transport',
    'soma_v4_fd_delivery',
    'hml_global_slerp_transport',
)
SOMA_PARENTS = (-1,0,1,2,3,4,5,6,6,6,3,10,11,12,13,13,3,16,17,18,19,19,0,22,23,24,0,26,27,28)
SMPLX_PARENTS = (-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19)
BODY_FROM_SOMA = (0,22,26,1,23,27,2,24,28,3,25,29,4,10,16,6,11,17,12,18,13,19)
HAND_ORDER = np.array([12,13,14,0,1,2,3,4,5,9,10,11,6,7,8])


def body_rotations(local):
    value = np.asarray(local, dtype=np.float64)
    if value.ndim != 4 or value.shape[1:] != (30,3,3) or not len(value):
        raise ValueError('Expected nonempty SOMA30 rotations [T,30,3,3]')
    if (not np.isfinite(value).all()
            or np.max(np.abs(value.swapaxes(-1,-2) @ value - np.eye(3))) > 2e-4
            or np.min(np.linalg.det(value)) < 0.9998):
        raise ValueError('Expected finite proper rotation matrices')
    # Every selected edge has the same parent except Neck -> Neck2 -> Head.
    # Multiply both local rotations; do not neutralize a reconstructed Neck2.
    result = value[:,BODY_FROM_SOMA].copy()
    result[:,15] = value[:,5] @ value[:,6]
    return result


def handrep_fingers(hand):
    """HandRep uses columns; undo thumb-first order, omit its wrist channels."""
    hand = np.asarray(hand, dtype=np.float64)
    if hand.ndim != 3 or hand.shape[1:] != (2,96) or not len(hand) or not np.isfinite(hand).all():
        raise ValueError('Expected finite HandRep [T,2,96]')
    six = hand[...,6:].reshape(len(hand),2,15,6)
    x = six[...,:3]
    nx = np.linalg.norm(x,axis=-1,keepdims=True)
    if np.min(nx) <= 1e-6:
        raise ValueError('Degenerate HandRep first column')
    x = x / nx
    y = six[...,3:] - (six[...,3:] * x).sum(-1,keepdims=True) * x
    ny = np.linalg.norm(y,axis=-1,keepdims=True)
    if np.min(ny) <= 1e-6:
        raise ValueError('Degenerate HandRep second column')
    y = y / ny
    matrix = np.stack((x,y,np.cross(x,y)),axis=-1)
    return matrix[:,:,np.argsort(HAND_ORDER)].reshape(len(hand),30,3,3)


def soma_to_smplx(local, root, *, producer, root_scale, hands=None):
    """Return SMPL-X poses[T,165] and translation; scale must be explicit.

    `hands` is aligned SMPL-X local finger rotvecs[T,30,3], not a mesh fit.
    Output shape is a separate caller-owned contract, not inferred from SOMA.
    """
    if producer not in PRODUCERS:
        raise ValueError('Unknown producer; native transport must not be guessed from array shape')
    body = body_rotations(local)
    root = np.asarray(root,dtype=np.float64)
    if root.shape != (len(body),3) or not np.isfinite(root).all():
        raise ValueError('Root/body clocks differ or root is nonfinite')
    if np.ndim(root_scale) != 0 or not np.isfinite(root_scale) or root_scale <= 0:
        raise ValueError('Explicit positive producer root_scale required')
    poses = np.zeros((len(body),165),dtype=np.float64)
    poses[:,:66] = Rotation.from_matrix(body.reshape(-1,3,3)).as_rotvec().reshape(len(body),66)
    if hands is not None:
        hands = np.asarray(hands,dtype=np.float64)
        if hands.shape != (len(body),30,3) or not np.isfinite(hands).all():
            raise ValueError('Expected aligned finite SMPL-X fingers [T,30,3]')
        poses[:,75:] = hands.reshape(len(body),90)
    return poses, root / float(root_scale)
