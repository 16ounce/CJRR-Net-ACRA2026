"""Purpose: Define PoseRL-compatible joint mappings.

Main functionality: Convert MPI-28 and COCO-17 poses to the 15-joint layout.
"""

from __future__ import annotations

import numpy as np
import torch

from .geometry import MPI28_JOINT_NAMES


JOINT_NAMES = [
    "neck",
    "nose",
    "mid_hip",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
]
N_JOINTS = len(JOINT_NAMES)
ROOT_INDEX = JOINT_NAMES.index("mid_hip")

# Same semantic order as NBV-PoseRL's official PanopticDataset_nbv.JOINTS_DEF.
POSERL15_TO_MPI28 = [
    MPI28_JOINT_NAMES.index("neck"),
    MPI28_JOINT_NAMES.index("head"),  # nose proxy: MPI-28 has no nose
    MPI28_JOINT_NAMES.index("pelvis"),
    MPI28_JOINT_NAMES.index("left_shoulder"),
    MPI28_JOINT_NAMES.index("left_elbow"),
    MPI28_JOINT_NAMES.index("left_wrist"),
    MPI28_JOINT_NAMES.index("left_hip"),
    MPI28_JOINT_NAMES.index("left_knee"),
    MPI28_JOINT_NAMES.index("left_ankle"),
    MPI28_JOINT_NAMES.index("right_shoulder"),
    MPI28_JOINT_NAMES.index("right_elbow"),
    MPI28_JOINT_NAMES.index("right_wrist"),
    MPI28_JOINT_NAMES.index("right_hip"),
    MPI28_JOINT_NAMES.index("right_knee"),
    MPI28_JOINT_NAMES.index("right_ankle"),
]

# NBV-PoseRL parent topology expressed as undirected edges.
SKELETON_EDGES = [
    (0, 2), (0, 1),
    (0, 3), (3, 4), (4, 5),
    (0, 9), (9, 10), (10, 11),
    (2, 6), (6, 7), (7, 8),
    (2, 12), (12, 13), (13, 14),
]


def mpi28_to_poserl15(x):
    """Select PoseRL-15 joints from an array/tensor with joint axis ``-2``."""
    if x.shape[-2] != 28:
        raise ValueError(f"expected 28 MPI joints, got shape {tuple(x.shape)}")
    if torch.is_tensor(x):
        index = torch.as_tensor(POSERL15_TO_MPI28, device=x.device)
        return x.index_select(-2, index)
    return np.take(x, POSERL15_TO_MPI28, axis=-2)


def coco17_to_poserl15(keypoints):
    """Convert COCO-17 ``(...,17,C)`` keypoints to PoseRL-15 order.

    Coordinates for neck/mid-hip are endpoint midpoints. Their confidence is
    the smaller endpoint confidence, so a synthetic joint is reliable only
    when both source joints are reliable.
    """
    if keypoints.shape[-2] != 17:
        raise ValueError(f"expected 17 COCO joints, got shape {tuple(keypoints.shape)}")

    def midpoint(a, b):
        out = (a + b) * 0.5
        if out.shape[-1] >= 3:
            if torch.is_tensor(out):
                out = out.clone()
                out[..., 2] = torch.minimum(a[..., 2], b[..., 2])
            else:
                out = out.copy()
                out[..., 2] = np.minimum(a[..., 2], b[..., 2])
        return out

    neck = midpoint(keypoints[..., 5, :], keypoints[..., 6, :])
    mid_hip = midpoint(keypoints[..., 11, :], keypoints[..., 12, :])
    joints = [
        neck,
        keypoints[..., 0, :],
        mid_hip,
        keypoints[..., 5, :], keypoints[..., 7, :], keypoints[..., 9, :],
        keypoints[..., 11, :], keypoints[..., 13, :], keypoints[..., 15, :],
        keypoints[..., 6, :], keypoints[..., 8, :], keypoints[..., 10, :],
        keypoints[..., 12, :], keypoints[..., 14, :], keypoints[..., 16, :],
    ]
    stack = torch.stack if torch.is_tensor(keypoints) else np.stack
    return stack(joints, dim=-2) if torch.is_tensor(keypoints) else stack(joints, axis=-2)


def root_of(x):
    """Return the PoseRL-15 mid-hip root."""
    if x.shape[-2] != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joints, got shape {tuple(x.shape)}")
    return x[..., ROOT_INDEX, :]

