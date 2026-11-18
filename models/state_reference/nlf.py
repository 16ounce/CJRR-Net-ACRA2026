"""Purpose: Load NLF-S pose estimates for the reference pipeline.

Main functionality: Select joints, correct roots, and calibrate articulation uncertainty.
"""

import json
import os
from pathlib import Path

import numpy as np

from .geometry import COCO17_TO_MPI28, MPI28_JOINT_NAMES

NLFNET = str(Path(__file__).resolve().parents[2] / 'data' / 'nlfnet')
ROOT = [11, 12]        # COCO has no pelvis; the hip midpoint is the root
IMG = 2048.0


def load_nlf(split, skeleton='coco17'):
    """Load NLF 3D in COCO-17 or PoseRL-15 order.

    ``skeleton='coco17'`` is the historical default and preserves every
    existing call site. The RL path uses ``skeleton='poserl15'``.
    """
    d = os.path.join(NLFNET, 'data', f'nlf3d_split_{split}')
    names = json.load(open(os.path.join(d, 'mpi28_joint_names.json')))
    assert names == list(MPI28_JOINT_NAMES), 'MPI-28 joint order differs between the two trees'
    nlf = np.load(os.path.join(d, 'nlf3d_mv.npy'))            # (N,8,28,3) mm, camera frame
    unc = np.load(os.path.join(d, 'nlf3d_unc_mv.npy'))        # (N,8,28)
    valid = np.load(os.path.join(d, 'nlf_valid_mv.npy'))      # (N,8) bool
    gt28 = np.load(os.path.join(d, 'gt3d_mv.npy'))
    if skeleton == 'coco17':
        joint_indices = COCO17_TO_MPI28
    elif skeleton == 'poserl15':
        from .poserl15 import POSERL15_TO_MPI28
        joint_indices = POSERL15_TO_MPI28
    else:
        raise ValueError(f'unknown skeleton {skeleton!r}; expected coco17 or poserl15')
    return (nlf[..., joint_indices, :].astype(np.float32),
            unc[..., joint_indices].astype(np.float32),
            valid, gt28[..., joint_indices, :].astype(np.float32))


def root_of(x):
    return x[..., ROOT, :].mean(-2)


def rebase(pose, new_root):
    """Keep pose's shape, move its root to new_root."""
    return pose - root_of(pose)[..., None, :] + new_root[..., None, :]


def moment_scale(err, unc):
    """s such that E[(s*unc)^2] = E[err^2] -- a second-moment fit. The
    original mean-ratio fit left the test sigma/err ratio at 0.773, outside
    the pre-registered [0.80, 1.25] band; this puts it at 0.883."""
    return float(np.sqrt((err ** 2).mean() / (unc ** 2).mean()))
