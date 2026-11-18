"""Purpose: Load data for the reference state pipeline.

Main functionality: Provide multiview datasets, state selection, and view dropout.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LeanMultiViewDataset(Dataset):
    def __init__(self, data_dir: str):
        d = Path(data_dir)
        self.kp_mv         = np.load(d / 'keypoints_mv.npy').astype(np.float32)
        self.mpjpe_per_cam = np.load(d / 'mpjpe_per_cam.npy').astype(np.float32)
        assert len(self.kp_mv) == len(self.mpjpe_per_cam)

    def __len__(self):
        return len(self.kp_mv)

    def __getitem__(self, idx):
        return (torch.tensor(self.kp_mv[idx],         dtype=torch.float32),
                torch.tensor(self.mpjpe_per_cam[idx], dtype=torch.float32))


class StateMultiViewDataset(LeanMultiViewDataset):
    """LeanMultiViewDataset + the per-camera lifter state for that row."""

    def __init__(self, data_dir, npz_name='r3_guard_state.npz'):
        super().__init__(data_dir)
        z = np.load(Path(data_dir) / npz_name)
        self.st_mean = z['mean'].astype(np.float32)          # (N,8,17,3)
        self.st_var = z['var'].astype(np.float32)
        assert len(self.st_mean) == len(self.kp_mv), (
            f'state has {len(self.st_mean)} rows, split has {len(self.kp_mv)} -- '
            'wrong split or a stale export')

    def __getitem__(self, idx):
        kp, sc = super().__getitem__(idx)
        return kp, sc, torch.from_numpy(self.st_mean[idx]), torch.from_numpy(self.st_var[idx])


def pick_state(st_mean, st_var, ref):
    """(B,8,17,3) x (B,) -> the CURRENT camera's state, (B,17,3)."""
    b = torch.arange(st_mean.size(0), device=st_mean.device)
    return st_mean[b, ref], st_var[b, ref]


def make_view_dropout(train_k: int, ref_mode: str):
    """stage131's apply_view_dropout_with_ref with the two policy knobs exposed.

    train_k = 0 -> K ~ U{1..n_cams} (stage131's behaviour); otherwise K is
    fixed at min(train_k, n_cams).
    ref_mode = 'first' -> ref = min(keep) (stage131's behaviour, which skews
    the reference camera onto low indices); 'uniform' -> ref ~ U(keep).

    The (train_k=0, ref_mode='first') path consumes exactly randint then
    sample, in stage131's order, so it reproduces the shipped runs bit-for-bit.
    """
    def apply(kp: torch.Tensor):
        B, out = kp.size(0), kp.clone()
        n_cams = kp.size(1)
        K_fixed = min(train_k, n_cams) if train_k > 0 else 0
        ref = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            K = random.randint(1, n_cams) if K_fixed == 0 else K_fixed
            keep = sorted(random.sample(range(n_cams), K))
            mask = torch.zeros(n_cams, dtype=torch.bool); mask[keep] = True
            out[b, ~mask] = 0.0
            ref[b] = keep[0] if ref_mode == 'first' else random.choice(keep)
        return out, ref
    return apply
