"""Purpose: Define shared-input comparison models.

Main functionality: Provide the matched-input MLP and mean-pooled reprojection control.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from models.cjrrnet.cjrrnet import (
    JointTokenReprojVisRankNet, CAM_ANGLES_DEG, compute_all_deltas, BASELINE_NORM, CLIP)
from mpi_camera_geometry import build_camera_constants
from reproj_joint_features import build_RTK, normalize_features

CONFIGS = {'Mean12': (True, False, 'none', 'attn'),
           'MatchedMLP': (False, False, 'none', 'attn')}

class Mean12RankNet(JointTokenReprojVisRankNet):
    """B with mean over joints BEFORE its 12->32->32 MLP, without FiLM.

    Normalize/clip all 12 channels exactly as B/K, then mean only over J.
    Construct B first to preserve initialization of every surviving module.
    Delete the unused attention scorer to keep the parameter count honest.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.joint_tokens or self.film_mode != 'none':
            raise ValueError('Mean12 requires joint_tokens=True and no FiLM')
        del self.joint_attn

    def _pool_joint_tokens(self, feats, cur, q_src=None, q_full=None):
        z = normalize_features(feats, self.feat_center, self.feat_scale, CLIP)
        pooled = self.joint_mlp(z.mean(dim=-2))
        weights = z.new_full(z.shape[:-1], 1.0 / z.shape[-2])
        return pooled, weights, None

class MatchedInputMLP(nn.Module):
    """Flat MLP: masked observed 2D/conf + current NLF mean/variance + rig.

    No joint tokens, attention, FiLM, reprojection features or lifter.
    All available observations survive the SAME K~U{1..8} input dropout;
    evaluation supplies only the current view, exactly as for Full.
    A current-view one-hot identifies the coordinate frame of the NLF state.
    Fixed invertible unit transforms use no val/test statistics: mm -> metres,
    variance -> log1p(mm^2)/10, intrinsics -> /2048 (MPI image size).
    """
    def __init__(self, n_cands=8, n_joints=17, n_features=3, dropout=0.1,
                 cam_ids=None, cam_Rs=None, cam_Ts=None, cam_Ks=None,
                 cam_angles=None, baseline_matrix=None, **unused):
        super().__init__()
        self.n_cands, self.n_joints, self.n_features = n_cands, n_joints, n_features
        Rs, Ts, Ks = build_RTK(cam_ids) if cam_Rs is None else (cam_Rs, cam_Ts, cam_Ks)
        if baseline_matrix is None:
            _, _, baseline_matrix = (build_camera_constants(cam_ids=cam_ids)
                                     if cam_ids is not None else build_camera_constants())
        angles = CAM_ANGLES_DEG if cam_angles is None else np.asarray(cam_angles)
        for name, value in [('cam_R', Rs), ('cam_T', Ts), ('cam_K', Ks),
                            ('all_deltas', compute_all_deltas(angles)),
                            ('baseline_matrix', baseline_matrix / BASELINE_NORM)]:
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32))
        if self.cam_R.shape != (n_cands, 3, 3):
            raise ValueError('camera geometry does not match candidate count')
        # observed 2D/conf, NLF mean/var, current one-hot, whole rig R/T/K,
        # and Full's relative azimuth/elevation/baseline descriptors.
        self.input_dim = n_cands * n_joints * n_features + 6 * n_joints + 25 * n_cands
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_cands))

    def input_vector(self, x, cur, external_state):
        if external_state is None:
            raise ValueError('MatchedMLP requires the current NLF external_state')
        mean, var = external_state
        b = x.shape[0]
        if x.shape[1:] != (self.n_cands, self.n_joints, self.n_features):
            raise ValueError('unexpected keypoint shape')
        if mean.shape != (b, self.n_joints, 3) or var.shape != mean.shape:
            raise ValueError('expected current-camera mean/variance [B,J,3]')
        rig = torch.cat([self.cam_R.flatten(), self.cam_T.flatten() / 1000.,
                         self.cam_K.flatten() / 2048.])
        return torch.cat([
            x.flatten(1), mean.flatten(1) / 1000., torch.log1p(var.flatten(1)) / 10.,
            F.one_hot(cur, self.n_cands).to(x.dtype), rig.expand(b, -1),
            self.all_deltas[cur].flatten(1), self.baseline_matrix[cur]], dim=1)

    def forward(self, x, cur, external_state=None, return_aux=False):
        scores = self.mlp(self.input_vector(x, cur, external_state))
        return (scores, None) if return_aux else scores

def build_control(config, **kwargs):
    return {'Mean12': Mean12RankNet, 'MatchedMLP': MatchedInputMLP}[config](**kwargs)
