"""Purpose: Build joint reprojection features.

Main functionality: Load the Gaussian lifter and project its pose estimate into candidate views.
"""

import math
from pathlib import Path
import importlib.util

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent

import sys
sys.path.insert(0, str(_HERE))
from mpi_camera_geometry import parse_calibration, SELECTED_CAM_IDS  # noqa: E402
from skeleton_config import N_JOINTS, SKELETON_EDGES  # noqa: E402

IMAGE_SIZE = 2048.0
SQRT2 = math.sqrt(2.0)
BODY_RADIUS_MM = 80.0      # identical to exp132
DEPTH_MARGIN_MM = 50.0     # identical to exp132

N_FEATURES = 12
FEATURE_NAMES = [
    'proj_x', 'proj_y', 'proj_depth', 'sigma_x', 'sigma_y',
    'p_in_frame', 'p_self_occ', 'border_dist', 'person_scale',
    'bone_scale', 'overlap', 'cur_conf',
]

# camera index -> physical MPI-INF-3DHP camera id, SELECTED_CAM_IDS order
CAM_NAMES = [f'cam{c}' for c in SELECTED_CAM_IDS]   # cam0 cam1 cam2 cam4 cam5 cam6 cam7 cam8
DOMINANT_IDX = [4, 7]                                # cam5, cam8
RARE_IDX = [0, 1, 2, 3, 5, 6]                        # everything else (matches
                                                     # analyze_rare_camera_discrimination.py)
CAM1_IDX = 1


def load_lifter(ckpt_path, n_joints=N_JOINTS, n_features=3,
                d_hidden=256, output_scale=3000.0):
    spec = importlib.util.spec_from_file_location(
        'lift3d_gaussian_mlp', _HERE / 'gaussian_lifter.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lifter = mod.Lift3DGaussianMLP(n_joints=n_joints, n_features=n_features,
                                   d_hidden=d_hidden, dropout=0.0,
                                   output_scale=output_scale)
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        lifter.load_state_dict(ckpt['model_state_dict'])
    for p in lifter.parameters():
        p.requires_grad = False
    lifter.eval()
    return lifter


def build_RTK(cam_ids=None):
    if cam_ids is None:
        cam_ids = SELECTED_CAM_IDS
    cams = parse_calibration()
    Rs = np.stack([cams[c]['R'] for c in cam_ids]).astype(np.float32)
    Ts = np.stack([cams[c]['T'] for c in cam_ids]).astype(np.float32)
    Ks = np.stack([cams[c]['K'] for c in cam_ids]).astype(np.float32)
    return Rs, Ts, Ks


def build_non_adjacent_mask(n_joints=N_JOINTS, edges=SKELETON_EDGES):
    """(J,J) float -- 1 where j' is a VALID potential occluder of j."""
    adj = np.zeros((n_joints, n_joints), dtype=bool)
    for i, j in edges:
        adj[i, j] = True
        adj[j, i] = True
    non_adjacent = ~adj
    np.fill_diagonal(non_adjacent, False)
    return non_adjacent.astype(np.float32)


def build_bone_incidence(n_joints=N_JOINTS, edges=SKELETON_EDGES):
    """(J,J) float row-normalised incidence -- row j is uniform over j's
    direct skeleton neighbours, used to average incident bone lengths."""
    A = np.zeros((n_joints, n_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A /= np.maximum(A.sum(axis=1, keepdims=True), 1.0)
    return A


def gaussian_cdf(z):
    return 0.5 * (1.0 + torch.erf(z / SQRT2))


class ReprojJointFeatures(torch.nn.Module):
    """Computes the (B, C, J, 12) per-joint candidate-reprojection tensor.

    Holds the frozen Gaussian lifter + camera constants. Stateless apart from
    buffers; every op is differentiable-free (wrapped in no_grad by callers
    that only need values, left as-is for the model which keeps the lifter
    frozen anyway).
    """

    def __init__(self, lift3d_ckpt=None, cam_ids=None, n_joints=N_JOINTS,
                 var_floor=1.0, lifter=None):
        super().__init__()
        self.n_joints = n_joints
        self.var_floor = var_floor
        Rs, Ts, Ks = build_RTK(cam_ids)
        self.register_buffer('cam_R', torch.from_numpy(Rs))
        self.register_buffer('cam_T', torch.from_numpy(Ts))
        self.register_buffer('cam_K', torch.from_numpy(Ks))
        self.register_buffer('non_adjacent_mask',
                             torch.from_numpy(build_non_adjacent_mask(n_joints)))
        self.register_buffer('bone_incidence',
                             torch.from_numpy(build_bone_incidence(n_joints)))
        self.lifter = lifter if lifter is not None else load_lifter(lift3d_ckpt, n_joints=n_joints)

    def train(self, mode=True):
        super().train(mode)
        self.lifter.eval()
        return self

    def forward(self, cur_kp: torch.Tensor, cur: torch.Tensor):
        """
        Args:
            cur_kp : (B, J, 3) the CURRENT view's 2D keypoints (x, y, conf) --
                     the features depend on the input views only through this
                     one view's lift, exactly as in exp132
            cur    : (B,) long, index of the current camera in SELECTED_CAM_IDS order
        Returns:
            (B, C, J, 12) float tensor, raw units (see FEATURE_NAMES)
        """
        cur_conf = cur_kp[..., 2]                                       # (B,J)

        with torch.no_grad():
            mean_cam, var_cam = self.lifter(cur_kp)                     # (B,J,3)

        return self.field_from_gaussian(mean_cam, var_cam, cur_conf, cur)

    def field_from_gaussian(self, mean_cam: torch.Tensor, cov_cam: torch.Tensor,
                            cur_conf: torch.Tensor, cur: torch.Tensor):
        """The (B,C,J,12) field from an ARBITRARY current-camera-frame Gaussian.

        Split out of forward() (2026-07-28) so the static-active-view planner can
        push a REFINED mean/covariance through the exact same camera geometry
        instead of re-deriving R/T/K -- forward() itself is unchanged, it just
        calls this with the frozen lifter's own output.

        Args:
            mean_cam : (B, J, 3) 3D mean in the CURRENT camera's frame, mm
            cov_cam  : (B, J, 3)   per-axis variances (diagonal), mm^2  -- the
                                   ORIGINAL code path, kept bit-for-bit; or
                       (B, J, 3, 3) full per-joint covariance, mm^2
            cur_conf : (B, J) current-view ViTPose confidence (feature 11)
            cur      : (B,) long, current camera index in SELECTED_CAM_IDS order
        Returns:
            (B, C, J, 12) float tensor, raw units (see FEATURE_NAMES)
        """
        B, J, _ = mean_cam.shape

        # ---- current-camera frame -> world (identical to exp132) ----
        R_cur = self.cam_R[cur]                                         # (B,3,3)
        T_cur = self.cam_T[cur]                                         # (B,3)
        Rt_cur = R_cur.transpose(1, 2)
        X_world = torch.einsum('bij,bkj->bki', Rt_cur, mean_cam - T_cur.unsqueeze(1))
        if cov_cam.dim() == 3:      # diagonal -- untouched original einsum
            Cov_world = torch.einsum('bpi,bqi,bki->bkpq', Rt_cur, Rt_cur, cov_cam)
        else:                       # full 3x3 per joint
            Cov_world = torch.einsum('bpi,bkij,bqj->bkpq', Rt_cur, cov_cam, Rt_cur)

        # ---- world -> every candidate camera ----
        X_cand = torch.einsum('cij,bkj->bcki', self.cam_R, X_world) + self.cam_T.view(1, -1, 1, 3)
        Cov_cand = torch.einsum('cip,bkpq,cjq->bckij', self.cam_R, Cov_world, self.cam_R)

        Z = X_cand[..., 2].clamp_min(1e-3)                              # (B,C,J)
        fx = self.cam_K[:, 0, 0].view(1, -1, 1)
        fy = self.cam_K[:, 1, 1].view(1, -1, 1)
        cx = self.cam_K[:, 0, 2].view(1, -1, 1)
        cy = self.cam_K[:, 1, 2].view(1, -1, 1)
        Xc, Yc = X_cand[..., 0], X_cand[..., 1]
        x_px = fx * Xc / Z + cx                                         # (B,C,J)
        y_px = fy * Yc / Z + cy

        Cov00 = Cov_cand[..., 0, 0]; Cov02 = Cov_cand[..., 0, 2]; Cov22 = Cov_cand[..., 2, 2]
        Cov11 = Cov_cand[..., 1, 1]; Cov12 = Cov_cand[..., 1, 2]
        XZ = Xc / Z; YZ = Yc / Z
        var_x = (fx / Z) ** 2 * (Cov00 - 2 * XZ * Cov02 + XZ ** 2 * Cov22)
        var_y = (fy / Z) ** 2 * (Cov11 - 2 * YZ * Cov12 + YZ ** 2 * Cov22)
        std_x = var_x.clamp_min(self.var_floor).sqrt()
        std_y = var_y.clamp_min(self.var_floor).sqrt()

        p_x = gaussian_cdf((IMAGE_SIZE - x_px) / std_x) - gaussian_cdf((0.0 - x_px) / std_x)
        p_y = gaussian_cdf((IMAGE_SIZE - y_px) / std_y) - gaussian_cdf((0.0 - y_px) / std_y)
        p_in_frame = (p_x * p_y).clamp(0.0, 1.0)                        # (B,C,J)

        # ---- self-occlusion (identical to exp132) ----
        dx = x_px.unsqueeze(-1) - x_px.unsqueeze(-2)                    # (B,C,J,J) [...,j,j']
        dy = y_px.unsqueeze(-1) - y_px.unsqueeze(-2)
        pixel_dist = (dx ** 2 + dy ** 2 + 1e-6).sqrt()

        proj_radius = (fx.squeeze(-1).unsqueeze(-1) * BODY_RADIUS_MM / Z.clamp_min(1e-3))
        proj_radius_occluder = proj_radius.unsqueeze(-2).expand(-1, -1, J, -1)
        proximity = torch.exp(-(pixel_dist / proj_radius_occluder.clamp_min(1e-3)) ** 2)

        depth_diff = Z.unsqueeze(-1) - Z.unsqueeze(-2)                  # +: j' closer
        depth_mask = torch.sigmoid(depth_diff / DEPTH_MARGIN_MM)

        non_adj = self.non_adjacent_mask.view(1, 1, J, J)
        occluder_score = (depth_mask * proximity * non_adj).clamp(0.0, 1.0 - 1e-6)
        p_self_occ = 1.0 - torch.prod(1.0 - occluder_score, dim=-1)     # (B,C,J)

        # ---- NEW features (not present anywhere in exp132) ----
        # depth-agnostic crowding: same proximity kernel, no depth ordering
        overlap_score = (proximity * non_adj).clamp(0.0, 1.0 - 1e-6)
        overlap = 1.0 - torch.prod(1.0 - overlap_score, dim=-1)         # (B,C,J)

        # signed distance to nearest image border (negative => outside)
        border_dist = torch.minimum(
            torch.minimum(x_px, IMAGE_SIZE - x_px),
            torch.minimum(y_px, IMAGE_SIZE - y_px))                     # (B,C,J)

        # whole-skeleton projected bbox diagonal (per candidate, broadcast over J)
        w = x_px.amax(dim=-1) - x_px.amin(dim=-1)                       # (B,C)
        h = y_px.amax(dim=-1) - y_px.amin(dim=-1)
        person_scale = (w ** 2 + h ** 2 + 1e-6).sqrt().unsqueeze(-1).expand(-1, -1, J)

        # mean projected length of the bones incident to joint j
        bone_len = (dx ** 2 + dy ** 2 + 1e-6).sqrt()                    # (B,C,J,J)
        bone_scale = torch.einsum('bcij,ij->bci', bone_len, self.bone_incidence)

        cur_conf_bc = cur_conf.unsqueeze(1).expand(-1, x_px.size(1), -1)  # (B,C,J)

        return torch.stack([
            x_px, y_px, Z, std_x, std_y, p_in_frame, p_self_occ,
            border_dist, person_scale, bone_scale, overlap, cur_conf_bc,
        ], dim=-1)                                                       # (B,C,J,12)

    @staticmethod
    def exp132_scalar(feats: torch.Tensor) -> torch.Tensor:
        """Reproduce exp132's camera-level visibility scalar from the tensor,
        as a consistency check: mean_j[(1-conf)*p_in_frame*(1-p_self_occ)]."""
        p_in_frame = feats[..., 5]
        p_self_occ = feats[..., 6]
        cur_conf = feats[..., 11]
        return ((1.0 - cur_conf) * p_in_frame * (1.0 - p_self_occ)).mean(dim=-1)


# ---------------------------------------------------------------------------
# Normalisation. Statistics are computed ONCE from the TRAIN split only and
# frozen to disk (docs/reproj_feature_norm.json) so nothing downstream can
# leak val/test statistics into either the probe or the model.
#
# Two feature families need special handling, or a plain median/IQR standard-
# isation mangles them:
#   LOG1P_IDX -- heavy-tailed positive scales (propagated sigmas run 55 px to
#                39,000 px; bone/person scales span two orders of magnitude).
#                log1p first, then standardise.
#   PROB_IDX  -- probabilities in [0,1]. p_in_frame in particular is exactly
#                1.0 for well over half of all (candidate, joint) pairs, so
#                its IQR is ~8e-4 and a robust z-score would drive 26% of the
#                values straight into the clip, destroying precisely the
#                signal exp132 relies on. These are simply mapped to [-1,1].
# ---------------------------------------------------------------------------
NORM_PATH = _HERE.parents[1] / 'config' / 'mpi_inf_3dhp' / 'reproj_feature_norm.json'

LOG1P_IDX = [3, 4, 8, 9]        # sigma_x, sigma_y, person_scale, bone_scale
PROB_IDX = [5, 6, 10, 11]       # p_in_frame, p_self_occ, overlap, cur_conf


def apply_transform(feats):
    """log1p the heavy-tailed positive features; leave the rest alone.
    Works on numpy arrays and torch tensors alike."""
    if isinstance(feats, torch.Tensor):
        out = feats.clone()
        out[..., LOG1P_IDX] = torch.log1p(out[..., LOG1P_IDX].clamp_min(0.0))
        return out
    out = np.array(feats, copy=True)
    out[..., LOG1P_IDX] = np.log1p(np.maximum(out[..., LOG1P_IDX], 0.0))
    return out


def fit_norm_stats(feats: np.ndarray) -> dict:
    """feats: (..., 12) RAW. Robust (median / IQR-based sigma) after the
    transform, so a handful of wild reprojections (joints projected behind a
    candidate camera) don't dominate."""
    flat = apply_transform(feats.reshape(-1, feats.shape[-1]).astype(np.float64))
    med = np.median(flat, axis=0)
    q75, q25 = np.percentile(flat, [75, 25], axis=0)
    scale = (q75 - q25) / 1.349
    scale = np.where(scale < 1e-6, 1.0, scale)
    med[PROB_IDX] = 0.5
    scale[PROB_IDX] = 0.5
    transform = ['log1p' if i in LOG1P_IDX else ('prob' if i in PROB_IDX else 'identity')
                 for i in range(len(FEATURE_NAMES))]
    return dict(names=FEATURE_NAMES, transform=transform,
                center=med.tolist(), scale=scale.tolist())


def load_norm_stats(path=NORM_PATH):
    import json
    d = json.load(open(path))
    return np.asarray(d['center'], dtype=np.float32), np.asarray(d['scale'], dtype=np.float32)


def normalize_features(feats, center, scale, clip=8.0):
    """Transform, standardise, then clip -- clipping matters because joints
    projected behind a candidate camera produce |proj_x| in the millions."""
    out = (apply_transform(feats) - center) / scale
    if isinstance(out, torch.Tensor):
        return out.clamp(-clip, clip)
    return np.clip(out, -clip, clip)


if __name__ == '__main__':
    ckpt = _HERE.parents[1] / 'weights' / 'lifter_bypassed' / 'lift3d_gaussian_mlp_best.pt'
    ex = ReprojJointFeatures(lift3d_ckpt=str(ckpt) if ckpt.exists() else None)
    x = torch.rand(4, 17, 3) - 0.5
    x[..., 2] = torch.rand(4, 17)
    cur = torch.randint(0, 8, (4,))
    f = ex(x, cur)
    print('features', f.shape, 'finite:', bool(torch.isfinite(f).all()))
    for i, n in enumerate(FEATURE_NAMES):
        v = f[..., i]
        print(f'  {i:2d} {n:14s} min={v.min():12.2f} med={v.median():12.2f} max={v.max():12.2f}')
