"""Purpose: Define the adapted ActiveMoCap-inspired view planner.

Main functionality: Refine a static pose and score candidate cameras by expected information gain.
"""

from pathlib import Path
import importlib.util
import math
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / 'utils' / 'activemocap'))

from reproj_joint_features import (                              # noqa: E402
    ReprojJointFeatures, N_FEATURES, load_norm_stats, normalize_features,
    NORM_PATH, load_lifter)
from mpi_camera_geometry import build_camera_constants           # noqa: E402
from skeleton_config import N_JOINTS                             # noqa: E402
from gaussian_refinement import (           # noqa: E402
    StaticGaussianRefiner)

BASELINE_NORM = 8000.0
CLIP = 8.0
N_CANDS = 8

# feature-channel indices inside the 12-channel field (FEATURE_NAMES order)
F_PROJ_X, F_PROJ_Y, F_DEPTH = 0, 1, 2
F_SIGMA_X, F_SIGMA_Y = 3, 4
F_IN_FRAME, F_SELF_OCC = 5, 6

OBS_SIGMA_PX = 20.0        # base 2D localisation noise of a DETECTED joint
P_VIS_FLOOR = 1e-3         # visibility below this is treated as this
VAR_RATIO_MAX = 1.0e4      # cap on the visibility inflation of R
Z_MIN_CAND_MM = 200.0      # candidate-frame depth below this -> zero weight
DELTA_LOGVAR_RANGE = 2.0   # learned residual bounded to e^+-2 (x0.14 .. x7.4)
DELTA_LOGIT_RANGE = 4.0    # learned visibility residual, in logit units

CONFIGS = {   # name -> (use_refine_reproj, use_bone, geom_R, learned_head)
    'A': (None, None, None, None),        # the original exp142_K, verbatim
    'B': (False, False, False, False),
    'C': (True,  False, False, False),
    'D': (True,  True,  False, False),
    'E': (True,  True,  True,  False),
    'F': (True,  True,  True,  True),
}


def _load_exp142_module():
    spec = importlib.util.spec_from_file_location(
        'exp142', _HERE / 'reference_ranker.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def blockdiag_from_joint_blocks(A):
    """(B, C, J, 3, 3) per-joint 3x3 -> (B, C, 3J, 3J) block diagonal.

    Correct because a 2D measurement of joint j depends on X_j only: the
    information a candidate camera adds is block diagonal even though
    Lambda_prior (bone-coupled) is not.
    """
    B, C, J = A.shape[:3]
    out = A.new_zeros(B, C, J, 3, J, 3)
    ar = torch.arange(J, device=A.device)
    out[:, :, ar, :, ar, :] = A.permute(2, 0, 1, 3, 4)
    return out.reshape(B, C, 3 * J, 3 * J)


def batched_logdet_spd(A, ridge_rel=1e-6, max_tries=5):
    """logdet of a batch of SPD matrices via Cholesky, with an escalating
    ridge. Returns (logdet, ok_mask). Never calls torch.inverse / .det."""
    flat = A.reshape(-1, A.shape[-2], A.shape[-1])
    n = flat.shape[-1]
    eye = torch.eye(n, device=A.device, dtype=A.dtype)
    bump = flat.diagonal(dim1=-2, dim2=-1).abs().amax(-1).clamp_min(1e-12) * ridge_rel
    cur = flat
    ok = torch.zeros(flat.shape[0], dtype=torch.bool, device=A.device)
    logdet = torch.zeros(flat.shape[0], dtype=A.dtype, device=A.device)
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(cur)
        good = (info == 0) & torch.isfinite(L).flatten(1).all(1)
        ld = 2.0 * torch.log(L.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30)).sum(-1)
        newly = good & ~ok
        logdet = torch.where(newly, ld, logdet)
        ok = ok | good
        if bool(ok.all()):
            break
        cur = torch.where((~ok).view(-1, 1, 1), cur + bump.view(-1, 1, 1) * eye, cur)
        bump = bump * 100.0
    return logdet.reshape(A.shape[:-2]), ok.reshape(A.shape[:-2])


class JointUncertaintyHead(nn.Module):
    """Shared per-(candidate, joint) uncertainty head.

    in  : 12 normalised reprojection channels
          + joint identity embedding (d_emb)
          + relative geometry (az delta, el delta, baseline) of cur -> candidate
    out : visibility_logits (B, C, J)      RESIDUAL on the geometric logit
          delta_log_var     (B, C, J, 2)   RESIDUAL on log R_geometry

    The geometry is the PRIOR; the network only corrects it. Both residuals are
    bounded (tanh), so a badly-trained head can never blow the planner up, and
    a zero-initialised final layer means F starts numerically identical to E.
    """

    def __init__(self, n_joints=N_JOINTS, d_emb=16, d_hidden=64,
                 n_features=N_FEATURES,
                 dlogvar_range=DELTA_LOGVAR_RANGE,
                 dlogit_range=DELTA_LOGIT_RANGE):
        super().__init__()
        self.dlogvar_range = dlogvar_range
        self.dlogit_range = dlogit_range
        self.joint_emb = nn.Embedding(n_joints, d_emb)
        in_dim = n_features + d_emb + 3
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.LayerNorm(d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.out = nn.Linear(d_hidden, 3)      # [vis_logit, dlogvar_u, dlogvar_v]
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, z_feats, geo):
        """z_feats (B,C,J,12) NORMALISED; geo (B,C,3) -> (vis (B,C,J),
        dlogvar (B,C,J,2))."""
        B, C, J, _ = z_feats.shape
        emb = self.joint_emb.weight.view(1, 1, J, -1).expand(B, C, J, -1)
        g = geo.unsqueeze(2).expand(B, C, J, 3)
        h = self.mlp(torch.cat([z_feats, emb, g], dim=-1))
        o = self.out(h)
        vis = torch.tanh(o[..., 0]) * self.dlogit_range
        dlv = torch.tanh(o[..., 1:]) * self.dlogvar_range
        return vis, dlv


class StaticActiveViewPlanner(nn.Module):
    """Static Bayesian next-best-view planner over the 8 fixed rig cameras."""

    def __init__(self, config='F', lift3d_ckpt=None, cam_ids=None,
                 n_joints=N_JOINTS, n_cands=N_CANDS,
                 utility_mode='global_logdet',
                 obs_sigma_px=OBS_SIGMA_PX, p_vis_floor=P_VIS_FLOOR,
                 var_ratio_max=VAR_RATIO_MAX, z_min_cand=Z_MIN_CAND_MM,
                 vis_in_R=True, norm_path=None, refiner_kwargs=None,
                 exp142k_ckpt=None, cam_angles=None, seed=None):
        super().__init__()
        assert config in CONFIGS, f'config must be one of {list(CONFIGS)}'
        assert utility_mode in ('global_logdet', 'uncertainty_weighted_joints')
        self.config = config
        self.utility_mode = utility_mode
        self.J = n_joints
        self.C = n_cands
        self.D = 3 * n_joints
        self.obs_sigma_px = obs_sigma_px
        self.p_vis_floor = p_vis_floor
        self.var_ratio_max = var_ratio_max
        self.z_min_cand = z_min_cand
        self.vis_in_R = vis_in_R

        # ---- config A: hand everything back to the original model ----------
        if config == 'A':
            exp142 = _load_exp142_module()
            self.exp142k = exp142.JointTokenReprojVisRankNet(
                n_cands=n_cands, n_joints=n_joints, n_features=3,
                d_model=128, n_heads=4, n_layers=2, dropout=0.1,
                cam_ids=cam_ids, lift3d_ckpt=lift3d_ckpt, var_floor=1.0,
                joint_tokens=True, aux_head=False, d_joint=32,
                film_mode='input', pool_mode='attn', norm_path=norm_path,
                cam_angles=cam_angles)
            if exp142k_ckpt is not None:
                ck = torch.load(str(exp142k_ckpt), map_location='cpu',
                                weights_only=False)
                self.exp142k.load_state_dict(ck['model_state_dict'])
            for p in self.exp142k.parameters():
                p.requires_grad = False
            self.exp142k.eval()
            return

        use_reproj, use_bone, geom_R, learned = CONFIGS[config]
        self.geom_R = geom_R
        self.learned = learned

        # ---- shared, single-source geometry + frozen lifter -----------------
        self.reproj = ReprojJointFeatures(lift3d_ckpt=lift3d_ckpt,
                                          cam_ids=cam_ids, n_joints=n_joints,
                                          var_floor=1.0)
        center, scale = load_norm_stats(norm_path or NORM_PATH)
        self.register_buffer('feat_center', torch.from_numpy(center))
        self.register_buffer('feat_scale', torch.from_numpy(scale))

        exp142 = _load_exp142_module()
        angles = exp142.CAM_ANGLES_DEG if cam_angles is None \
            else np.asarray(cam_angles, dtype=np.float32)
        self.register_buffer('all_deltas',
                             torch.from_numpy(exp142.compute_all_deltas(angles)))
        _, _, baseline = build_camera_constants(cam_ids=cam_ids) \
            if cam_ids is not None else build_camera_constants()
        self.register_buffer('baseline_matrix',
                             torch.from_numpy(baseline / BASELINE_NORM))

        rk = dict(use_reproj=use_reproj, use_bone=use_bone)
        rk.update(refiner_kwargs or {})
        self.refiner = StaticGaussianRefiner(n_joints=n_joints, **rk)

        if learned:
            # fork_rng: constructing the head must not shift any other module's
            # initialisation, so E vs F differ ONLY by the head.
            with torch.random.fork_rng(devices=[]):
                if seed is not None:
                    torch.manual_seed(seed)
                self.unc_head = JointUncertaintyHead(n_joints=n_joints)

        # learned affine used ONLY to map an unbounded logdet (nats) onto the
        # ranking losses' logit scale; it is monotone, so it cannot change
        # argmax or any ranking metric.
        self.score_scale = nn.Parameter(torch.ones(()))
        self.score_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer('eye3', torch.eye(3))

    def train(self, mode=True):
        super().train(mode)
        if self.config == 'A':
            self.exp142k.eval()
        else:
            self.reproj.lifter.eval()
        return self

    # -- measurement model --------------------------------------------------

    def _measurement_model(self, feats, z_feats, geo):
        """(B,C,J,12) field -> p_detect (B,C,J) and var_2d (B,C,J,2) px^2.

        Geometry is always the prior. Configs B-D use a homoscedastic R and
        p_detect = 1; E conditions both on p_visible = p_in_frame *
        (1 - p_self_occ); F adds the learned residual on top of E's values.

        A HARD geometric gate (candidate-frame depth > z_min, all channels
        finite) is applied in EVERY config -- a joint that lands behind the
        candidate camera is not "unlikely", it is impossible, and letting it
        contribute information would be a bug rather than an ablation.
        """
        B, C, J, _ = feats.shape
        Z = feats[..., F_DEPTH]
        valid = (Z > self.z_min_cand) & torch.isfinite(feats).all(-1)

        var_base = feats.new_full((B, C, J, 2), self.obs_sigma_px ** 2)
        if self.geom_R:
            p_vis = (feats[..., F_IN_FRAME] * (1.0 - feats[..., F_SELF_OCC])
                     ).clamp(self.p_vis_floor, 1.0)
            ratio = (1.0 / p_vis).clamp_max(self.var_ratio_max)
            var = var_base * ratio.unsqueeze(-1) if self.vis_in_R else var_base
            p_detect = p_vis
        else:
            p_vis = feats.new_ones(B, C, J)
            var = var_base
            p_detect = p_vis

        vis_logit = torch.log(p_vis.clamp(1e-6, 1 - 1e-6)) \
            - torch.log1p(-p_vis.clamp(1e-6, 1 - 1e-6))
        if self.learned:
            dvis, dlogvar = self.unc_head(z_feats, geo)
            vis_logit = vis_logit + dvis
            p_detect = torch.sigmoid(vis_logit)
            var = var * torch.exp(dlogvar)

        p_detect = torch.where(valid, p_detect, torch.zeros_like(p_detect))
        vis_logit = torch.where(valid, vis_logit,
                                torch.full_like(vis_logit, -20.0))
        var = var.clamp_min(1e-6)
        return p_detect, var, valid, vis_logit

    # -- Jacobian, straight off the extractor's own projected points ---------

    def _candidate_jacobian(self, feats, cur):
        """(B, C, J, 2, 3) d(u,v)_c / dX_cur, in the CURRENT camera's frame.

        chain rule:  X_cand = R_c R_cur^T (X_cur - T_cur) + T_c
                     d(u,v)/dX_cur = Jpix(X_cand) @ (R_c R_cur^T)
        Jpix is written in terms of the field's own (u, v, Z) so the
        linearisation point is exactly the extractor's projected point.
        """
        u, v, Z = feats[..., F_PROJ_X], feats[..., F_PROJ_Y], feats[..., F_DEPTH]
        Z = Z.clamp_min(1e-3)
        fx = self.reproj.cam_K[:, 0, 0].view(1, -1, 1)
        fy = self.reproj.cam_K[:, 1, 1].view(1, -1, 1)
        cx = self.reproj.cam_K[:, 0, 2].view(1, -1, 1)
        cy = self.reproj.cam_K[:, 1, 2].view(1, -1, 1)
        zero = torch.zeros_like(Z)
        Jpix = torch.stack([
            torch.stack([fx / Z, zero, -(u - cx) / Z], dim=-1),
            torch.stack([zero, fy / Z, -(v - cy) / Z], dim=-1),
        ], dim=-2)                                             # (B,C,J,2,3)
        # M[b,c] = R_c @ R_cur[b]^T
        M = torch.einsum('cij,bjk->bcik', self.reproj.cam_R,
                         self.reproj.cam_R[cur].transpose(1, 2))
        return torch.einsum('bcjxi,bcik->bcjxk', Jpix, M)

    @staticmethod
    def _joint_blocks(M):
        """(B, 3J, 3J) -> (B, J, 3, 3), the per-joint diagonal blocks."""
        B, D = M.shape[0], M.shape[-1]
        J = D // 3
        ar = torch.arange(J, device=M.device)
        blk = M.reshape(B, J, 3, J, 3)[:, ar, :, ar, :]      # (J, B, 3, 3)
        blk = blk.permute(1, 0, 2, 3)
        return 0.5 * (blk + blk.transpose(-1, -2))

    # -- utility ------------------------------------------------------------

    def _information_gain(self, Lambda_prior, Sigma_prior, A):
        """A (B,C,J,3,3) added information -> IG (B,C) for the active mode."""
        B, C = A.shape[0], A.shape[1]
        if self.utility_mode == 'global_logdet':
            post = Lambda_prior.unsqueeze(1) + blockdiag_from_joint_blocks(A)
            ld_post, _ = batched_logdet_spd(post)                  # (B,C)
            ld_prior, _ = batched_logdet_spd(Lambda_prior.unsqueeze(1))
            return ld_post - ld_prior, None

        # per-joint MARGINAL covariance (block of Sigma, not of Lambda) ->
        # Lambda_j via Cholesky solve, so IG_j is a true marginal information
        # gain for joint j rather than a conditional one.
        J = self.J
        Sj = self._joint_blocks(Sigma_prior)                       # (B,J,3,3)
        eye = self.eye3.expand(B, J, 3, 3)
        Lj, info = torch.linalg.cholesky_ex(
            Sj + 1e-9 * Sj.diagonal(dim1=-2, dim2=-1).abs().amax(-1)
            .clamp_min(1e-12).view(B, J, 1, 1) * eye)
        bad = (info != 0).view(B, J, 1, 1)
        Lj = torch.where(bad, eye, Lj)
        Lam_j = torch.cholesky_solve(eye, Lj)                      # (B,J,3,3)
        Lam_j = 0.5 * (Lam_j + Lam_j.transpose(-1, -2))
        ld_prior_j, _ = batched_logdet_spd(Lam_j)                  # (B,J)
        ld_post_j, _ = batched_logdet_spd(
            Lam_j.unsqueeze(1) + A)                                # (B,C,J)
        ig_j = ld_post_j - ld_prior_j.unsqueeze(1)                 # (B,C,J)

        w = Sj.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(0.0)   # trace, (B,J)
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-12)
        return (w.unsqueeze(1) * ig_j).sum(-1), ig_j

    # -- forward ------------------------------------------------------------

    def forward(self, x, cur, quality_mask=None, current_camera_penalty=0.0,
                return_aux=False):
        """
        Args:
            x    : (B, 8, 17, 3) per-camera keypoints; only the observed slot
                   needs to be populated (the rest are zeros, exactly as in
                   exp142_K's K=1 deployment protocol)
            cur  : (B,) long, observed camera index in SELECTED_CAM_IDS order
            quality_mask : (B, 8) bool or None -- False gets -inf
            current_camera_penalty : subtracted from the observed camera's own
                   score (the static one-step formulation has no notion of
                   "already observed", so staying put is otherwise rewarded
                   for information it has, in effect, already collected)
        Returns:
            scores (B, 8) float, higher = better   (+ aux dict if requested)
        """
        B = x.shape[0]
        batch_idx = torch.arange(B, device=x.device)

        if self.config == 'A':
            scores = self.exp142k(x, cur)
            return self._finalise(scores, cur, quality_mask,
                                  current_camera_penalty,
                                  dict() if return_aux else None)

        cur_kp = x[batch_idx, cur]                                 # (B,J,3)
        cur_conf = cur_kp[..., 2]
        with torch.no_grad():
            mu, var = self.reproj.lifter(cur_kp)                   # (B,J,3)

        K_cur = self.reproj.cam_K[cur]                             # (B,3,3)
        ref = self.refiner(mu, var, cur_kp, K_cur)
        X = ref['X_refined']                                       # (B,J,3)
        Lambda = ref['Lambda']                                     # (B,3J,3J)
        Sigma = ref['Sigma_current']

        # per-joint marginal 3x3 blocks, in the CURRENT camera's frame, fed to
        # the extractor so the field's propagated sigmas describe the REFINED
        # posterior rather than the raw lifter variance
        cov_j = self._joint_blocks(Sigma)                          # (B,J,3,3)

        feats = self.reproj.field_from_gaussian(X, cov_j, cur_conf, cur)
        z_feats = normalize_features(feats, self.feat_center, self.feat_scale, CLIP)

        geo = torch.cat([self.all_deltas[cur],
                         self.baseline_matrix[cur].unsqueeze(-1)], dim=-1)  # (B,C,3)
        p_detect, var_2d, valid, vis_logit = self._measurement_model(
            feats, z_feats, geo)

        Jc = self._candidate_jacobian(feats, cur)                  # (B,C,J,2,3)
        w = (p_detect.unsqueeze(-1) / var_2d)                      # (B,C,J,2)
        A = torch.einsum('bcjxi,bcjx,bcjxk->bcjik', Jc, w, Jc)     # (B,C,J,3,3)
        A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A = 0.5 * (A + A.transpose(-1, -2))

        ig, ig_j = self._information_gain(Lambda, Sigma, A)         # (B,C)
        scores = torch.nan_to_num(ig, nan=0.0, posinf=0.0, neginf=0.0)

        aux = None
        if return_aux:
            aux = dict(feats=feats, p_detect=p_detect, var_2d=var_2d,
                       vis_logit=vis_logit,
                       valid=valid, ig=ig, ig_per_joint=ig_j,
                       X_refined=X, Sigma_current=Sigma, Lambda=Lambda,
                       reproj_px_before=ref['reproj_px_before'],
                       reproj_px_after=ref['reproj_px_after'],
                       all_blind=(p_detect.sum(-1) <= 1e-8))
        return self._finalise(scores, cur, quality_mask,
                              current_camera_penalty, aux)

    def _finalise(self, scores, cur, quality_mask, penalty, aux):
        if penalty:
            pen = torch.zeros_like(scores)
            pen[torch.arange(scores.size(0), device=scores.device), cur] = penalty
            scores = scores - pen
        if quality_mask is not None:
            scores = scores.masked_fill(~quality_mask.bool(),
                                        float('-inf'))
        if aux is None:
            return scores
        aux['scores'] = scores
        return scores, aux

    # -- convenience --------------------------------------------------------

    def calibrated_logits(self, scores):
        """Monotone affine map of the raw nats onto a logit scale, for the
        ranking/CE losses only. Monotone => argmax and every ranking metric
        are untouched."""
        return scores * self.score_scale + self.score_bias


def build_planner(config, lift3d_ckpt, exp142k_ckpt=None, **kw):
    return StaticActiveViewPlanner(config=config, lift3d_ckpt=lift3d_ckpt,
                                   exp142k_ckpt=exp142k_ckpt, **kw)


if __name__ == '__main__':
    W = Path('/srv/scratch/z5511538/exp142K_walkthrough/weights')
    lifter = str(W / 'lift3d_gaussian_mlp' / 'best.pt')
    torch.manual_seed(0)
    B = 4
    x = torch.rand(B, 8, 17, 3) - 0.5
    x[..., 2] = torch.rand(B, 8, 17)
    cur = torch.randint(0, 8, (B,))
    for cfg in ('B', 'C', 'D', 'E', 'F'):
        for mode in ('global_logdet', 'uncertainty_weighted_joints'):
            m = StaticActiveViewPlanner(config=cfg, lift3d_ckpt=lifter,
                                        utility_mode=mode, seed=0).eval()
            with torch.no_grad():
                s = m(x, cur)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            print(f'{cfg} {mode:28s} scores {tuple(s.shape)} '
                  f'finite={bool(torch.isfinite(s).all())} trainable={n}')
