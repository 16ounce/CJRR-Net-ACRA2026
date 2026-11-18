"""Purpose: Define the reference joint-token camera ranker.

Main functionality: Combine reprojection features and candidate geometry to score views.
"""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[1] / 'utils' / 'activemocap'))

_spec = importlib.util.spec_from_file_location('exp43_rank', _HERE / 'ranking_layers.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_FFN = _mod._FFN

from reproj_joint_features import (  # noqa: E402
    ReprojJointFeatures, N_FEATURES, load_norm_stats, normalize_features, NORM_PATH)
from mpi_camera_geometry import build_camera_constants  # noqa: E402

BASELINE_NORM = 8000.0
CLIP = 8.0

# identical to exp132 / exp131 / exp130
CAM_ANGLES_DEG = np.array([
    [ -5.999342, -24.232821],
    [-36.733883, -24.627060],
    [-120.892868, -28.016209],
    [-48.914967, -18.251417],
    [110.447212, -12.449697],
    [-169.133163, -37.388592],
    [153.881378, -18.736523],
    [ 91.824913, -21.445538],
], dtype=np.float32)


def compute_all_deltas(cam_angles_deg: np.ndarray = CAM_ANGLES_DEG) -> np.ndarray:
    n = cam_angles_deg.shape[0]
    out = np.zeros((n, n, 2), dtype=np.float32)
    for cur in range(n):
        az = cam_angles_deg[:, 0] - cam_angles_deg[cur, 0]
        az = (az + 180) % 360 - 180
        el = cam_angles_deg[:, 1] - cam_angles_deg[cur, 1]
        out[cur] = np.stack([az, el], axis=1) / 180.0
    return out


class JointTokenReprojVisRankNet(nn.Module):
    def __init__(self, n_cands=8, n_joints=17, n_features=3, d_model=128,
                 n_heads=4, n_layers=2, dropout=0.1, cam_ids=None,
                 lift3d_ckpt=None, var_floor=1.0,
                 joint_tokens=True, aux_head=True, d_joint=32,
                 film_mode='none', pool_mode='attn', norm_path=None,
                 cam_angles=None):
        super().__init__()
        assert film_mode in ('none', 'identity', 'geometry', 'input')
        assert pool_mode in ('attn', 'query', 'query_geo')
        self.n_joints = n_joints
        self.joint_tokens = joint_tokens
        self.use_aux = aux_head
        self.film_mode = film_mode
        self.pool_mode = pool_mode

        in_dim = n_joints * n_features
        self.cam_encoder = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU(),
        )
        self.queries = nn.Parameter(torch.randn(1, n_cands, d_model) * 0.02)

        self.geo_proj = nn.Sequential(
            nn.Linear(2, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )
        # cam_angles=None reproduces the 8-camera table byte for byte; the
        # scale study (stage142) passes the 11/14-camera az/el table read from
        # the split's own camera_angles.npy.
        _angles = CAM_ANGLES_DEG if cam_angles is None \
            else np.asarray(cam_angles, dtype=np.float32)
        assert _angles.shape[0] == n_cands, \
            f'cam_angles has {_angles.shape[0]} rows but n_cands={n_cands}'
        self.register_buffer('all_deltas', torch.from_numpy(compute_all_deltas(_angles)))

        _, _, baseline = build_camera_constants(cam_ids=cam_ids) if cam_ids is not None \
            else build_camera_constants()
        self.register_buffer('baseline_matrix', torch.from_numpy(baseline / BASELINE_NORM))
        self.baseline_proj = nn.Sequential(
            nn.Linear(1, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
        )

        # frozen Gaussian lifter + camera constants, shared with the Phase-1 dump
        self.reproj = ReprojJointFeatures(lift3d_ckpt=lift3d_ckpt, cam_ids=cam_ids,
                                          n_joints=n_joints, var_floor=var_floor)

        center, scale = load_norm_stats(norm_path or NORM_PATH)
        self.register_buffer('feat_center', torch.from_numpy(center))
        self.register_buffer('feat_scale', torch.from_numpy(scale))

        if joint_tokens:
            # (B,C,J,12) -> shared MLP -> attention pooling over J -> (B,C,d_model)
            self.joint_mlp = nn.Sequential(
                nn.Linear(N_FEATURES, d_joint), nn.LayerNorm(d_joint), nn.ReLU(),
                nn.Linear(d_joint, d_joint),
            )
            self.joint_attn = nn.Linear(d_joint, 1)
            self.reprojvis_proj = nn.Linear(d_joint, d_model)

            # ---- PHASE 3: candidate-conditional FiLM on the joint tokens ----
            # The Phase-1 diagnosis: the discriminative direction in the
            # 12-channel space is CANDIDATE-CAMERA specific (cam1's within-pair
            # AUC is 0.792 vs cam5 and 0.793 vs cam8 -- essentially identical,
            # so what matters is which candidate, not which pair). A single
            # shared joint MLP cannot hold one direction per camera; a
            # multiplicative per-camera channel gate can, for 512 parameters.
            #
            #   identity : gamma/beta looked up per candidate camera slot
            #   geometry : gamma/beta predicted from the RELATIVE geometry
            #              (az delta, el delta, baseline) -- strictly
            #              camera-agnostic, the control for whether camera
            #              identity per se is needed at all
            #
            # Both are zero-initialised so that at init (1+gamma)=1, beta=0 and
            # the model is bit-identical to the no-FiLM config.
            # fork_rng so that building the FiLM submodule consumes no draws
            # from the main RNG stream -- otherwise every layer constructed
            # after it would get a different init than the no-FiLM config and
            # the ablation would confound "FiLM" with "different seed".
            # ---- query-conditioned joint cross-attention (pool_mode != 'attn') ----
            # Rather than bolting a per-camera parameter table onto a shared
            # MLP, reuse the architecture's own camera-query idiom: let the
            # candidate's camera query attend over its own 17 joint tokens.
            # Conditioning then comes from a function SHARED across cameras and
            # merely evaluated at each camera's query -- which for cam1 (164
            # train frames) is trained on all 4266 frames, not just cam1's.
            #   query     : full q[c] = learned camera query + geometry
            #   query_geo : geometry only, no learned per-camera query
            #               (strictly camera-agnostic control)
            if pool_mode in ('query', 'query_geo'):
                with torch.random.fork_rng(devices=[]):
                    self.joint_q_proj = nn.Linear(d_model, d_joint)
                    self.joint_mha = nn.MultiheadAttention(
                        d_joint, n_heads, dropout=dropout, batch_first=True)

            with torch.random.fork_rng(devices=[]):
                if film_mode == 'identity':
                    self.film = nn.Parameter(torch.zeros(n_cands, 2 * d_joint))
                elif film_mode == 'input':
                    # PHASE 3b. 'identity' modulates AFTER the shared MLP:
                    #     gamma_c * MLP(z)
                    # which rescales output features but cannot change WHICH
                    # direction in the 12-channel input space the MLP reads --
                    # so it cannot express what Phase 1 actually measured.
                    # Modulating the input instead:
                    #     MLP(gamma_c * z) -> first layer computes W diag(gamma_c) z
                    # which IS a per-camera direction in the 12-channel space.
                    self.film = nn.Parameter(torch.zeros(n_cands, 2 * N_FEATURES))
                elif film_mode == 'geometry':
                    self.film_mlp = nn.Sequential(
                        nn.Linear(3, d_joint), nn.ReLU(),
                        nn.Linear(d_joint, 2 * d_joint),
                    )
                    nn.init.zeros_(self.film_mlp[-1].weight)
                    nn.init.zeros_(self.film_mlp[-1].bias)
        else:
            # exp132's original scalar path, unchanged
            self.reprojvis_proj = nn.Sequential(
                nn.Linear(1, d_model // 4), nn.ReLU(),
                nn.Linear(d_model // 4, d_model),
            )

        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)])
        self.ffn = nn.ModuleList([_FFN(d_model, dropout) for _ in range(n_layers)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )
        if aux_head:
            self.delta_head = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 1),
            )

    def train(self, mode=True):
        super().train(mode)
        self.reproj.lifter.eval()
        return self

    def _film_params(self, cur):
        """Returns (gamma, beta), each (B or 1, C, d_joint), or (None, None)."""
        if self.film_mode == 'none':
            return None, None
        if self.film_mode in ('identity', 'input'):
            gb = self.film.unsqueeze(0)                       # (1,C,2*width)
        else:
            geo = torch.cat([self.all_deltas[cur],            # (B,C,2)
                             self.baseline_matrix[cur].unsqueeze(-1)], dim=-1)  # (B,C,3)
            gb = self.film_mlp(geo)                           # (B,C,2*d_joint)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta

    def _pool_joint_tokens(self, feats, cur, q_src=None):
        """(B,C,J,12) raw -> (B,C,d_joint) candidate representation, plus the
        (B,C,J) joint weights for diagnostics."""
        B, C, J, _ = feats.shape
        z = normalize_features(feats, self.feat_center, self.feat_scale, CLIP)
        gamma, beta = self._film_params(cur)
        if gamma is not None and self.film_mode == 'input':
            z = z * (1.0 + gamma).unsqueeze(-2) + beta.unsqueeze(-2)
        h = self.joint_mlp(z)                                 # (B,C,J,d_joint)
        if gamma is not None and self.film_mode != 'input':
            h = h * (1.0 + gamma).unsqueeze(-2) + beta.unsqueeze(-2)

        if self.pool_mode == 'attn':
            w = torch.softmax(self.joint_attn(h).squeeze(-1), dim=-1)   # (B,C,J)
            return (w.unsqueeze(-1) * h).sum(dim=-2), w

        # query-conditioned: the candidate's camera query reads its own joints
        qj = self.joint_q_proj(q_src).reshape(B * C, 1, -1)   # (B*C,1,d_joint)
        kv = h.reshape(B * C, J, -1)                          # (B*C,J,d_joint)
        out, w = self.joint_mha(query=qj, key=kv, value=kv, need_weights=True)
        return out.reshape(B, C, -1), w.reshape(B, C, J)

    def _joint_query_source(self, B, cur):
        """The query the joint cross-attention reads with. 'query' uses the
        full camera query (identity + geometry); 'query_geo' drops the learned
        per-camera term so the model stays strictly camera-agnostic."""
        if self.pool_mode == 'attn':
            return None
        geo = self.geo_proj(self.all_deltas[cur]) \
            + self.baseline_proj(self.baseline_matrix[cur].unsqueeze(-1))
        if self.pool_mode == 'query_geo':
            return geo
        return self.queries.expand(B, -1, -1) + geo

    def joint_attention_weights(self, x, cur):
        """Diagnostic hook: (B, C, J) softmax weights the pooling assigns."""
        assert self.joint_tokens
        batch_idx = torch.arange(x.size(0), device=x.device)
        q_src = self._joint_query_source(x.size(0), cur)
        return self._pool_joint_tokens(self.reproj(x[batch_idx, cur], cur), cur, q_src)[1]

    def forward(self, x: torch.Tensor, cur: torch.Tensor, return_aux: bool = False):
        B, N, J, Feat = x.shape
        tokens = self.cam_encoder(x.view(B * N, J * Feat)).view(B, N, -1)
        batch_idx = torch.arange(B, device=x.device)

        q = self.queries.expand(B, -1, -1) + self.geo_proj(self.all_deltas[cur])
        q = q + self.baseline_proj(self.baseline_matrix[cur].unsqueeze(-1))

        feats = self.reproj(x[batch_idx, cur], cur)                   # (B,C,J,12), raw

        if self.joint_tokens:
            cand, _ = self._pool_joint_tokens(
                feats, cur, self._joint_query_source(B, cur))         # (B,C,d_joint)
            q = q + self.reprojvis_proj(cand)
        else:
            gain = ReprojJointFeatures.exp132_scalar(feats)           # (B,C)
            q = q + self.reprojvis_proj(gain.unsqueeze(-1))

        for cross, ffn, n1, n2 in zip(self.cross_attn, self.ffn, self.norm1, self.norm2):
            attn_out, _ = cross(query=q, key=tokens, value=tokens)
            q = n1(q + attn_out)
            q = n2(q + ffn(q))

        logits = self.score_head(q).squeeze(-1)
        if return_aux:
            delta = self.delta_head(q).squeeze(-1) if self.use_aux else None
            return logits, delta
        return logits


if __name__ == '__main__':
    ck = _HERE.parents[1] / 'weights' / 'lifter_bypassed' / 'lift3d_gaussian_mlp_best.pt'
    for jt in (False, True):
        for ah in (False, True):
            m = JointTokenReprojVisRankNet(lift3d_ckpt=str(ck) if ck.exists() else None,
                                           joint_tokens=jt, aux_head=ah)
            tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
            B = 4
            x = torch.rand(B, 8, 17, 3) - 0.5
            x[..., 2] = torch.rand(B, 8, 17)
            cur = torch.randint(0, 8, (B,))
            out = m(x, cur, return_aux=True)
            lo, de = out
            assert lo.shape == (B, 8) and not torch.isnan(lo).any()
            assert (de is None) if not ah else (de.shape == (B, 8))
            print(f'joint_tokens={jt} aux_head={ah}  trainable={tr:,}  ok')
