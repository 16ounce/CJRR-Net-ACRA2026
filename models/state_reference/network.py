"""Purpose: Define the reference state-aware ranker.

Main functionality: Score candidate cameras using pose state and reprojection features.
"""

from pathlib import Path
import importlib.util
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from .ffn import _FFN

from .features import (  # noqa: E402
    ReprojJointFeatures, N_FEATURES, load_norm_stats, normalize_features, NORM_PATH)
from .geometry import build_camera_constants  # noqa: E402

BASELINE_NORM = 8000.0
CLIP = 8.0


def _load_semgcn():
    """Imported lazily so the shipped cam_enc='mlp' path keeps exactly the
    dependency set nbv_pipeline already loads."""
    raise NotImplementedError(
        "cam_enc=semgcn is not shipped in this package: it is reachable "
        "only from config K_semgcn, which this configuration never uses.")

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
                 cam_angles=None, zero_channels=None,
                 learned_query=True, view_mask=False,
                 cam_enc='mlp', semgcn_hid=32, semgcn_out=8, semgcn_blocks=2,
                 semgcn_norm='ln', change_head=False, motion_head=False,
                 bias_mode=None, cam_Rs=None, cam_Ts=None, cam_Ks=None,
                 baseline_matrix=None, lifter_calib_scale=None, lifter_calib_offset=None,
                 lifter=None,
                 branch_obs=True, branch_cand=True, camq_mode='full'):
        super().__init__()
        assert film_mode in ('none', 'identity', 'geometry', 'input', 'concat', 'query_add')
        assert pool_mode in ('attn', 'mean', 'max', 'query', 'query_geo')
        assert bias_mode in (None, 'aux', 'feature')
        # ---- exp178 (2026-08-12): WHOLE-BRANCH ablation switches ----
        # Everything above this line ablates INSIDE one of the three branches
        # that build the candidate query. These three switches delete a branch
        # outright, which is the first-layer table the paper needs:
        #
        #   branch_obs=False   the observed-view cross-attention never runs.
        #                      The FFN/LayerNorm depth is KEPT (exp160's
        #                      convention) so the only thing removed is the
        #                      attention over the other views' context tokens,
        #                      not two residual blocks of depth. cam_encoder
        #                      and cross_attn are then deleted, since nothing
        #                      reads them. NOTE this is NOT "no visual input":
        #                      the candidate-joint branch still reads the
        #                      reference view's own keypoints. Name it
        #                      "w/o multi-view observation fusion".
        #   branch_cand=False  F_{c,j} is never computed and reprojvis_proj is
        #                      never applied -- Q_c = Q^cam_c. self.reproj (the
        #                      FROZEN lifter, 0 trainable params) is kept so the
        #                      checkpoint's buffer set is unchanged, but it is
        #                      not called, so this arm is also the cheap one.
        #   camq_mode          which parts of Q^cam_c survive:
        #                      'full' queries + geo(daz,del) + baseline  [K]
        #                      'id'   learned per-camera table only
        #                      'geo'  relative geometry only (== learned_query
        #                             =False, kept separate so the deleted
        #                             parameters are actually deleted here)
        #                      'none' Q^cam_c = 0, whole branch removed
        #
        # CAVEAT that must reach the paper: camq_mode='none' does NOT make the
        # model camera-identity-free. film_mode='input' is a per-candidate-slot
        # gamma/beta lookup table, i.e. identity conditioning living inside the
        # candidate-joint branch, and F_{c,j} is candidate-specific by
        # construction. 'none' removes the EXPLICIT query, not all identity.
        assert camq_mode in ('full', 'id', 'geo', 'none')
        assert branch_obs or branch_cand or camq_mode != 'none', \
            'all three branches deleted: the model would have no input at all'
        assert camq_mode == 'full' or pool_mode in ('attn', 'mean', 'max'), \
            f'pool_mode={pool_mode!r} reads the camera query inside the joint ' \
            f'pooling (_joint_query_source), so it is incompatible with ' \
            f'camq_mode={camq_mode!r}'
        assert branch_obs or not motion_head, \
            'motion_head reads the observed-view context token tokens[cur], ' \
            'which branch_obs=False never computes'
        self.branch_obs = branch_obs
        self.branch_cand = branch_cand
        self.camq_mode = camq_mode
        self.bias_mode = bias_mode
        self.n_cands = n_cands
        self.d_model = d_model
        self.n_joints = n_joints
        self.joint_tokens = joint_tokens
        self.use_aux = aux_head
        self.film_mode = film_mode
        self.pool_mode = pool_mode
        # exp168 (2026-07-27): channels to blank out of the 12 per-joint
        # reprojection features. Registered as a buffer so it moves with the
        # model and is saved in the checkpoint -- an arm that silently reverts
        # to the full feature set on reload would be undetectable.
        self.zero_channels = zero_channels
        # exp169 (2026-07-27). learned_query=False drops the per-camera query
        # TABLE from the candidate query, leaving it a pure function of the
        # relative geometry (az/el delta + baseline). self.queries is still
        # CONSTRUCTED so the RNG stream and init of every later module are
        # unchanged; its 1024 entries simply stop being read. view_mask=True
        # adds the key_padding_mask exp159 measured as null on its own -- it is
        # carried here only because the lite composition asks for it.
        self.learned_query = learned_query
        self.view_mask = view_mask
        if zero_channels is not None:
            keep = torch.ones(N_FEATURES)
            for c in zero_channels:
                keep[c] = 0.0
            self.register_buffer('channel_keep', keep)

        in_dim = n_joints * n_features
        self.cam_encoder = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU(),
        )
        # exp176 (2026-07-28): swap the context encoder for SemGCN (Zhao et al.
        # 2019) -- message passing on the COCO skeleton with learned per-edge
        # weights, instead of flattening the 17 joints into one 51-vector. This
        # is the ONE module in exp142_K that has never been touched: it is
        # line-for-line identical in exp43/102/115/131/132/142/151.
        #
        # The flat encoder above is still CONSTRUCTED first, so it consumes the
        # same RNG draws it always did, and SemGCN is built inside fork_rng --
        # the hygiene rule this project already established for the FiLM
        # (see the Phase-3 note below): every module built AFTER this point then
        # gets an initialisation bit-identical to config K's at the same seed,
        # so exp176 vs exp142_K is not confounded with "different seed". The
        # unused flat encoder is then deleted so it is neither trained nor
        # counted. See the project memory note on init sensitivity: two
        # forward-identical models differ by 0.51 pp top-1 (p=0.021) on init RNG
        # alone, which is larger than most effects measured here.
        assert cam_enc in ('mlp', 'semgcn')
        self.cam_enc = cam_enc
        if cam_enc == 'semgcn':
            with torch.random.fork_rng(devices=[]):
                self.semgcn = _load_semgcn().SemGCNCamEncoder(
                    in_features=n_features, hid=semgcn_hid,
                    out_features=semgcn_out, n_blocks=semgcn_blocks,
                    d_model=d_model, dropout=dropout, norm=semgcn_norm,
                    n_joints=n_joints)
            del self.cam_encoder
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

        # baseline_matrix=None (every existing call site) resolves via
        # cam_ids into MPI's own calibration, byte-identical to before this
        # param existed. Passing it explicitly (together with cam_Rs/Ts/Ks
        # below) is for camera rigs with no MPI cam_id -- the AIST++
        # transfer-learning line's own 8-camera geometry.
        if baseline_matrix is None:
            _, _, baseline = build_camera_constants(cam_ids=cam_ids) if cam_ids is not None \
                else build_camera_constants()
            baseline_matrix = baseline
        self.register_buffer('baseline_matrix', torch.from_numpy(baseline_matrix / BASELINE_NORM))
        self.baseline_proj = nn.Sequential(
            nn.Linear(1, d_model // 4), nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
        )

        # frozen Gaussian lifter + camera constants, shared with the Phase-1 dump.
        # lifter=None (every existing call site) resolves via lift3d_ckpt into
        # Lift3DGaussianMLP, byte-identical to before this param existed.
        # Passing an already-built nn.Module (2026-08-09: stage164, the
        # phase1b L4 multi-hypothesis lifter) bypasses lift3d_ckpt entirely --
        # ReprojJointFeatures itself already supported this, this class just
        # never threaded it through.
        self.reproj = ReprojJointFeatures(lift3d_ckpt=lift3d_ckpt, cam_ids=cam_ids,
                                          n_joints=n_joints, var_floor=var_floor,
                                          Rs=cam_Rs, Ts=cam_Ts, Ks=cam_Ks,
                                          lifter_calib_scale=lifter_calib_scale,
                                          lifter_calib_offset=lifter_calib_offset,
                                          lifter=lifter)

        center, scale = load_norm_stats(norm_path or NORM_PATH)
        self.register_buffer('feat_center', torch.from_numpy(center))
        self.register_buffer('feat_scale', torch.from_numpy(scale))

        if joint_tokens:
            # (B,C,J,12) -> shared MLP -> attention pooling over J -> (B,C,d_model)
            # exp172: film_mode='concat' is the "just concatenate the camera
            # identity" alternative. It widens the shared MLP's input by an
            # 8-way one-hot instead of modulating the 12 channels, so the first
            # layer computes W_z z + W_c onehot_c -- a per-camera BIAS in the
            # hidden space. That is strictly the additive half of what input
            # FiLM does: FiLM's gamma gives W diag(gamma_c) z, a per-camera
            # DIRECTION in the 12-channel space, which no amount of
            # concatenated identity can produce. Whether the multiplicative
            # half is load-bearing is exactly what this arm measures.
            # 2026-08-05: optional ViTPose-bias head, per (candidate, joint)
            # token -- predicts how far off ViTPose's own 2D detection would
            # be in that candidate's view (dx,dy, normalised units), from the
            # SAME 12-channel geometric features (occlusion/depth/in-frame)
            # this MLP already reads. probe_vitpose_bias_head.py verified the
            # signal is real even for candidates never observed (+32.4% MSE
            # vs a zero-bias baseline, test split) -- unlike the future-motion
            # head, whose displacement prediction collapsed to ~0 correlation
            # with ground truth. Two modes:
            #   'aux'     : predicted, MSE-supervised, but NOT read by
            #               joint_mlp -- pure regulariser (mirrors motion_head
            #               with use_future_motion=False).
            #   'feature' : predicted, MSE-supervised, AND concatenated onto z
            #               before joint_mlp -- the joint-token pathway gets
            #               an explicit "how much do I distrust this
            #               candidate's detection here" channel per joint.
            #               joint_mlp's input therefore widens by 2 (fixed at
            #               construction, since the layer shape can't depend
            #               on a runtime flag).
            if bias_mode is not None:
                self.bias_head = nn.Sequential(
                    nn.Linear(N_FEATURES, 64), nn.ReLU(),
                    nn.Linear(64, 64), nn.ReLU(),
                    nn.Linear(64, 2),
                )
            _in = (N_FEATURES + (n_cands if film_mode == 'concat' else 0)
                   + (2 if bias_mode == 'feature' else 0))
            self.joint_mlp = nn.Sequential(
                nn.Linear(_in, d_joint), nn.LayerNorm(d_joint), nn.ReLU(),
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
                elif film_mode == 'query_add':
                    # exp174 (2026-07-28). Ports exp151's mechanism onto config
                    # K's skeleton: exp151_additive_joint_field_camagnostic.py
                    # injects the candidate's own query ADDITIVELY into the
                    # per-joint hidden state before any shared joint-level
                    # computation runs --
                    #   h = reproj_proj(feats) + joint_emb + q.unsqueeze(2)
                    # -- so the correspondence between "which candidate" and
                    # "what the joint features mean" is given by construction,
                    # not learned by a downstream module. That is the mechanism
                    # docs/input_representation.txt's exp120 post-mortem blames
                    # for exp120's failure (heatmap features concatenated into
                    # a SHARED token, candidate identity left for the query to
                    # infer) and credits for exp151's success.
                    #
                    # 'input' (config K) already conditions the SAME insertion
                    # point (before joint_mlp) -- but MULTIPLICATIVELY, via a
                    # per-camera gamma/beta lookup table indexed by `cur`. This
                    # arm swaps that for exp151's ADDITIVE mechanism: the actual
                    # running candidate query q (geometry + baseline, and the
                    # learned per-camera term when learned_query=True) is
                    # projected down to the 12-channel feature space and added
                    # to z, before the shared joint_mlp -- i.e. "does additive
                    # injection of the real query beat a multiplicative
                    # per-camera lookup table at the same insertion point,
                    # holding everything else in config K fixed."
                    #
                    # Zero-initialised: at init the injected term is exactly 0,
                    # so this arm is bit-identical to config B (no conditioning
                    # at all) before training -- the query has to EARN its way
                    # into the joint pathway, not start already influencing it.
                    self.query_to_feat = nn.Linear(d_model, N_FEATURES)
                    nn.init.zeros_(self.query_to_feat.weight)
                    nn.init.zeros_(self.query_to_feat.bias)
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
        # 2026-08-05: optional frame-level auxiliary head for the futunc-aux
        # experiment -- predicts one scalar per FRAME (not per candidate),
        # from the mean-pooled candidate queries, e.g. "does the oracle camera
        # change between t and t+25". Off by default (change_head=False);
        # every existing config is unaffected, this module simply isn't built.
        self.use_change_head = change_head
        if change_head:
            self.change_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )
        # 2026-08-05: optional motion-residual head -- predicts the observed
        # camera's own per-joint 3D displacement (t -> t+25, mm, in that
        # camera's frame) from its encoded context token. Two independent
        # experiments share this module:
        #   motion-aux        : delta is supervised (MSE vs GT residual) but
        #                        never touched by the rest of the forward
        #                        pass -- a pure regulariser, like change_head.
        #   future_motion_head: delta is ADDED to the frozen lifter's current
        #                        3D mean and the sum is fed back into
        #                        self.reproj via its existing `state=`
        #                        argument (built for exp162's temporal
        #                        fusion), so the joint-token features the
        #                        model reasons over describe the PREDICTED
        #                        t+25 geometry, not the current one. Which
        #                        mode is active is a forward()-time flag
        #                        (use_future_motion), not a second ctor arg,
        #                        so the same trained head can be inspected
        #                        both ways.
        self.use_motion_head = motion_head
        if motion_head:
            self.motion_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.ReLU(),
                nn.Linear(d_model // 2, n_joints * 3),
            )

        # ---- exp178: delete the modules a deleted branch made unreachable ----
        # Done HERE, after every module has been constructed, and never with a
        # guard around the construction itself: that is the same hygiene rule
        # the FiLM's fork_rng follows. Construction order is byte-identical to
        # config K's, so every surviving module is initialised bit-for-bit as
        # config K would initialise it at the same seed, and each arm is a
        # genuine paired contrast against K rather than "K at a different RNG
        # draw" (the measured init-sensitivity floor is 0.51 pp top-1).
        #
        # Deleting rather than merely not-reading also keeps `trainable_params`
        # honest -- the paper reports it per row, and exp169's convention of
        # leaving self.queries built-but-unread would have inflated the
        # camera-query rows by 1,024 parameters that receive no gradient.
        if not branch_obs:
            del self.cross_attn                      # 2x MHA(128,4) = 132,096
            if self.cam_enc == 'semgcn':
                del self.semgcn
            else:
                del self.cam_encoder                 # 51->128->128 = 23,680
        if not branch_cand:
            assert not motion_head and bias_mode is None, \
                'motion/bias heads live inside the candidate-joint branch'
            if joint_tokens:
                del self.joint_mlp, self.joint_attn
                if film_mode in ('identity', 'input'):
                    del self.film
                elif film_mode == 'geometry':
                    del self.film_mlp
                elif film_mode == 'query_add':
                    del self.query_to_feat
                if pool_mode in ('query', 'query_geo'):
                    del self.joint_q_proj, self.joint_mha
            del self.reprojvis_proj
            # self.reproj is KEPT: 0 trainable params (the lifter is frozen at
            # load_lifter) and its buffers are part of every existing
            # checkpoint's state_dict. It is simply never called.
        if camq_mode in ('geo', 'none'):
            del self.queries
        if camq_mode in ('id', 'none'):
            del self.geo_proj, self.baseline_proj
            # all_deltas / baseline_matrix stay registered: they are geometry
            # BUFFERS, they cost nothing, film_mode='geometry' and
            # pool_mode='query*' still read them, and stage144's --init_ckpt
            # filter keys off their names.

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

    def _pool_joint_tokens(self, feats, cur, q_src=None, q_full=None):
        """(B,C,J,12) raw -> (B,C,d_joint) candidate representation, plus the
        (B,C,J) joint weights for diagnostics.

        q_full: the running candidate query (B,C,d_model), only read when
        film_mode=='query_add' -- exp151-style additive injection at the same
        insertion point config K's FiLM uses, see __init__'s exp174 note.
        """
        B, C, J, _ = feats.shape
        z = normalize_features(feats, self.feat_center, self.feat_scale, CLIP)
        bias_pred = self.bias_head(z) if self.bias_mode is not None else None  # (B,C,J,2), off raw z12
        if self.film_mode == 'concat':
            eye = torch.eye(C, device=z.device, dtype=z.dtype)
            z = torch.cat([z, eye.view(1, C, 1, C).expand(B, C, J, C)], dim=-1)
        if self.zero_channels is not None:
            # exp168. Zeroed AFTER normalisation, not sliced out: the joint MLP
            # keeps its 12-wide input so the parameter count, the init and the
            # RNG stream are identical to the full model, and the arm differs
            # only in whether the channel carries information. A constant-zero
            # input is absorbed by the first layer's bias, so this removes the
            # channel's information exactly.
            z = z * self.channel_keep
        if self.film_mode == 'query_add':
            z = z + self.query_to_feat(q_full).unsqueeze(-2)  # (B,C,1,12) broadcast over J
        gamma, beta = (None, None) if self.film_mode in ('concat', 'query_add') \
            else self._film_params(cur)
        if gamma is not None and self.film_mode == 'input':
            z = z * (1.0 + gamma).unsqueeze(-2) + beta.unsqueeze(-2)
        if self.bias_mode == 'feature':
            z = torch.cat([z, bias_pred], dim=-1)             # (B,C,J,N_FEATURES+2)
        h = self.joint_mlp(z)                                 # (B,C,J,d_joint)
        if gamma is not None and self.film_mode != 'input':
            h = h * (1.0 + gamma).unsqueeze(-2) + beta.unsqueeze(-2)

        if self.pool_mode == 'attn':
            w = torch.softmax(self.joint_attn(h).squeeze(-1), dim=-1)   # (B,C,J)
            return (w.unsqueeze(-1) * h).sum(dim=-2), w, bias_pred

        # ---- exp166: parameter-free pooling controls ----
        # self.joint_attn is still CONSTRUCTED for these modes (see __init__)
        # even though nothing reads it, so the RNG stream and the trainable
        # parameter count stay identical to pool_mode='attn'. The only thing
        # that differs between the three arms is this operation, which is what
        # makes them a clean paired comparison at a shared seed.
        if self.pool_mode == 'mean':
            w = h.new_full((B, C, J), 1.0 / J)
            return h.mean(dim=-2), w, bias_pred
        if self.pool_mode == 'max':
            out, idx = h.max(dim=-2)                             # (B,C,d_joint)
            # diagnostic w: fraction of the d_joint channels each joint won
            w = torch.zeros(B, C, J, device=h.device, dtype=h.dtype)
            w.scatter_add_(-1, idx, torch.ones_like(idx, dtype=h.dtype))
            return out, w / h.size(-1), bias_pred

        # query-conditioned: the candidate's camera query reads its own joints
        qj = self.joint_q_proj(q_src).reshape(B * C, 1, -1)   # (B*C,1,d_joint)
        kv = h.reshape(B * C, J, -1)                          # (B*C,J,d_joint)
        out, w = self.joint_mha(query=qj, key=kv, value=kv, need_weights=True)
        return out.reshape(B, C, -1), w.reshape(B, C, J), bias_pred

    def _joint_query_source(self, B, cur):
        """The query the joint cross-attention reads with. 'query' uses the
        full camera query (identity + geometry); 'query_geo' drops the learned
        per-camera term so the model stays strictly camera-agnostic."""
        if self.pool_mode in ('attn', 'mean', 'max'):
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

    def forward(self, x: torch.Tensor, cur: torch.Tensor, return_aux: bool = False,
                return_change: bool = False, return_motion: bool = False,
                use_future_motion: bool = False, return_bias: bool = False,
                external_state: tuple = None, return_tokens: bool = False):
        B, N, J, Feat = x.shape
        # exp176: same (B,N,d_model) context tokens either way -- the flat
        # encoder reads a camera as one 51-vector, SemGCN reads it as 17 nodes
        # on the skeleton graph. Nothing downstream of this line changes.
        tokens = None
        if self.branch_obs:
            tokens = self.semgcn(x.view(B * N, J, Feat)).view(B, N, -1) \
                if self.cam_enc == 'semgcn' \
                else self.cam_encoder(x.view(B * N, J * Feat)).view(B, N, -1)
        batch_idx = torch.arange(B, device=x.device)

        # camq_mode='full' is written out separately rather than folded into a
        # general accumulator so the shipped path keeps its exact float
        # addition ORDER (geo, then queries, then baseline) and stays
        # bit-identical to the pre-exp178 code -- verify_exp178_branch_parity.py
        # asserts that on real data.
        if self.camq_mode == 'full':
            q = self.geo_proj(self.all_deltas[cur])
            if self.learned_query:
                q = q + self.queries.expand(B, -1, -1)
            q = q + self.baseline_proj(self.baseline_matrix[cur].unsqueeze(-1))
        elif self.camq_mode == 'id':
            q = self.queries.expand(B, -1, -1)
        elif self.camq_mode == 'geo':
            q = self.geo_proj(self.all_deltas[cur]) \
                + self.baseline_proj(self.baseline_matrix[cur].unsqueeze(-1))
        else:   # 'none' -- the candidate-joint branch is the only source of q
            q = x.new_zeros(B, self.n_cands, self.d_model)

        motion_delta = None
        bias_pred = None
        if self.branch_cand:
            q, motion_delta, bias_pred = self._add_candidate_branch(
                x, cur, q, batch_idx, tokens, external_state, use_future_motion)

        kpm = None
        if self.branch_obs:
            if self.view_mask:
                # True = ignore this key. A dropped view is exactly 0.0 over all
                # 17x3 entries (stage131:118), so this is unambiguous. exp159's
                # caveat applies: at K=1 exactly one key survives, the softmax is
                # forced to 1.0 and attn_out stops depending on the query.
                kpm = (x.abs().sum(dim=(-1, -2)) == 0)
            for cross, ffn, n1, n2 in zip(self.cross_attn, self.ffn, self.norm1, self.norm2):
                attn_out, _ = cross(query=q, key=tokens, value=tokens,
                                    key_padding_mask=kpm)
                q = n1(q + attn_out)
                q = n2(q + ffn(q))
        else:
            # exp178 / exp160 convention: the attention is deleted, the residual
            # DEPTH is not, so the arm differs from config K in the fusion over
            # observed views and in nothing else. Only x[:, cur] can now reach
            # the output, which makes view dropout a literal no-op for this arm.
            for ffn, n1, n2 in zip(self.ffn, self.norm1, self.norm2):
                q = n1(q)
                q = n2(q + ffn(q))

        # RL and representation-learning callers need the three-branch
        # candidate representation before the supervised eight-score head.
        # Keeping this behind an opt-in flag preserves the historical forward
        # result (and checkpoint parity) for every existing caller.
        if return_tokens:
            return q

        logits = self.score_head(q).squeeze(-1)
        if return_bias:
            return logits, bias_pred
        if return_motion:
            return logits, motion_delta
        if return_change:
            change_logit = self.change_head(q.mean(dim=1)).squeeze(-1) if self.use_change_head else None
            return logits, change_logit
        if return_aux:
            delta = self.delta_head(q).squeeze(-1) if self.use_aux else None
            return logits, delta
        return logits

    def _add_candidate_branch(self, x, cur, q, batch_idx, tokens, external_state,
                              use_future_motion):
        """Q_c <- Q^cam_c + P(R_c): the candidate-joint reprojection branch.

        Split out of forward() verbatim by exp178 so that branch_cand=False can
        skip it -- including the frozen lifter call, which is the expensive part
        -- without duplicating a second copy of forward() that would drift out
        of sync (exp160's AttnAblatedRankNet subclass is exactly that failure:
        its copied forward still unpacks _pool_joint_tokens as a 2-tuple and
        would now raise). Returns (q, motion_delta, bias_pred).
        """
        B, N, J, Feat = x.shape
        # 2026-08-05: x may carry extra channels beyond (x,y,conf) -- e.g. the
        # latency-shift velocity experiment's vel_x/vel_y appended at 3:5. The
        # frozen lifter inside self.reproj was trained on exactly 3 channels,
        # so it is always sliced down here; a no-op when Feat==3 (every
        # existing config). cam_encoder above still sees the FULL x.
        cur_kp3 = x[batch_idx, cur][..., :3].contiguous()

        motion_delta = None
        if self.use_motion_head:
            # predicted per-joint (dx,dy,dz) mm, in the OBSERVED camera's own
            # frame, from that camera's encoded context token -- the same
            # token cross-attention reads as `tokens`. Not routed through
            # self.reproj unless use_future_motion=True (see class docstring).
            motion_delta = self.motion_head(tokens[batch_idx, cur]).view(B, J, 3)

        if external_state is not None:
            # 2026-08-09: caller-supplied (mean,var), each (B,J,3) camera-
            # frame mm -- e.g. a windowed/temporal lifter's output computed
            # by the CALLER (self.reproj.lifter, the frozen single-frame
            # lifter, is never invoked on this path; nothing else in this
            # method changes). Mutually exclusive with use_future_motion by
            # construction (external_state already IS the state the motion
            # head would otherwise be asked to predict) -- takes priority.
            feats = self.reproj(cur_kp3, cur, state=external_state)      # (B,C,J,12)
        elif self.use_motion_head and use_future_motion:
            with torch.no_grad():
                mean_cam, var_cam = self.reproj.lifter(cur_kp3)        # (B,J,3) each
            future_mean = mean_cam + motion_delta
            feats = self.reproj(cur_kp3, cur, state=(future_mean, var_cam))  # (B,C,J,12), PREDICTED t+25 geometry
        else:
            feats = self.reproj(cur_kp3, cur)                          # (B,C,J,12), raw, current-time geometry

        bias_pred = None
        if self.joint_tokens:
            # q is already (B, C, d_model) -- one query per candidate camera,
            # exactly the shape query_to_feat/_pool_joint_tokens needs.
            q_full = q if self.film_mode == 'query_add' else None
            cand, _, bias_pred = self._pool_joint_tokens(
                feats, cur, self._joint_query_source(B, cur), q_full)  # (B,C,d_joint)
            q = q + self.reprojvis_proj(cand)
        else:
            gain = ReprojJointFeatures.exp132_scalar(feats)           # (B,C)
            q = q + self.reprojvis_proj(gain.unsqueeze(-1))

        return q, motion_delta, bias_pred


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
