"""Purpose: Provide CJRR-Net training and evaluation primitives.

Main functionality: Load multiview data and compute ranking losses and per-position metrics.

Usage:
    python models/cjrrnet/training_core.py --help
"""

import argparse
import importlib.util
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BVPNET = Path(__file__).resolve().parent
DOMINANT = [4, 7]          # cam5, cam8 (SELECTED_CAM_IDS = [0,1,2,4,5,6,7,8])
# mm. Standard deviation of the TRAIN split's relative error
# (mpjpe - frame mean): 368.6 mm. Dividing by it makes the aux target unit-
# variance, so SmoothL1(beta=1) is quadratic over the bulk and Huber-robust on
# the tail (relative errors reach 7 m on a handful of catastrophic frames).
# A train-split statistic only -- val/test never enter it.
DELTA_SCALE = 368.6

CONFIGS = {   # name -> (joint_tokens, aux, film_mode, pool_mode)
    'A': (False, False, 'none', 'attn'),
    'B': (True,  False, 'none', 'attn'),
    'C': (False, True,  'none', 'attn'),
    'D': (True,  True,  'none', 'attn'),
    # PHASE 3 -- candidate-conditional fusion. The Phase-1 diagnosis said the
    # discriminative direction is candidate-camera specific while the fusion is
    # camera-agnostic; these add a per-candidate FiLM gate on the joint tokens.
    # -- route 1: bolt a per-camera parameter table onto the shared MLP --
    'E': (True,  False, 'identity', 'attn'),   # B + camera-identity FiLM
    'F': (True,  True,  'identity', 'attn'),   # D + camera-identity FiLM
    'G': (True,  False, 'geometry', 'attn'),   # control: FiLM from RELATIVE
                                               # geometry only (camera-agnostic)
    # -- route 2: reuse the architecture's own camera-query mechanism, i.e.
    #    let q[c] cross-attend over candidate c's 17 joint tokens. Conditioning
    #    is then a function SHARED across cameras, evaluated at each camera's
    #    query -- a regularised alternative to E's free per-camera parameters,
    #    which matters when cam1 has only 164 training frames.
    'H': (True,  False, 'none', 'query'),      # B + camera-query joint attention
    'I': (True,  True,  'none', 'query'),      # D + camera-query joint attention
    'J': (True,  False, 'none', 'query_geo'),  # control: geometry-only query
    # -- route 3: modulate the 12 INPUT channels, before the shared MLP.
    #    E/F/G and H/I/J all condition DOWNSTREAM of joint_mlp, by which point
    #    the camera-agnostic 12->32 mixing has already happened; none of them
    #    can restore a per-camera direction in the 12-channel space, which is
    #    what Phase 1 measured. MLP(gamma_c * z) can. 192 parameters.
    'K': (True,  False, 'input', 'attn'),      # B + input-channel FiLM
    'L': (True,  True,  'input', 'attn'),      # D + input-channel FiLM
    # -- exp166 (2026-07-27): is the learned JOINT pooling load-bearing? --
    #    stage135_joint_attention_analysis.py measured the trained weights at
    #    KL(w || uniform) = 0.00175 nats and 17.0/17 effective joints for
    #    config B -- i.e. the softmax is numerically indistinguishable from a
    #    mean. These two arms replace it with the operations it has apparently
    #    converged to (uniform mean) and with a genuinely different one (max),
    #    holding EVERYTHING else at config K. joint_attn is still built in both
    #    so the RNG stream and the 325,090 parameter count are unchanged and
    #    the three arms are paired at a shared seed.
    'K_mean': (True, False, 'input', 'mean'),  # K, uniform mean over 17 joints
    'K_max':  (True, False, 'input', 'max'),   # K, max over 17 joints
}

# exp168 (2026-07-27): which of the 12 per-joint channels a config blanks out.
# reproj_joint_features.py:251 broadcasts cur_conf identically across all 8
# candidates -- channel 11 holds the SAME 17-vector for every candidate of a
# given frame, so on its own it cannot rank candidates at all. It also already
# reaches the model a second way, through cam_encoder(x), which sees the
# current view's confidences as x[..., 2]. Both facts are structural, not
# hypotheses. What is NOT settled is whether the channel is therefore dead
# weight: config K's input-FiLM multiplies it by a per-candidate gamma_c, and
# the joint MLP is nonlinear, so it can gate the candidate-VARYING channels --
# which is exactly what exp132's hand-designed scalar did with it,
# mean_j[(1 - cur_conf_j) * p_in_frame_j * (1 - p_self_occ_j)]. Only an
# ablation decides it.
ZERO_CHANNELS = {
    'K_noconf': [11],           # drop cur_conf
    'K_noprob': [5, 6],         # control: drop p_in_frame + p_self_occ, the two
                                # channels exp132's scalar multiplies cur_conf
                                # BY. If noconf is null and noprob is not, the
                                # interaction story survives with cur_conf as
                                # the removable half.
}
for _k, _ch in ZERO_CHANNELS.items():
    CONFIGS[_k] = (True, False, 'input', 'attn')

# exp169 (2026-07-27): the 142K-lite deletion order. Each arm is ONE deletion
# from config K except K_lite, which is the composition.
#   name -> dict of extra JointTokenReprojVisRankNet kwargs
EXTRA_KWARGS = {
    'K_geoquery': dict(learned_query=False),
    'K_lite':     dict(learned_query=False, view_mask=True),
}
CONFIGS['K_geoquery'] = (True, False, 'input', 'attn')
# exp170: the two deletions that survived BOTH the norm% test and the collapse
# constraint, composed. Tested as its own arm rather than assumed, because
# exp169 measured the deletions as non-additive: K_lite's -0.98 pp is worse
# than the sum of its three parts (~-0.5 pp), so "each is free" does not imply
# "both are free".
# exp171: the two deletions that survived the gated read, composed, with
# everything else left alone. A strict subset of K_lite -- same mean pooling
# and same geometry-only query, but keeping BOTH cross-attention layers and no
# view mask, i.e. dropping exactly the part of K_lite whose collapse cost was
# significant on its own (K_1layer: cam5+8 +2.14 pp, p=0.020 gated).
# exp172: is input FiLM's multiplicative half load-bearing, or would a plain
# concatenated camera identity do? Both are candidate-conditioning on the same
# 12 channels; only FiLM can change WHICH direction the shared MLP reads.
CONFIGS['K_concat'] = (True, False, 'concat', 'attn')   # one-hot concat, no FiLM
CONFIGS['K_nofilm'] = (True, False, 'none',   'attn')   # = config B, for reference

# exp173: K_meanq with the input FiLM also swapped for the one-hot concat.
# All three deletions/replacements that individually passed their own test,
# stacked. Run because the two halves push cam5+8 in OPPOSITE directions
# (K_meanq -1.61 pp, K_concat +1.64 pp at gate 0.3/10), so the composed
# collapse behaviour cannot be predicted from either.
CONFIGS['K_meanqc'] = (True, False, 'concat', 'mean')
EXTRA_KWARGS['K_meanqc'] = dict(learned_query=False)

# exp174 (2026-07-28): additive query injection (exp151's mechanism) at
# config K's own FiLM insertion point, vs the existing multiplicative
# per-camera lookup table. Same insertion point, opposite mechanism --
# isolates "does the REAL query beat a learned per-camera gamma/beta table",
# holding architecture/pooling/objective identical to config K otherwise.
CONFIGS['K_queryadd'] = (True, False, 'query_add', 'attn')

# exp176 (2026-07-28): the context encoder itself. Everything above ablates
# things INSIDE the joint-token / query block; cam_encoder is the one module the
# project has never touched -- the same flat 51->128->128 MLP as exp43, which
# hands the 17 joints to the fusion stack as one flattened vector. These arms
# replace it with SemGCN (Zhao et al. CVPR 2019): message passing on the COCO
# skeleton with LEARNED per-edge weights. Nothing else changes -- joint tokens,
# input FiLM, attention pooling, both cross-attention layers, the objective and
# the K~U{1..8} dropout protocol are config K's.
#
# Prior expectation is a null, and it is on record: exp159 measured 72-78% of
# the cross-attention mass landing on the content-free sentinel tokens these
# very encoders produce for dropped views, and exp160 found deleting the whole
# cross-attention stack costs only -0.55 pp (p=0.11). A better context encoder
# feeding a near-deletable pathway should not move the metric. The arms are run
# because "should not" is not a measurement, and because the two published
# nulls for graph structure here (exp26 GCN 70.3 vs flat 70.0, exp28) were on
# the retracted i=25 data generation, on a single-view quality head, with a
# FIXED adjacency -- which is exactly what SemGCN's contribution replaces.
#
#   K_semgcn    param-matched (hid 32, 2 residual blocks, 8-d readout):
#               27,506 encoder params vs the flat MLP's 23,680, so a null
#               cannot be dismissed as "the graph encoder was starved".
#   K_semgcnbn  same, with the original's BatchNorm instead of LayerNorm.
#               Not the primary arm: ~half the slots in every batch are
#               structural zeros (dropped views), which enter BN's statistics.
#               Run so that deviation is measured rather than assumed.
#   K_semgcnw   hid 128 / 4 blocks / 16-d readout = SemGCN at its own published
#               scale, 306,230 encoder params. Nearly doubles the model (325k ->
#               608k), so it is the capacity-generous control, not a fair swap.
for _name, _kw in {
    'K_semgcn':   dict(cam_enc='semgcn'),
    'K_semgcnbn': dict(cam_enc='semgcn', semgcn_norm='bn'),
    'K_semgcnw':  dict(cam_enc='semgcn', semgcn_hid=128, semgcn_out=16,
                       semgcn_blocks=4),
}.items():
    CONFIGS[_name] = (True, False, 'input', 'attn')   # = config K in every other respect
    EXTRA_KWARGS[_name] = _kw

EXTRA_KWARGS['K_meanq'] = dict(learned_query=False)
CONFIGS['K_meanq'] = (True, False, 'input', 'mean')

ZERO_CHANNELS['K_slim'] = [11]
EXTRA_KWARGS['K_slim'] = dict(learned_query=False)
CONFIGS['K_slim'] = (True, False, 'input', 'attn')
CONFIGS['K_1layer']   = (True, False, 'input', 'attn')   # depth set via --n_layers
CONFIGS['K_lite']     = (True, False, 'input', 'mean')

# ---- exp178 (2026-08-12): WHOLE-BRANCH ablation, the paper's first-layer table.
#
# Every arm above this line ablates something INSIDE one of the three branches
# that build the candidate query Q_c. These delete a branch outright:
#
#     Q_c = Q^cam_c            (camera-query branch: learned per-camera table
#                               + relative az/el + baseline)
#         + P(R_c)             (candidate-joint branch: the frozen lifter's 3D
#                               state reprojected into candidate c, 17x12 field)
#     Q_c <- cross-attn(Q_c, tokens)   (observed-view branch)
#
# The switches live in the model's ctor (branch_obs / branch_cand / camq_mode),
# not here, so a deleted branch's parameters are genuinely deleted -- the paper
# reports trainable params per row, and exp169's older convention of leaving
# self.queries built-but-unread would have inflated the camera-query rows by
# 1,024 dead parameters. Construction order is unchanged, so every surviving
# module is initialised bit-for-bit as config K would initialise it at the same
# seed (verify_exp178_branch_parity.py asserts config K itself is numerically
# untouched by the refactor, to 1e-6 on the real test split).
#
# The tuples stay at config K's (True, False, 'input', 'attn') even for
# K_nocand, where joint_tokens/FiLM describe modules that are then deleted --
# holding the tuple fixed is what keeps "the arm differs from K in ONE branch"
# literally true of the construction path.
#
#   K_noobs    -- w/o multi-view observation fusion. NOT "w/o visual input":
#                 the candidate-joint branch still reads the reference view's
#                 own keypoints, so the arm answers "do the OTHER views'
#                 context tokens add anything beyond the reference view".
#                 View dropout is a literal no-op for this arm.
#   K_nocamq   -- w/o camera-query branch. Q_c = P(R_c) alone. Reads as "given
#                 that F_{c,j} is already candidate-specific by construction,
#                 is an EXPLICIT camera identity/relative pose still needed".
#                 Caveat for the caption: film_mode='input' is a per-candidate
#                 gamma/beta lookup, so this arm is not camera-identity-free.
#   K_nocand   -- w/o candidate-joint branch. The one reviewers will ask about:
#                 no Gaussian 3D lifting, no reprojection into the candidate.
#                 Expect norm% to move less than 0%-rec and cam5+8 do; if so
#                 the branch's job is de-collapsing, not mean accuracy.
#   K_prior    -- prior-only sanity check. Q_c is the learned per-camera table
#                 and nothing else, so the logits are CONSTANT in the input and
#                 in `cur`: the model can only emit one fixed camera. Bounds
#                 how much of the headline number is "cam5/cam8 are usually
#                 good". Should land at Best-Fixed-Camera; if it beats it,
#                 something leaks.
#   K_camid    -- camera-query branch reduced to the learned identity table.
#   K_camgeo   -- camera-query branch reduced to relative geometry (az/el delta
#                 + baseline), no per-camera table. The pair (K_camid,K_camgeo)
#                 answers "identity prior or generalisable relative geometry",
#                 which is the one that matters for a different camera rig.
#                 Appendix rows unless the paper claims rig transfer.
for _name, _kw in {
    'K_noobs':  dict(branch_obs=False),
    'K_nocamq': dict(camq_mode='none'),
    'K_nocand': dict(branch_cand=False),
    'K_prior':  dict(branch_obs=False, branch_cand=False, camq_mode='id'),
    'K_camid':  dict(camq_mode='id'),
    'K_camgeo': dict(camq_mode='geo'),
}.items():
    CONFIGS[_name] = (True, False, 'input', 'attn')   # = config K otherwise
    EXTRA_KWARGS[_name] = _kw


def load_exp142():
    spec = importlib.util.spec_from_file_location(
        'exp142', BVPNET / 'cjrrnet.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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


def apply_view_dropout_with_ref(kp: torch.Tensor):
    """UNCHANGED K-view dropout protocol (identical to exp132/exp136)."""
    B, out = kp.size(0), kp.clone()
    n_cams = kp.size(1)
    ref = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        K    = random.randint(1, n_cams)
        keep = sorted(random.sample(range(n_cams), K))
        mask = torch.zeros(n_cams, dtype=torch.bool); mask[keep] = True
        out[b, ~mask] = 0.0
        ref[b] = keep[0]
    return out, ref


def lambda_rank_loss(logits, norm_mpjpe):
    d_score = logits.unsqueeze(1) - logits.unsqueeze(2)
    d_mpjpe = norm_mpjpe.unsqueeze(1) - norm_mpjpe.unsqueeze(2)
    weight  = F.relu(d_mpjpe)
    loss    = (weight * F.softplus(d_score)).sum(dim=(1, 2)).mean()
    N       = logits.size(1)
    return loss / (N * (N - 1) / 2)


def build_pair_weights(mpjpe, gap_thresh, w_dom, w_runnerup, rare_boost):
    """(B,C,C) weights W[b,i,j] for 'candidate i is better than candidate j'.

    Covers all five pair families the study asks for, in one vectorised op:
      - large-gap pairs            : relative MPJPE gap > gap_thresh
      - oracle vs cam5 / cam8      : + w_dom
      - runner-up vs cam5 / cam8   : + w_runnerup
      - rare-oracle vs dominant    : the oracle-vs-dominant term is multiplied
                                     by rare_boost on rare-oracle frames
    W is zero wherever i is NOT strictly better than j, so the loss can never
    push the ranking in the wrong direction.
    """
    B, C = mpjpe.shape
    device = mpjpe.device
    frame_mean = mpjpe.mean(dim=1, keepdim=True) + 1e-8
    rel = mpjpe / frame_mean                                   # (B,C)
    better = (rel.unsqueeze(2) < rel.unsqueeze(1)).float()     # [b,i,j] i better than j
    gap = (rel.unsqueeze(1) - rel.unsqueeze(2)).abs()          # (B,C,C)

    W = better * (gap > gap_thresh).float()                    # large-gap family

    order = torch.argsort(mpjpe, dim=1)
    oracle = order[:, 0]
    runner = order[:, 1]
    is_rare = (~torch.isin(oracle, torch.tensor(DOMINANT, device=device))).float()  # (B,)

    dom_col = torch.zeros(B, C, device=device)
    dom_col[:, DOMINANT] = 1.0                                 # j is a dominant camera

    boost = 1.0 + (rare_boost - 1.0) * is_rare                 # (B,)
    oracle_row = F.one_hot(oracle, C).float()                  # (B,C) over i
    runner_row = F.one_hot(runner, C).float()

    W = W + w_dom * boost.view(B, 1, 1) * oracle_row.unsqueeze(2) * dom_col.unsqueeze(1)
    W = W + w_runnerup * runner_row.unsqueeze(2) * dom_col.unsqueeze(1)
    return W * better                                          # re-mask: never invert a true order


def pairwise_loss(logits, W):
    d_score = logits.unsqueeze(2) - logits.unsqueeze(1)        # [b,i,j] = s_i - s_j
    l = W * F.softplus(-d_score)
    return l.sum() / (W.sum() + 1e-8)


def compute_losses(logits, delta_hat, mpjpe, args):
    norm_mpjpe = mpjpe / (mpjpe.mean(dim=1, keepdim=True) + 1e-8)
    oracle_tgt = torch.argmin(mpjpe, dim=-1)
    loss_ce = F.cross_entropy(logits, oracle_tgt)
    loss_lr = lambda_rank_loss(logits, norm_mpjpe)
    total = loss_ce + args.lambda_lr * loss_lr

    zero = torch.zeros((), device=logits.device)
    loss_err, loss_pair = zero, zero
    if args.lambda_err > 0 and delta_hat is not None:
        delta_true = mpjpe - mpjpe.mean(dim=1, keepdim=True)
        loss_err = F.smooth_l1_loss(delta_hat, delta_true / DELTA_SCALE, beta=1.0)
        total = total + args.lambda_err * loss_err
    if args.lambda_pair > 0:
        W = build_pair_weights(mpjpe, args.gap_thresh, args.w_dom,
                               args.w_runnerup, args.rare_boost)
        loss_pair = pairwise_loss(logits, W)
        total = total + args.lambda_pair * loss_pair
    return total, loss_ce.detach(), loss_lr.detach(), loss_err.detach(), loss_pair.detach()


def compute_pick_dist(pred_col, oracle_col, N):
    pred_pct = np.bincount(pred_col, minlength=8) / N * 100
    recall = np.zeros(8)
    for c in range(8):
        mask = oracle_col == c
        recall[c] = float(np.mean(pred_col[mask] == c)) * 100 if mask.sum() > 0 else float('nan')
    return pred_pct, recall


def evaluate_per_position(model, data_dir, batch_size, device, split_name):
    """Same masked per-position protocol every experiment in this line uses."""
    ds = LeanMultiViewDataset(data_dir)
    mpjpe_vals = ds.mpjpe_per_cam
    N, n_cands = mpjpe_vals.shape
    random_px = float(mpjpe_vals.mean())
    oracle_col_all = np.argmin(mpjpe_vals, axis=1)
    oracle_px = float(mpjpe_vals[np.arange(N), oracle_col_all].mean())
    rare_mask = ~np.isin(oracle_col_all, DOMINANT)
    dom_mask = ~rare_mask
    model.eval()

    per_pos, all_pred_cols = [], []
    for pos in range(n_cands):
        kp = np.zeros_like(ds.kp_mv); kp[:, pos] = ds.kp_mv[:, pos]
        cur = torch.full((N,), pos, dtype=torch.long, device=device)
        preds = []
        with torch.no_grad():
            for i in range(0, N, batch_size):
                logits = model(torch.tensor(kp[i:i+batch_size], device=device), cur[i:i+batch_size])
                preds.append(logits.cpu().numpy())
        pred_col = np.argmax(np.concatenate(preds), axis=1)
        all_pred_cols.append(pred_col)
        pred_px = mpjpe_vals[np.arange(N), pred_col]
        per_pos.append(dict(
            position=pos,
            norm_pct=(random_px - float(pred_px.mean())) / (random_px - oracle_px + 1e-8) * 100,
            top1=float(np.mean(pred_col == oracle_col_all))))

    avg_norm = float(np.mean([p['norm_pct'] for p in per_pos]))
    avg_top1 = float(np.mean([p['top1'] for p in per_pos]))

    dists = [compute_pick_dist(pc, oracle_col_all, N)[0] for pc in all_pred_cols]
    pred_pct_mean = np.mean(dists, axis=0)
    cam58 = pred_pct_mean[4] + pred_pct_mean[7]
    recalls = [compute_pick_dist(pc, oracle_col_all, N)[1] for pc in all_pred_cols]
    recall_mean = np.nanmean(recalls, axis=0)
    n_zero_recall = int(np.sum(np.nan_to_num(recall_mean, nan=100.0) < 0.05))

    tol5, tol10, rare_r, dom_r, revert = [], [], [], [], []
    for pc in all_pred_cols:
        gap = mpjpe_vals[np.arange(N), pc] - mpjpe_vals[np.arange(N), oracle_col_all]
        tol5.append(float(np.mean(gap <= 5.0)) * 100)
        tol10.append(float(np.mean(gap <= 10.0)) * 100)
        rare_r.append(float(np.mean(pc[rare_mask] == oracle_col_all[rare_mask])) * 100)
        dom_r.append(float(np.mean(pc[dom_mask] == oracle_col_all[dom_mask])) * 100)
        revert.append(float(np.mean(np.isin(pc[rare_mask], DOMINANT))) * 100)

    out = dict(per_position=per_pos, avg_norm_pct=avg_norm, avg_top1=avg_top1,
               tol5_top1=float(np.mean(tol5)), tol10_top1=float(np.mean(tol10)),
               cam58_share=float(cam58), n_zero_recall=n_zero_recall,
               pred_pct_mean=pred_pct_mean.tolist(), recall_mean=recall_mean.tolist(),
               rare_recall=float(np.mean(rare_r)), dom_recall=float(np.mean(dom_r)),
               revert_to_58=float(np.mean(revert)),
               cam1_recall=float(recall_mean[1]),
               random_px=random_px, oracle_px=oracle_px)
    print(f'\n[{split_name}] norm%={avg_norm:.2f}  top1={avg_top1*100:.1f}  '
          f'tol5={out["tol5_top1"]:.1f}  tol10={out["tol10_top1"]:.1f}  '
          f'cam5+8={cam58:.1f}  0%-recall={n_zero_recall}')
    print(f'  rare_recall={out["rare_recall"]:.2f}  cam1_recall={out["cam1_recall"]:.2f}  '
          f'dom_recall={out["dom_recall"]:.2f}  revert_to_58={out["revert_to_58"]:.2f}')
    print('  per-cam recall [cam0 cam1 cam2 cam4 cam5 cam6 cam7 cam8]: '
          + ' '.join(f'{r:.1f}' for r in recall_mean))
    return out


def train(args):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    joint_tokens, aux, film_mode, pool_mode = CONFIGS[args.config]
    if not aux:
        args.lambda_err, args.lambda_pair = 0.0, 0.0

    model = load_exp142().JointTokenReprojVisRankNet(
        n_cands=8, n_joints=17, n_features=3,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout_prob, lift3d_ckpt=args.lift3d_ckpt,
        var_floor=args.var_floor, joint_tokens=joint_tokens, aux_head=aux,
        film_mode=film_mode, pool_mode=pool_mode, d_joint=args.d_joint,
        zero_channels=ZERO_CHANNELS.get(args.config),
        **EXTRA_KWARGS.get(args.config, {})).to(device)

    train_ds = LeanMultiViewDataset(args.train_data)
    val_ds   = LeanMultiViewDataset(args.val_data)
    pin      = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val, patience_cnt, history = float('inf'), 0, []
    t0 = time.time()
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'config={args.config} (joint_tokens={joint_tokens}, aux={aux}, '
          f'film={film_mode}, pool={pool_mode})  seed={args.seed}')
    print(f'params={total_p:,} trainable={trainable_p:,}  '
          f'lambda_err={args.lambda_err} lambda_pair={args.lambda_pair} '
          f'rare_boost={args.rare_boost}\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for kp, scores in train_dl:
            kp_aug, ref = apply_view_dropout_with_ref(kp)
            kp_aug, scores, ref = kp_aug.to(device), scores.to(device), ref.to(device)
            optimizer.zero_grad()
            logits, delta_hat = model(kp_aug, ref, return_aux=True)
            loss, *_ = compute_losses(logits, delta_hat, scores, args)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item() * kp.size(0); total_n += kp.size(0)
        tr_loss = total_loss / total_n

        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for kp, scores in val_dl:
                kp_aug, ref = apply_view_dropout_with_ref(kp)
                kp_aug, scores, ref = kp_aug.to(device), scores.to(device), ref.to(device)
                logits, delta_hat = model(kp_aug, ref, return_aux=True)
                loss, *_ = compute_losses(logits, delta_hat, scores, args)
                val_loss_sum += loss.item() * kp.size(0); val_n += kp.size(0)
        val_loss = val_loss_sum / val_n
        scheduler.step(val_loss)
        history.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss))
        if epoch % 10 == 0 or epoch == 1:
            print(f'[{epoch:>3}/{args.epochs}] train={tr_loss:.4f}  val={val_loss:.4f}')

        ckpt = dict(epoch=epoch, model_state_dict=model.state_dict(),
                    val_loss=val_loss, args=vars(args))
        torch.save(ckpt, save_dir / 'last.pt')
        if val_loss < best_val:
            best_val, patience_cnt = val_loss, 0
            torch.save(ckpt, save_dir / 'best.pt')
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f'Early stop at epoch {epoch}'); break

    print(f'\nTraining time: {(time.time()-t0)/60:.1f} min')
    json.dump(history, open(save_dir / 'train_history.json', 'w'), indent=2)
    ckpt = torch.load(save_dir / 'best.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'Loaded best checkpoint (epoch={ckpt["epoch"]})')

    print('\n' + '=' * 70)
    print('Refs: exp132 norm%=77.48 rare_recall=~9.7 cam1=0.0 | exp136 74.70/13.3/0.7 '
          '| exp139 a1.0 ~72/17.4/2.1')
    val_m  = evaluate_per_position(model, args.val_data,  args.batch_size, device, 'val')
    test_m = evaluate_per_position(model, args.test_data, args.batch_size, device, 'test(S8)')
    meta = dict(tier=f'exp142_{args.config}', config=args.config,
                joint_tokens=joint_tokens, aux_head=aux, film_mode=film_mode,
                pool_mode=pool_mode,
                total_params=total_p, trainable_params=trainable_p,
                training_time_s=time.time() - t0, seed=args.seed,
                lambda_lr=args.lambda_lr, lambda_err=args.lambda_err,
                lambda_pair=args.lambda_pair, rare_boost=args.rare_boost,
                w_dom=args.w_dom, w_runnerup=args.w_runnerup, gap_thresh=args.gap_thresh,
                best_epoch=int(ckpt['epoch']), best_val_loss=float(ckpt['val_loss']))
    val_m.update(meta); test_m.update(meta)
    json.dump(val_m,  open(save_dir / 'val_metrics.json',  'w'), indent=2)
    json.dump(test_m, open(save_dir / 'test_metrics.json', 'w'), indent=2)
    print(f'\nSaved -> {save_dir}/test_metrics.json')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',       choices=list(CONFIGS), required=True)
    p.add_argument('--train_data',   default=str(BVPNET.parents[1] / 'data' / 'multiview_split_train'))
    p.add_argument('--val_data',     default=str(BVPNET.parents[1] / 'data' / 'multiview_split_val'))
    p.add_argument('--test_data',    default=str(BVPNET.parents[1] / 'data' / 'multiview_split_test'))
    p.add_argument('--save_dir',     required=True)
    p.add_argument('--lift3d_ckpt',  default=str(BVPNET.parents[1] / 'weights' / 'lifter_bypassed' / 'lift3d_gaussian_mlp_best.pt'))
    p.add_argument('--var_floor',    type=float, default=1.0)
    # loss
    p.add_argument('--lambda_lr',    type=float, default=0.1)   # LambdaRank, KEPT
    p.add_argument('--lambda_err',   type=float, default=0.5)   # relative-MPJPE aux
    p.add_argument('--lambda_pair',  type=float, default=0.5)   # pairwise candidate
    p.add_argument('--rare_boost',   type=float, default=2.0)
    p.add_argument('--w_dom',        type=float, default=2.0)
    p.add_argument('--w_runnerup',   type=float, default=1.0)
    p.add_argument('--gap_thresh',   type=float, default=0.15)
    # unchanged from exp132
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=256)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--dropout_prob', type=float, default=0.1)
    p.add_argument('--patience',     type=int,   default=15)
    p.add_argument('--d_model',      type=int,   default=128)
    p.add_argument('--n_heads',      type=int,   default=4)
    p.add_argument('--n_layers',     type=int,   default=2)
    p.add_argument('--d_joint',      type=int,   default=32)
    p.add_argument('--seed',         type=int,   default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
