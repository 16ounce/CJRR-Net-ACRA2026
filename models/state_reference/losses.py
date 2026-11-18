"""Purpose: Compute reference ranker losses.

Main functionality: Combine cross-entropy, ranking, regret, and pairwise terms.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# cam5, cam8 -- the pair term keys off which cameras are dominant
from .evaluate import DOMINANT


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


REGRET_SCALE = 174.818


def fit_regret_scale(data_dir) -> float:
    """(random_px - oracle_px) of a split -- the constant norm% is measured against.

    MUST be refit whenever the training distribution changes. The module-level
    REGRET_SCALE is the headline split's value; every LOSO fold trains on a
    different set of subjects and has its own, ranging widely (per-subject
    random->oracle spread runs 82.4 px to 284.3 px, per pbs/170's header).
    Reusing 174.818 on a fold whose true scale is, say, 100 px would silently
    shrink the regret term's effective weight by 1.75x and break the
    exact-surrogate property that motivates the term at all.

    Train-split statistic only -- val and test never enter it.
    """
    m = np.load(Path(data_dir) / 'mpjpe_per_cam.npy').astype(np.float64)
    return float(m.mean() - m.min(axis=1).mean())


def normalized_regret_loss(logits, mpjpe, scale=REGRET_SCALE):
    """Expected per-frame MPJPE regret under the softmax policy, normalised by
    the split-level (random_px - oracle_px) that norm% is measured against.

    On the train split -- whose (random_px - oracle_px) is REGRET_SCALE -- this
    equals (1 - norm%/100) exactly for any deterministic policy, verified to
    ~3e-7: 0.000000 for the oracle policy, 0.999999 for the uniform one. So
    minimising it minimises the reported metric directly rather than a
    rank-based proxy for it. Not clipped -- see header.
    """
    p = F.softmax(logits, dim=-1)                                # (B,C) policy
    regret = mpjpe - mpjpe.min(dim=1, keepdim=True).values       # (B,C) px, >= 0
    return (p * regret).sum(dim=1).mean() / scale


def compute_losses(logits, mpjpe, args):
    """L = 1.0*CE + lambda_lr*LambdaRank + lambda_regret*regret + lambda_pair*pair.

    CE, LambdaRank and the camera-pair term are stage131's, called directly so
    they cannot drift; only normalized_regret_loss is new.
    """
    oracle_tgt = torch.argmin(mpjpe, dim=-1)
    norm_mpjpe = mpjpe / (mpjpe.mean(dim=1, keepdim=True) + 1e-8)

    loss_ce = F.cross_entropy(logits, oracle_tgt)
    loss_lr = lambda_rank_loss(logits, norm_mpjpe)
    total = loss_ce + args.lambda_lr * loss_lr

    zero = torch.zeros((), device=logits.device)
    loss_regret, loss_pair = zero, zero
    if args.lambda_regret > 0:
        loss_regret = normalized_regret_loss(
            logits, mpjpe, getattr(args, 'regret_scale_used', REGRET_SCALE))
        total = total + args.lambda_regret * loss_regret
    if args.lambda_pair > 0:
        # stage131's camera-pair ranking: large-gap pairs + oracle/runner-up
        # vs the two dominant cameras. Operates on logits -- no aux head needed.
        W = build_pair_weights(mpjpe, args.gap_thresh, args.w_dom,
                                    args.w_runnerup, args.rare_boost)
        loss_pair = pairwise_loss(logits, W)
        total = total + args.lambda_pair * loss_pair
    return total, loss_ce.detach(), loss_lr.detach(), loss_regret.detach(), loss_pair.detach()
