#!/usr/bin/env python3
"""Purpose: Evaluate the training-free geometric camera baseline.

Main functionality: Rank candidates from cached joint reprojection features and report selection metrics.

Usage:
    python baselines/geometric_utility/evaluate.py --help
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

BVPNET = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(BVPNET / 'utils' / 'cjrrnet'))
DOCS = BVPNET / 'docs'
CAM_IDS = [0, 1, 2, 4, 5, 6, 7, 8]
DOMINANT = [4, 7]                      # cam5, cam8 in SELECTED_CAM_IDS order
N_CAM = 8

# reproj_joint_features.FEATURE_NAMES indices
F_P_IN, F_P_OCC, F_CUR_CONF = 5, 6, 11

GATE_CONF, GATE_MINVIS = 0.3, 10

# Published control rows (test split, S8). Sources named per row so a stale
# number can be traced instead of guessed at.
CONTROLS = {
    # docs/exp158_regret_loss.json -> cells.rand_full
    'exp158 ungated':   dict(mean=78.2301, sd=1.2447, n=15,
                             src='docs/exp158_regret_loss.json cells.rand_full'),
    # docs/trigger_sweep_exp158_gatecross.txt -> "always fire, gate_0.3_10"
    'exp158 gate 0.3/10': dict(mean=83.62, sd=0.64, n=15,
                               src='docs/trigger_sweep_exp158_gatecross.txt'),
}
# docs/gated_baselines.txt, shipped gate row
FIXED_CAM5_UNGATED = 75.28
GATE_RANDOM = 60.25
GATE_ORACLE = 99.69


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_split(name):
    d = BVPNET / 'data' / f'multiview_split_{name}'
    feats = np.load(d / 'reproj_joint_feats.npy')      # (N, 8cur, 8cand, 17, 12)
    mpjpe = np.load(d / 'mpjpe_per_cam.npy').astype(np.float64)   # (N, 8)
    kp = np.load(d / 'keypoints_mv.npy').astype(np.float32)       # (N, 8, 17, 3)
    assert feats.ndim == 5 and feats.shape[1] == feats.shape[2] == N_CAM, feats.shape
    assert feats.shape[0] == mpjpe.shape[0] == kp.shape[0]
    return feats, mpjpe, kp


def check_field_matches_current_view(feats, kp):
    """The cur_conf channel of the field for starting position p must be the
    confidence of view p, broadcast over candidates. If this fails, the cached
    dump is not aligned with the keypoints and every number below is void."""
    worst = 0.0
    for pos in range(N_CAM):
        got = feats[:, pos, :, :, F_CUR_CONF]                  # (N, 8cand, 17)
        want = kp[:, pos, :, 2][:, None, :]                    # (N, 1, 17)
        worst = max(worst, float(np.abs(got - want).max()))
    assert worst < 1e-5, f'cached field is misaligned with keypoints (max {worst:.2e})'
    return worst


def check_matches_shipped_scalar(feats, U_need_vis):
    """The whole claim of this script is that its utility IS the scalar the
    shipped model computes. Assert it against the model's own code path rather
    than asserting it in a comment. float32 vs float64 -> ~1e-7.

    torch is needed ONLY for this check -- the baseline itself is numpy. If
    torch is missing the check is skipped loudly (returns None) rather than
    taking the whole script down, but a run without it is an UNVERIFIED run."""
    try:
        import torch
        from reproj_joint_features import ReprojJointFeatures
    except ImportError:
        return None
    ship = ReprojJointFeatures.exp132_scalar(torch.from_numpy(feats)).numpy()
    err = float(np.abs(ship - U_need_vis).max())
    agree = float((ship.argmax(-1) == U_need_vis.argmax(-1)).mean())
    assert err < 1e-5 and agree == 1.0, f'diverged from exp132_scalar ({err}, {agree})'
    return err


def fixed_camera_reference(mpjpe):
    """norm% of always-cam_c, for all 8 c. Two jobs: (a) the fixed-camera bar
    this baseline has to clear, (b) a CALIBRATION CHECK -- always-cam5 must
    come out at 75.28, the published value in docs/gated_baselines.txt, or
    score() does not match the harness every other row was measured with."""
    rows = np.arange(mpjpe.shape[0])
    random_px = float(mpjpe.mean())
    oracle_px = float(mpjpe[rows, np.argmin(mpjpe, axis=1)].mean())
    return {f'cam{CAM_IDS[c]}': float((random_px - mpjpe[:, c].mean())
                                      / (random_px - oracle_px) * 100)
            for c in range(N_CAM)}


# --------------------------------------------------------------------------
# utilities  -- every one is (N, 8cur, 8cand)
# --------------------------------------------------------------------------
def build_utilities(feats):
    p_in = feats[..., F_P_IN].astype(np.float64)      # (N, cur, cand, J)
    p_occ = feats[..., F_P_OCC].astype(np.float64)
    need = 1.0 - feats[..., F_CUR_CONF].astype(np.float64)
    vis = p_in * (1.0 - p_occ)

    U = {
        'need_vis':      (need * vis).mean(axis=-1),
        'vis_only':      vis.mean(axis=-1),
        'need_inframe':  (need * p_in).mean(axis=-1),
        'need_notocc':   (need * (1.0 - p_occ)).mean(axis=-1),
        'need_only':     np.broadcast_to(need.mean(axis=-1), need.shape[:-1]).copy(),
    }
    U['need_vis_min'] = -U['need_vis']
    return U


def argmax_policy(U, exclude_current=False, passing=None):
    """U: (N, cur, cand) -> pred (N, cur) int.

    passing: (N, cand) bool gate mask. Candidates that fail are removed from
    contention; frames where nothing passes revert to the ungated argmax,
    exactly as the model harness does (analyze_gated_baselines.py).
    Ties break to the lowest index (np.argmax), same as torch.argmax.
    """
    S = U.copy()
    if exclude_current:
        idx = np.arange(N_CAM)
        S[:, idx, idx] = -np.inf
    if passing is not None:
        any_pass = passing.any(axis=1)                          # (N,)
        masked = np.where(passing[:, None, :], S, -np.inf)
        S = np.where(any_pass[:, None, None], masked, S)
    return np.argmax(S, axis=-1)


def gate_mask(kp, thr=GATE_CONF, min_vis=GATE_MINVIS):
    """(N, 8) bool -- candidate c passes if >= min_vis of its own joints have
    conf >= thr. Identical rule to nbv_pipeline/step3_quality.py."""
    return (kp[..., 2] >= thr).sum(axis=2) >= min_vis


# --------------------------------------------------------------------------
# metrics  -- mirrors stage131.evaluate_per_position
# --------------------------------------------------------------------------
def score(pred, mpjpe):
    """pred: (N, 8cur) picked candidate index. Returns the harness dict."""
    N = mpjpe.shape[0]
    rows = np.arange(N)
    random_px = float(mpjpe.mean())
    oracle_col = np.argmin(mpjpe, axis=1)
    oracle_px = float(mpjpe[rows, oracle_col].mean())
    rare_mask = ~np.isin(oracle_col, DOMINANT)
    dom_mask = ~rare_mask

    norms, top1s, tol5, tol10, rare_r, dom_r, dists, recalls = ([] for _ in range(8))
    for pos in range(N_CAM):
        pc = pred[:, pos]
        px = mpjpe[rows, pc]
        norms.append((random_px - float(px.mean())) / (random_px - oracle_px + 1e-8) * 100)
        top1s.append(float(np.mean(pc == oracle_col)) * 100)
        gap = px - mpjpe[rows, oracle_col]
        tol5.append(float(np.mean(gap <= 5.0)) * 100)
        tol10.append(float(np.mean(gap <= 10.0)) * 100)
        rare_r.append(float(np.mean(pc[rare_mask] == oracle_col[rare_mask])) * 100)
        dom_r.append(float(np.mean(pc[dom_mask] == oracle_col[dom_mask])) * 100)
        dists.append(np.bincount(pc, minlength=N_CAM) / N * 100)
        rec = np.array([np.mean(pc[oracle_col == c] == c) * 100 if (oracle_col == c).any()
                        else np.nan for c in range(N_CAM)])
        recalls.append(rec)

    pred_pct = np.mean(dists, axis=0)
    recall_mean = np.nanmean(recalls, axis=0)
    return dict(
        norm_pct=float(np.mean(norms)), top1=float(np.mean(top1s)),
        tol5=float(np.mean(tol5)), tol10=float(np.mean(tol10)),
        cam58_share=float(pred_pct[4] + pred_pct[7]),
        rare_recall=float(np.mean(rare_r)), dom_recall=float(np.mean(dom_r)),
        n_zero_recall=int(np.sum(np.nan_to_num(recall_mean, nan=100.0) < 0.05)),
        pred_pct=pred_pct.tolist(), recall_mean=recall_mean.tolist(),
        norm_per_pos=[float(x) for x in norms],
        random_px=random_px, oracle_px=oracle_px, N=N)


def spearman_diag(U, mpjpe):
    """Per-frame rank correlation between the utility and -mpjpe across the 8
    candidates, averaged over frames and starting positions. This separates
    'the utility carries ordering signal but argmax loses it' from 'there is
    no signal'. Chance = 0."""
    rs = []
    for pos in range(N_CAM):
        for i in range(U.shape[0]):
            r = stats.spearmanr(U[i, pos], -mpjpe[i]).statistic
            if np.isfinite(r):
                rs.append(r)
    rs = np.asarray(rs)
    t, p = stats.ttest_1samp(rs, 0.0)
    return dict(mean=float(rs.mean()), sd=float(rs.std(ddof=1)), n=int(rs.size),
                t=float(t), p=float(p))


def one_sample_vs_constant(mean, sd, n, const):
    """t-test of a seed distribution (mean, sd, n) against a deterministic
    constant. Equivalent to scipy.stats.ttest_1samp on the raw seeds."""
    se = sd / np.sqrt(n)
    t = (mean - const) / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    half = stats.t.ppf(0.975, df=n - 1) * se
    return dict(delta=float(mean - const), t=float(t), p=float(p),
                ci=[float(mean - const - half), float(mean - const + half)])


# --------------------------------------------------------------------------
def run_split(name, args):
    feats, mpjpe, kp = load_split(name)
    align = check_field_matches_current_view(feats, kp)
    U = build_utilities(feats)
    ship_err = check_matches_shipped_scalar(feats, U['need_vis'])
    fixed = fixed_camera_reference(mpjpe)
    passing = gate_mask(kp)

    arms = {}
    for key in ['need_vis', 'vis_only', 'need_inframe', 'need_notocc',
                'need_vis_min', 'need_only']:
        arms[key] = dict(
            ungated=score(argmax_policy(U[key]), mpjpe),
            gated=score(argmax_policy(U[key], passing=passing), mpjpe))
    arms['need_vis_xcur'] = dict(
        ungated=score(argmax_policy(U['need_vis'], exclude_current=True), mpjpe),
        gated=score(argmax_policy(U['need_vis'], exclude_current=True,
                                  passing=passing), mpjpe))

    out = dict(split=name, N=int(mpjpe.shape[0]), align_max_err=align,
               shipped_scalar_max_err=ship_err, fixed_camera=fixed,
               gate=dict(conf=GATE_CONF, min_vis=GATE_MINVIS,
                         pass_pct=float(passing.mean() * 100),
                         n_allfail=int((~passing.any(axis=1)).sum())),
               arms=arms)
    if args.spearman:
        out['spearman_need_vis'] = spearman_diag(U['need_vis'], mpjpe)
    return out


def fmt_split(o, fh):
    w = lambda s='': print(s, file=fh)
    a = o['arms']
    base_u = a['need_vis']['ungated']
    w('=' * 108)
    w(f"SPLIT {o['split']}   N={o['N']}   random={base_u['random_px']:.1f}px  "
      f"oracle={base_u['oracle_px']:.1f}px")
    w(f"  gate {GATE_CONF}/{GATE_MINVIS}: {o['gate']['pass_pct']:.1f}% of "
      f"(frame,candidate) cells pass, {o['gate']['n_allfail']} all-fail frames")
    ship = o['shipped_scalar_max_err']
    ship_s = (f"{ship:.1e} (argmax agreement 1.000)" if ship is not None
              else 'SKIPPED -- torch not importable, this run is UNVERIFIED')
    w(f"  checks: cached field vs keypoints {o['align_max_err']:.1e} | "
      f"utility vs ReprojJointFeatures.exp132_scalar {ship_s}")
    cal = (' -- docs/gated_baselines.txt publishes 75.28, must match'
           if o['split'] == 'test' else ' (test is the calibrated split)')
    w(f"  score() calibration: always-cam5 = {o['fixed_camera']['cam5']:.2f}{cal}")
    w()
    w(f"  {'arm':<15} {'norm%':>8} {'top1':>7} {'tol5':>7} {'tol10':>7} "
      f"{'cam5+8':>8} {'rare':>7} {'0%rec':>6}   {'gated norm%':>11} {'gated top1':>10}")
    w('  ' + '-' * 104)
    order = ['need_vis', 'need_vis_xcur', 'vis_only', 'need_inframe',
             'need_notocc', 'need_only', 'need_vis_min']
    for k in order:
        u, g = a[k]['ungated'], a[k]['gated']
        star = ' *' if k == 'need_vis' else '  '
        w(f"{star}{k:<15} {u['norm_pct']:>8.2f} {u['top1']:>7.2f} {u['tol5']:>7.2f} "
          f"{u['tol10']:>7.2f} {u['cam58_share']:>8.1f} {u['rare_recall']:>7.2f} "
          f"{u['n_zero_recall']:>6d}   {g['norm_pct']:>11.2f} {g['top1']:>10.2f}")
    w()
    w('  NOT A POLICY: need_only has no candidate index (mean_j(1-q_j) is the same')
    w(f"  for every c), so its argmax is always camera 0. Its row is exactly")
    w(f"  always-cam0 = {o['fixed_camera']['cam0']:.2f}. It is printed to show the "
      "degeneracy, not")
    w('  as a competitor -- do not read it as "the need term alone scores well".')
    w()
    w('  fixed-camera reference [' + ' '.join(f'cam{c}' for c in CAM_IDS) + ']:')
    w('    ' + ' '.join(f'{o["fixed_camera"][f"cam{c}"]:7.2f}' for c in CAM_IDS))
    w('  pick distribution, need_vis ungated [cam0 cam1 cam2 cam4 cam5 cam6 cam7 cam8]:')
    w('    ' + ' '.join(f'{v:5.1f}' for v in base_u['pred_pct']))
    w('  per-starting-position norm%: '
      + ' '.join(f'{v:6.2f}' for v in base_u['norm_per_pos']))
    if 'spearman_need_vis' in o:
        s = o['spearman_need_vis']
        w(f"  within-frame Spearman(U_c, -mpjpe_c): {s['mean']:+.4f} "
          f"(sd {s['sd']:.3f}, n={s['n']}, p={s['p']:.2e})   chance = 0")
    w()


def fmt_verdict(test, fh):
    w = lambda s='': print(s, file=fh)
    a = test['arms']
    b_u = a['need_vis']['ungated']['norm_pct']
    b_g = a['need_vis']['gated']['norm_pct']
    w('=' * 108)
    w('TEST-SPLIT COMPARISON TABLE  (norm%, 8-starting-position average)')
    w('=' * 108)
    w(f"  {'policy':<44} {'norm%':>8}   {'reads':>34}")
    w('  ' + '-' * 90)
    w(f"  {'random camera':<44} {0.00:>8.2f}   {'nothing':>34}")
    w(f"  {'always-cam5 (fixed)':<44} {FIXED_CAM5_UNGATED:>8.2f}   {'nothing':>34}")
    w(f"  {'geo utility, argmax (THIS BASELINE)':<44} {b_u:>8.2f}   "
      f"{'current view only':>34}")
    w(f"  {'exp158_rand_full (142k+158)':<44} "
      f"{CONTROLS['exp158 ungated']['mean']:>8.2f}   {'current view only':>34}")
    w(f"  {'gate + random':<44} {GATE_RANDOM:>8.2f}   {'candidate confidences':>34}")
    w(f"  {'geo utility @ gate 0.3/10':<44} {b_g:>8.2f}   "
      f"{'current view + cand. conf':>34}")
    w(f"  {'exp158_rand_full @ gate 0.3/10':<44} "
      f"{CONTROLS['exp158 gate 0.3/10']['mean']:>8.2f}   "
      f"{'current view + cand. conf':>34}")
    w(f"  {'gate + oracle (ceiling, not a bar)':<44} {GATE_ORACLE:>8.2f}   "
      f"{'ground truth':>34}")
    w()
    w('  ONE-SAMPLE t-TESTS -- exp158 seed distribution vs this deterministic baseline')
    for label, const in [('exp158 ungated', b_u), ('exp158 gate 0.3/10', b_g)]:
        c = CONTROLS[label]
        r = one_sample_vs_constant(c['mean'], c['sd'], c['n'], const)
        w(f"    {label:<20} {c['mean']:.2f}+-{c['sd']:.2f} (n={c['n']}) vs "
          f"baseline {const:.2f}  ->  delta {r['delta']:+.2f}pp  "
          f"CI [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]  t={r['t']:.2f}  p={r['p']:.2e}")
        w(f"    {'':<20} source: {c['src']}")
    w()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', nargs='+', default=['test', 'val'])
    ap.add_argument('--spearman', action='store_true', default=True)
    ap.add_argument('--no-spearman', dest='spearman', action='store_false')
    ap.add_argument('--out_json', default=str(DOCS / 'geo_utility_baseline.json'))
    ap.add_argument('--out_txt', default=str(DOCS / 'geo_utility_baseline.txt'))
    args = ap.parse_args()

    results = {name: run_split(name, args) for name in args.splits}

    import sys
    for fh in (sys.stdout, open(args.out_txt, 'w')):
        print('=' * 108, file=fh)
        print('TRAINING-FREE GEOMETRIC UTILITY BASELINE for 142k+158', file=fh)
        print('  U_c = mean_j (1 - q_j^cur) * p_in_frame_cj * (1 - p_self_occ_cj),'
              '   pick = argmax_c U_c', file=fh)
        print('  == ReprojJointFeatures.exp132_scalar, the scalar exp142_K/exp158'
              ' inject as a rank-1 bias', file=fh)
        print('  Zero trained parameters beyond the frozen Gaussian lifter the'
              ' model already ships.', file=fh)
        print('=' * 108, file=fh)
        print(file=fh)
        for name in args.splits:
            fmt_split(results[name], fh)
        if 'test' in results:
            fmt_verdict(results['test'], fh)
        if fh is not sys.stdout:
            fh.close()

    json.dump(results, open(args.out_json, 'w'), indent=1)
    print(f'wrote {args.out_json}')
    print(f'wrote {args.out_txt}')


if __name__ == '__main__':
    main()
