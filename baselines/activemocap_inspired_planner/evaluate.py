"""Purpose: Evaluate the adapted ActiveMoCap-inspired planner.

Main functionality: Measure view selection, calibration, and latency across planner configurations.

Usage:
    python baselines/activemocap_inspired_planner/evaluate.py --help
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy import stats

BVPNET = Path(__file__).resolve().parent
import sys                                                        # noqa: E402
sys.path.insert(0, str(BVPNET))
sys.path.insert(0, str(BVPNET.parents[1] / 'utils' / 'activemocap'))
from planner import (           # noqa: E402
    StaticActiveViewPlanner, CONFIGS)
from mpi_camera_geometry import parse_calibration, SELECTED_CAM_IDS  # noqa: E402

DOMINANT = [4, 7]
FACE_JOINTS = [0, 1, 2, 3, 4]
IMAGE_SIZE = 2048.0
WEIGHTS = Path(__file__).resolve().parents[2] / 'weights'


def load_split(d):
    d = Path(d)
    out = dict(kp=np.load(d / 'keypoints_mv.npy').astype(np.float32),
               mpjpe=np.load(d / 'mpjpe_per_cam.npy').astype(np.float32))
    if (d / 'occlusion_mask.npy').exists():
        out['vis'] = 1.0 - np.load(d / 'occlusion_mask.npy').astype(np.float32)
    if (d / 'lift3d_target_mv.npy').exists():
        tgt = np.load(d / 'lift3d_target_mv.npy').astype(np.float32)
        cams = parse_calibration()
        K = np.stack([cams[c]['K'] for c in SELECTED_CAM_IDS]).astype(np.float32)
        Z = np.clip(tgt[..., 2], 1e-3, None)
        u = K[None, :, 0, 0, None] * tgt[..., 0] / Z + K[None, :, 0, 2, None]
        v = K[None, :, 1, 1, None] * tgt[..., 1] / Z + K[None, :, 1, 2, None]
        out['err2d'] = np.stack([u - (out['kp'][..., 0] + 0.5) * IMAGE_SIZE,
                                 v - (out['kp'][..., 1] + 0.5) * IMAGE_SIZE],
                                axis=-1).astype(np.float32)
    return out


@torch.no_grad()
def all_scores(model, kp, batch_size, device):
    """(8 positions, N, 8) raw scores under the per-position protocol."""
    N = kp.shape[0]
    x_all = torch.from_numpy(kp)
    out = []
    for pos in range(8):
        x = torch.zeros_like(x_all); x[:, pos] = x_all[:, pos]
        chunks = []
        for i in range(0, N, batch_size):
            n = min(batch_size, N - i)
            cur = torch.full((n,), pos, dtype=torch.long, device=device)
            chunks.append(model(x[i:i + n].to(device), cur).cpu().numpy())
        out.append(np.concatenate(chunks))
    return np.stack(out)                                     # (8,N,8)


def selection_metrics(scores, mpjpe):
    """scores (P,N,C), mpjpe (N,C) -> metrics + per-frame selected error."""
    P, N, C = scores.shape
    random_px = float(mpjpe.mean())
    oracle_col = np.argmin(mpjpe, axis=1)
    oracle_px = float(mpjpe[np.arange(N), oracle_col].mean())
    rare = ~np.isin(oracle_col, DOMINANT)

    picks = np.argmax(scores, axis=2)                        # (P,N)
    sel = mpjpe[np.arange(N)[None, :], picks]                # (P,N)
    norm = (random_px - sel.mean(1)) / (random_px - oracle_px + 1e-8) * 100
    top1 = (picks == oracle_col[None, :]).mean(1)
    regret = (sel - mpjpe[np.arange(N), oracle_col][None, :]).mean(1)
    pct = np.stack([np.bincount(p, minlength=C) / N * 100 for p in picks])
    recall = np.stack([[float((p[oracle_col == c] == c).mean()) * 100
                        if (oracle_col == c).sum() else np.nan
                        for c in range(C)] for p in picks])

    # Spearman within frame, across the 8 candidates: does a higher score mean
    # a lower true per-camera MPJPE?
    rho = []
    for p in range(P):
        r = stats.spearmanr(scores[p], -mpjpe, axis=1)[0]
        r = np.diag(r[:N, N:]) if np.ndim(r) == 2 else np.array([r])
        rho.append(np.nanmean(r))

    return dict(norm_pct=float(norm.mean()), norm_per_pos=norm.tolist(),
                top1=float(top1.mean()) * 100,
                oracle_regret_mm=float(regret.mean()),
                cam58_share=float(pct.mean(0)[4] + pct.mean(0)[7]),
                pred_pct=pct.mean(0).tolist(),
                recall=np.nanmean(recall, axis=0).tolist(),
                rare_recall=float(np.mean(
                    [(p[rare] == oracle_col[rare]).mean() * 100 for p in picks])),
                cam1_recall=float(np.nanmean(recall[:, 1])),
                spearman_score_vs_mpjpe=float(np.mean(rho)),
                random_px=random_px, oracle_px=oracle_px), sel.mean(0)


@torch.no_grad()
def calibration(model, kp, split, batch_size, device, n_frames=256):
    """E[z^2], 1-sigma / 2-sigma coverage of the predicted 2D variance on
    VISIBLE joints, excluding the 5 face joints whose GT 2D is a shared point."""
    if 'err2d' not in split or 'vis' not in split or model.config == 'A':
        return None
    N = min(n_frames, kp.shape[0])
    x_all = torch.from_numpy(kp[:N])
    err = torch.from_numpy(split['err2d'][:N])
    vis = torch.from_numpy(split['vis'][:N])
    jm = torch.ones(17); jm[FACE_JOINTS] = 0.0
    z2s, cov1, cov2, n = 0.0, 0.0, 0.0, 0.0
    for pos in range(8):
        x = torch.zeros_like(x_all); x[:, pos] = x_all[:, pos]
        for i in range(0, N, batch_size):
            k = min(batch_size, N - i)
            cur = torch.full((k,), pos, dtype=torch.long, device=device)
            _, aux = model(x[i:i + k].to(device), cur, return_aux=True)
            var = aux['var_2d'].cpu()
            e = err[i:i + k].clamp(-400, 400)
            w = (vis[i:i + k].unsqueeze(-1) * jm.view(1, 1, -1, 1)).expand_as(e)
            z = e / var.sqrt()
            z2s += float((z ** 2 * w).sum()); n += float(w.sum())
            cov1 += float(((z.abs() <= 1) * w).sum())
            cov2 += float(((z.abs() <= 2) * w).sum())
    return dict(mean_z2=z2s / max(n, 1), coverage_1sigma=cov1 / max(n, 1),
                coverage_2sigma=cov2 / max(n, 1), n=int(n))


@torch.no_grad()
def latency(model, kp, device, batch=64, reps=5):
    x = torch.zeros(batch, 8, 17, 3)
    cur = torch.arange(batch) % 8
    x[torch.arange(batch), cur] = torch.from_numpy(kp[:batch])[torch.arange(batch), cur]
    x, cur = x.to(device), cur.to(device)
    model(x, cur)
    t0 = time.time()
    for _ in range(reps):
        model(x, cur)
    return (time.time() - t0) / reps / batch * 1000.0


def build(cfg, mode, args, device, ckpt=None):
    kw = {}
    if cfg == 'A':
        kw['exp142k_ckpt'] = args.exp142k_ckpt
    m = StaticActiveViewPlanner(config=cfg, lift3d_ckpt=args.lift3d_ckpt,
                                utility_mode=mode, seed=args.seed, **kw)
    if ckpt is not None:
        ck = torch.load(ckpt, map_location='cpu', weights_only=False)
        m.load_state_dict(ck['model_state_dict'])
    return m.to(device).eval()


def main(args):
    device = torch.device(args.device)
    split = load_split(args.test_data)
    kp, mpjpe = split['kp'], split['mpjpe']
    print(f'test split: {kp.shape[0]} frames, random={mpjpe.mean():.1f} mm, '
          f'oracle={mpjpe[np.arange(len(mpjpe)), mpjpe.argmin(1)].mean():.1f} mm\n')

    results, per_frame = {}, {}
    for cfg in args.configs:
        modes = ['global_logdet'] if cfg == 'A' else args.utility_modes
        for mode in modes:
            ckpts = args.f_ckpt if (cfg == 'F' and args.f_ckpt) else [None]
            for ck in ckpts:
                tag = f'{cfg}' + ('' if cfg == 'A' else f'/{mode}') \
                    + (f'/{Path(ck).parent.name}' if ck else '')
                m = build(cfg, mode, args, device, ck)
                sc = all_scores(m, kp, args.batch_size, device)
                met, pf = selection_metrics(sc, mpjpe)
                met['calibration'] = calibration(m, kp, split, args.batch_size, device)
                met['latency_ms_per_frame'] = latency(m, kp, device)
                met['trainable_params'] = sum(p.numel() for p in m.parameters()
                                              if p.requires_grad)
                met['config'], met['utility_mode'], met['ckpt'] = cfg, mode, ck
                results[tag], per_frame[tag] = met, pf
                print(f'{tag:34s} norm%={met["norm_pct"]:6.2f}  '
                      f'top1={met["top1"]:5.2f}  regret={met["oracle_regret_mm"]:7.1f}mm  '
                      f'cam5+8={met["cam58_share"]:5.1f}  rare={met["rare_recall"]:5.2f}  '
                      f'cam1={met["cam1_recall"]:5.2f}  rho={met["spearman_score_vs_mpjpe"]:+.3f}  '
                      f'{met["latency_ms_per_frame"]:.2f}ms')
                if met['calibration']:
                    c = met['calibration']
                    print(f'{"":34s} calibration E[z^2]={c["mean_z2"]:.2f} '
                          f'(1.0 = calibrated)  1sig={c["coverage_1sigma"]*100:.1f}% '
                          f'2sig={c["coverage_2sigma"]*100:.1f}%')

    # ---- paired contrasts vs config A, on per-frame selected MPJPE ----------
    if 'A' in per_frame:
        base = per_frame['A']
        print('\npaired vs A (exp142_K), per-frame selected MPJPE, '
              'lower is better:')
        for tag, pf in per_frame.items():
            if tag == 'A':
                continue
            d = pf - base
            t, p = stats.ttest_rel(pf, base)
            print(f'  {tag:34s} delta = {d.mean():+8.2f} mm  '
                  f'(t={t:+.2f}, p={p:.4g})  '
                  f'{"WORSE" if d.mean() > 0 else "better"}')
            results[tag]['paired_vs_A'] = dict(
                delta_mm=float(d.mean()), t=float(t), p=float(p))

    # ---- seed-level paired test across F checkpoints -----------------------
    ftags = [t for t in results if t.startswith('F/') and results[t]['ckpt']]
    if len(ftags) > 1 and 'A' in results:
        norms = np.array([results[t]['norm_pct'] for t in ftags])
        print(f'\nF over {len(ftags)} seeds: norm% = {norms.mean():.2f} '
              f'+- {norms.std(ddof=1):.2f}   (A = {results["A"]["norm_pct"]:.2f})')
        t, p = stats.ttest_1samp(norms, results['A']['norm_pct'])
        print(f'  one-sample vs A: t={t:+.2f}, p={p:.4g}')

    if args.out:
        json.dump(results, open(args.out, 'w'), indent=2)
        print(f'\nSaved -> {args.out}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--configs', nargs='+', default=['F'])
    p.add_argument('--utility_modes', nargs='+',
                   default=['global_logdet'])
    p.add_argument('--f_ckpt', nargs='+', required=True)
    p.add_argument('--test_data', required=True)
    p.add_argument('--lift3d_ckpt', default=str(WEIGHTS / 'lifter_bypassed' / 'lift3d_gaussian_mlp_best.pt'))
    p.add_argument('--exp142k_ckpt', default=None)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    return p.parse_args()


if __name__ == '__main__':
    torch.set_num_threads(4)
    main(parse_args())
