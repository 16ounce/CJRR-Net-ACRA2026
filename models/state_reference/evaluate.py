"""Purpose: Evaluate the reference state ranker.

Main functionality: Compute camera-selection metrics over all starting views.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import LeanMultiViewDataset

DOMINANT = [4, 7]        # cam5, cam8 (SELECTED_CAM_IDS = [0,1,2,4,5,6,7,8])


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


@torch.no_grad()
def evaluate_per_position_state(model, data_dir, batch_size, device, split_name,
                                npz_name='r3_guard_state.npz'):
    """evaluate_per_position with the external state. Every line that
    computes a metric is S131's, reached by calling it with a model wrapper --
    the protocol, the masking and the metric definitions are not restated here,
    so the two can never drift."""
    z = np.load(Path(data_dir) / npz_name)
    st_mean = torch.from_numpy(z['mean'].astype(np.float32))
    st_var = torch.from_numpy(z['var'].astype(np.float32))

    class _Wrapped(nn.Module):
        """Forwards to `model`, injecting the state of whichever camera each
        sample is currently placed at. evaluate_per_position calls the model as
        model(kp_batch, cur_batch) with cur constant within a position sweep and
        slices rows in order, so the row offset is tracked from the batch sizes
        it hands over -- there is no other consumer of this class."""
        def __init__(self):
            super().__init__()
            self.m = model
            self.off = 0
            self.pos = None

        def eval(self):
            self.m.eval(); return self

        def train(self, mode=True):
            self.m.train(mode); return self

        def forward(self, kp, cur):
            pos = int(cur[0].item())
            if pos != self.pos:                       # a new position sweep began
                self.pos, self.off = pos, 0
            n = kp.size(0)
            sl = slice(self.off, self.off + n)
            self.off += n
            mean = st_mean[sl, pos].to(kp.device)
            var = st_var[sl, pos].to(kp.device)
            return self.m(kp, cur, external_state=(mean, var))

    return evaluate_per_position(_Wrapped(), data_dir, batch_size, device, split_name)
