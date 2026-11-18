"""Purpose: Train the adapted ActiveMoCap-inspired planner.

Main functionality: Fit its uncertainty head and evaluate camera selections.

Usage:
    python baselines/activemocap_inspired_planner/train.py --help
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BVPNET = Path(__file__).resolve().parent
import sys                                                        # noqa: E402
sys.path.insert(0, str(BVPNET))
sys.path.insert(0, str(BVPNET.parents[1] / 'utils' / 'activemocap'))
from planner import (           # noqa: E402
    StaticActiveViewPlanner, CONFIGS)
from mpi_camera_geometry import SELECTED_CAM_IDS                  # noqa: E402

IMAGE_SIZE = 2048.0
DOMINANT = [4, 7]                 # cam5, cam8 in SELECTED_CAM_IDS order
FACE_JOINTS = [0, 1, 2, 3, 4]     # all map to MPI 'head'
PX_CLIP = 400.0                   # px; residuals above this are detector
                                  # failures, not localisation noise


class StaticPlannerDataset(Dataset):
    """Returns everything the losses need, per frame.

    kp        (8,17,3)   all 8 cameras' detections (only the observed slot is
                         fed to the model; the others are TARGETS only)
    mpjpe     (8,)
    vis       (8,17)     1 = visible
    err2d     (8,17,2)   |proj(GT3D) - detection| per axis, px
    has_joint bool       whether joint-level labels were found
    """

    def __init__(self, data_dir, cam_ids=SELECTED_CAM_IDS, need_joint=True):
        from mpi_camera_geometry import parse_calibration
        d = Path(data_dir)
        self.kp = np.load(d / 'keypoints_mv.npy').astype(np.float32)
        self.mpjpe = np.load(d / 'mpjpe_per_cam.npy').astype(np.float32)
        assert len(self.kp) == len(self.mpjpe)
        self.has_joint = ((d / 'occlusion_mask.npy').exists()
                          and (d / 'lift3d_target_mv.npy').exists())
        N, C, J, _ = self.kp.shape
        if self.has_joint and need_joint:
            occ = np.load(d / 'occlusion_mask.npy').astype(np.float32)
            tgt = np.load(d / 'lift3d_target_mv.npy').astype(np.float32)
            cams = parse_calibration()
            K = np.stack([cams[c]['K'] for c in cam_ids]).astype(np.float32)
            Z = np.clip(tgt[..., 2], 1e-3, None)
            u = K[None, :, 0, 0, None] * tgt[..., 0] / Z + K[None, :, 0, 2, None]
            v = K[None, :, 1, 1, None] * tgt[..., 1] / Z + K[None, :, 1, 2, None]
            uk = (self.kp[..., 0] + 0.5) * IMAGE_SIZE
            vk = (self.kp[..., 1] + 0.5) * IMAGE_SIZE
            self.vis = (1.0 - occ).astype(np.float32)
            self.err2d = np.stack([u - uk, v - vk], axis=-1).astype(np.float32)
        else:
            self.vis = np.zeros((N, C, J), dtype=np.float32)
            self.err2d = np.zeros((N, C, J, 2), dtype=np.float32)

    def __len__(self):
        return len(self.kp)

    def __getitem__(self, i):
        return (torch.from_numpy(self.kp[i]), torch.from_numpy(self.mpjpe[i]),
                torch.from_numpy(self.vis[i]), torch.from_numpy(self.err2d[i]))


def observed_only(kp, cur):
    """(B,8,17,3) -> the K=1 deployment input: only slot `cur` populated."""
    out = torch.zeros_like(kp)
    b = torch.arange(kp.size(0), device=kp.device)
    out[b, cur] = kp[b, cur]
    return out


def lambda_rank_loss(logits, norm_mpjpe):
    """Identical to stage131's -- same ranking objective the family uses."""
    d_score = logits.unsqueeze(1) - logits.unsqueeze(2)
    d_mpjpe = norm_mpjpe.unsqueeze(1) - norm_mpjpe.unsqueeze(2)
    loss = (F.relu(d_mpjpe) * F.softplus(d_score)).sum(dim=(1, 2)).mean()
    N = logits.size(1)
    return loss / (N * (N - 1) / 2)


def losses(model, aux, scores, mpjpe, vis, err2d, args, joint_mask):
    """Returns (total, parts dict). Every term is optional and reported."""
    parts = {}
    zero = torch.zeros((), device=scores.device)

    # --- ranking (weak, camera-level) -----------------------------------
    logits = model.calibrated_logits(scores)
    norm = mpjpe / (mpjpe.mean(dim=1, keepdim=True) + 1e-8)
    l_ce = F.cross_entropy(logits, torch.argmin(mpjpe, dim=-1))
    l_lr = lambda_rank_loss(logits, norm)
    total = args.lambda_ce * l_ce + args.lambda_lr * l_lr
    parts['ce'], parts['lr'] = l_ce.detach(), l_lr.detach()

    if args.supervision == 'camera':
        return total, parts

    # --- visibility BCE (joint-level) -----------------------------------
    l_vis = F.binary_cross_entropy_with_logits(
        aux['vis_logit'].clamp(-20, 20), vis)
    total = total + args.lambda_vis * l_vis
    parts['vis'] = l_vis.detach()

    # --- heteroscedastic 2D NLL, visible joints only --------------------
    # An occluded joint's "2D error" is whatever the detector hallucinated;
    # regressing localisation noise on it would teach the head nonsense.
    e = err2d.clamp(-PX_CLIP, PX_CLIP)
    var = aux['var_2d']
    w = (vis.unsqueeze(-1) * joint_mask.view(1, 1, -1, 1)).expand_as(e)
    nll = 0.5 * (e ** 2 / var + torch.log(var))
    l_nll = (nll * w).sum() / w.sum().clamp_min(1.0)
    total = total + args.lambda_nll * l_nll
    parts['nll'] = l_nll.detach()

    # --- calibration: E[z^2] should be 1 --------------------------------
    z2 = (e ** 2 / var)
    m = (z2 * w).sum() / w.sum().clamp_min(1.0)
    l_cal = (m - 1.0) ** 2
    total = total + args.lambda_cal * l_cal
    parts['cal'] = l_cal.detach()
    parts['Ez2'] = m.detach()
    return total, parts


@torch.no_grad()
def evaluate_per_position(model, data_dir, batch_size, device, split_name,
                          quiet=False):
    """The protocol every experiment in this line uses (stage131), verbatim:
    feed ONE camera at a time, argmax, average over the 8 starting positions."""
    ds = StaticPlannerDataset(data_dir, need_joint=False)
    mp = ds.mpjpe
    N, C = mp.shape
    random_px = float(mp.mean())
    oracle_col = np.argmin(mp, axis=1)
    oracle_px = float(mp[np.arange(N), oracle_col].mean())
    rare = ~np.isin(oracle_col, DOMINANT)
    model.eval()
    kp_all = torch.from_numpy(ds.kp)

    per_pos, preds_all = [], []
    for pos in range(C):
        x = torch.zeros_like(kp_all); x[:, pos] = kp_all[:, pos]
        out = []
        for i in range(0, N, batch_size):
            cur = torch.full((min(batch_size, N - i),), pos, dtype=torch.long,
                             device=device)
            out.append(model(x[i:i + batch_size].to(device), cur).cpu().numpy())
        pc = np.argmax(np.concatenate(out), axis=1)
        preds_all.append(pc)
        pred_px = mp[np.arange(N), pc]
        per_pos.append(dict(position=pos,
                            norm_pct=(random_px - float(pred_px.mean()))
                            / (random_px - oracle_px + 1e-8) * 100,
                            top1=float(np.mean(pc == oracle_col))))

    pred_pct = np.mean([np.bincount(pc, minlength=C) / N * 100
                        for pc in preds_all], axis=0)
    recall = np.nanmean([[float(np.mean(pc[oracle_col == c] == c)) * 100
                          if (oracle_col == c).sum() else np.nan
                          for c in range(C)] for pc in preds_all], axis=0)
    regret = np.mean([float((mp[np.arange(N), pc]
                             - mp[np.arange(N), oracle_col]).mean())
                      for pc in preds_all])
    out = dict(per_position=per_pos,
               avg_norm_pct=float(np.mean([p['norm_pct'] for p in per_pos])),
               avg_top1=float(np.mean([p['top1'] for p in per_pos])),
               oracle_regret_mm=float(regret),
               cam58_share=float(pred_pct[4] + pred_pct[7]),
               pred_pct_mean=pred_pct.tolist(), recall_mean=recall.tolist(),
               rare_recall=float(np.mean(
                   [float(np.mean(pc[rare] == oracle_col[rare])) * 100
                    for pc in preds_all])),
               cam1_recall=float(recall[1]),
               random_px=random_px, oracle_px=oracle_px)
    if not quiet:
        print(f'[{split_name}] norm%={out["avg_norm_pct"]:.2f}  '
              f'top1={out["avg_top1"]*100:.1f}  regret={out["oracle_regret_mm"]:.1f}mm  '
              f'cam5+8={out["cam58_share"]:.1f}  rare_recall={out["rare_recall"]:.2f}')
    return out


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)

    model = StaticActiveViewPlanner(
        config=args.config, lift3d_ckpt=args.lift3d_ckpt,
        utility_mode=args.utility_mode, seed=args.seed).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f'config={args.config} utility={args.utility_mode} seed={args.seed}  '
          f'trainable={n_tr:,}')
    if not CONFIGS[args.config][3]:
        print('  NOTE: this config has no uncertainty head. The only trainable\n'
              '        parameters are the MONOTONE score affine, which cannot\n'
              '        change argmax or any ranking metric. Train F to learn\n'
              '        anything that affects selection.')

    train_ds = StaticPlannerDataset(args.train_data)
    val_ds = StaticPlannerDataset(args.val_data)
    if args.supervision == 'joint' and not train_ds.has_joint:
        print('  joint-level labels absent -> falling back to --supervision camera')
        args.supervision = 'camera'
    print(f'  supervision={args.supervision}  '
          f'train={len(train_ds)} val={len(val_ds)}')

    joint_mask = torch.ones(17, device=device)
    if not args.face_joints_in_nll:
        joint_mask[FACE_JOINTS] = 0.0

    dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=0)
    vdl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                     num_workers=0)
    opt = torch.optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                       factor=0.5, patience=3)
    best, patience, hist = float('inf'), 0, []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tl, tn = 0.0, 0
        for kp, mpjpe, vis, err in dl:
            B = kp.size(0)
            cur = torch.randint(0, 8, (B,))
            x = observed_only(kp, cur).to(device)
            cur, mpjpe = cur.to(device), mpjpe.to(device)
            vis, err = vis.to(device), err.to(device)
            opt.zero_grad()
            scores, aux = model(x, cur, return_aux=True)
            loss, _ = losses(model, aux, scores, mpjpe, vis, err, args, joint_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            tl += loss.item() * B; tn += B
        tr_loss = tl / tn

        model.eval()
        vl, vn, agg = 0.0, 0, {}
        with torch.no_grad():
            for kp, mpjpe, vis, err in vdl:
                B = kp.size(0)
                cur = torch.arange(B) % 8
                x = observed_only(kp, cur).to(device)
                cur, mpjpe = cur.to(device), mpjpe.to(device)
                vis, err = vis.to(device), err.to(device)
                scores, aux = model(x, cur, return_aux=True)
                loss, parts = losses(model, aux, scores, mpjpe, vis, err,
                                     args, joint_mask)
                vl += loss.item() * B; vn += B
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0.0) + float(v) * B
        val_loss = vl / vn
        sched.step(val_loss)
        hist.append(dict(epoch=epoch, train_loss=tr_loss, val_loss=val_loss,
                         **{k: v / vn for k, v in agg.items()}))
        if epoch == 1 or epoch % 5 == 0:
            det = '  '.join(f'{k}={v/vn:.4f}' for k, v in agg.items())
            print(f'[{epoch:>3}/{args.epochs}] train={tr_loss:.4f} '
                  f'val={val_loss:.4f}  {det}')

        ck = dict(epoch=epoch, model_state_dict=model.state_dict(),
                  val_loss=val_loss, args=vars(args))
        torch.save(ck, save_dir / 'last.pt')
        if val_loss < best:
            best, patience = val_loss, 0
            torch.save(ck, save_dir / 'best.pt')
        else:
            patience += 1
            if patience >= args.patience:
                print(f'Early stop at epoch {epoch}'); break

    ck = torch.load(save_dir / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(ck['model_state_dict'])
    print(f'\nTraining time {(time.time()-t0)/60:.1f} min, '
          f'best epoch {ck["epoch"]}')
    json.dump(hist, open(save_dir / 'train_history.json', 'w'), indent=2)

    meta = dict(config=args.config, utility_mode=args.utility_mode,
                seed=args.seed, supervision=args.supervision,
                trainable_params=n_tr, best_epoch=int(ck['epoch']),
                best_val_loss=float(ck['val_loss']))
    for split, path in (('val', args.val_data), ('test', args.test_data)):
        m = evaluate_per_position(model, path, args.batch_size, device, split)
        m.update(meta)
        json.dump(m, open(save_dir / f'{split}_metrics.json', 'w'), indent=2)
    print(f'Saved -> {save_dir}')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', choices=list(CONFIGS), default='F')
    p.add_argument('--utility_mode', default='global_logdet',
                   choices=['global_logdet', 'uncertainty_weighted_joints'])
    p.add_argument('--supervision', default='joint', choices=['joint', 'camera'])
    p.add_argument('--train_data', required=True)
    p.add_argument('--val_data', required=True)
    p.add_argument('--test_data', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--lift3d_ckpt', default=str(
        BVPNET.parents[1] / 'weights' / 'lifter_bypassed'
        / 'lift3d_gaussian_mlp_best.pt'))
    p.add_argument('--lambda_ce', type=float, default=1.0)
    p.add_argument('--lambda_lr', type=float, default=0.1)
    p.add_argument('--lambda_vis', type=float, default=1.0)
    p.add_argument('--lambda_nll', type=float, default=0.01)
    p.add_argument('--lambda_cal', type=float, default=0.1)
    p.add_argument('--face_joints_in_nll', action='store_true')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
