"""Purpose: Train the adapted PoseErrNet baseline.

Main functionality: Reuse the CJRR-Net data and evaluation protocol for the pose-error arm.

Usage:
    python baselines/poseerrnet_adapted/train.py --help
"""

import argparse, importlib.util, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# This experiment lives in /srv/scratch/z5511538/baseline/ but deliberately
# reuses bvpnet's machinery rather than forking it: the dataset class, the
# optimiser/schedule, the loss terms and the entire 8-position single-
# observation evaluation harness all come from stage131/143/144, so exp180's
# numbers are produced by the SAME code that produced exp142_K's, exp157's and
# exp160's. Forking any of it would make the comparison meaningless.
# Both paths are absolute so moving this file again cannot silently break it.
HERE = Path(__file__).resolve().parent
BVPNET = Path(__file__).resolve().parents[2] / 'models' / 'cjrrnet'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S131 = _load('stage131', BVPNET / 'training_core.py')
S143 = _load('stage143', BVPNET / 'view_dropout.py')
S144 = _load('stage144', BVPNET / 'train.py')
E180 = _load('exp180', HERE / 'model.py')

# arm -> (pose_norm, head, objective, use_conf, rot_sign)
ARMS = {
    'uav':         ('body',  'shared_geo', 'mse', True,  'fixed'),
    'uav_imgnorm': ('image', 'shared_geo', 'mse', True,  'fixed'),
    'uav_fixed8':  ('body',  'fixed8',     'mse', True,  'fixed'),
    'uav_ce':      ('body',  'shared_geo', 'ce',  True,  'fixed'),
    'base_mlp':    ('image', 'fixed8',     'ce',  True,  'fixed'),
    'poseerr':     ('body',  'fixed8',     'mse', False, 'reference'),
}

DROPOUT_ARM = 'k1'   # see header: dropout is a no-op for this architecture


def compute_loss(logits, mpjpe, objective, args):
    """logits are NEGATED predicted error, so -logits is the prediction."""
    if objective == 'mse':
        target = mpjpe / (mpjpe.mean(dim=1, keepdim=True) + 1e-8)
        return F.mse_loss(-logits, target)
    total, *_ = S144.compute_losses(logits, mpjpe, args)
    return total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    pose_norm, head, objective, use_conf, rot_sign = ARMS[args.arm]
    args.lambda_regret, args.lambda_pair = 0.0, 0.0
    train_k, ref_mode = S143.ARMS[DROPOUT_ARM]
    view_dropout = S143.make_view_dropout(train_k, ref_mode)

    model = E180.UAVGeoMLP(
        n_cands=8, n_joints=17, n_features=3, d_hidden=args.d_hidden,
        d_latent=args.d_latent, d_dec=args.d_dec, dropout=args.dropout_prob,
        pose_norm=pose_norm, head=head, use_conf=use_conf,
        rot_sign=rot_sign).to(device)

    train_ds = S131.LeanMultiViewDataset(args.train_data)
    val_ds = S131.LeanMultiViewDataset(args.val_data)
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=pin)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=5)
    best_val, patience_cnt, history = float('inf'), 0, []
    t0 = time.time()
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'arm={args.arm}  pose_norm={pose_norm}  head={head}  objective={objective}  '
          f'use_conf={use_conf}  rot_sign={rot_sign}  dropout={DROPOUT_ARM}  seed={args.seed}')
    print(f'params={total_p:,} trainable={trainable_p:,}  '
          f'(exp160 noattn = 192,994; shipped exp142_K = 325,090)\n')

    for epoch in range(1, args.epochs + 1):
        model.train(); tot, n_tot = 0.0, 0
        for kp, sc in train_dl:
            kp_aug, ref = view_dropout(kp)
            kp_aug, sc, ref = kp_aug.to(device), sc.to(device), ref.to(device)
            optimizer.zero_grad()
            loss = compute_loss(model(kp_aug, ref), sc, objective, args)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            tot += loss.item() * kp.size(0); n_tot += kp.size(0)
        tr_loss = tot / n_tot

        model.eval(); vs, vn = 0.0, 0
        with torch.no_grad():
            for kp, sc in val_dl:
                kp_aug, ref = view_dropout(kp)
                kp_aug, sc, ref = kp_aug.to(device), sc.to(device), ref.to(device)
                vs += compute_loss(model(kp_aug, ref), sc, objective, args).item() * kp.size(0)
                vn += kp.size(0)
        val_loss = vs / vn
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
    val_m = S131.evaluate_per_position(model, args.val_data, args.batch_size, device, 'val')
    test_m = S131.evaluate_per_position(model, args.test_data, args.batch_size, device, 'test(S8)')
    meta = dict(tier=f'exp180_{args.arm}', exp='exp180', arm=args.arm,
                pose_norm=pose_norm, head=head, objective=objective,
                use_conf=use_conf, rot_sign=rot_sign,
                dropout_arm=DROPOUT_ARM, train_k=train_k, ref_mode=ref_mode,
                lambda_lr=args.lambda_lr, total_params=total_p,
                trainable_params=trainable_p, training_time_s=time.time() - t0,
                seed=args.seed, best_epoch=int(ckpt['epoch']),
                best_val_loss=float(ckpt['val_loss']))
    val_m.update(meta); test_m.update(meta)
    json.dump(val_m, open(save_dir / 'val_metrics.json', 'w'), indent=2)
    json.dump(test_m, open(save_dir / 'test_metrics.json', 'w'), indent=2)
    print(f'\nSaved -> {save_dir}/test_metrics.json')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--train_data', required=True)
    p.add_argument('--val_data', required=True)
    p.add_argument('--test_data', required=True)
    p.add_argument('--save_dir', required=True)
    p.add_argument('--lambda_lr', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--dropout_prob', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--d_hidden', type=int, default=256)
    p.add_argument('--d_latent', type=int, default=128)
    p.add_argument('--d_dec', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
