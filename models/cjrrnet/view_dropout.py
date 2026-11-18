"""Purpose: Provide observation-budget training controls.

Main functionality: Apply view dropout policies to the shared training pipeline.

Usage:
    python models/cjrrnet/view_dropout.py --help
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
from torch.utils.data import DataLoader

BVPNET = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# every piece of training math and the entire eval protocol come from stage131
S131 = _load('stage131', BVPNET / 'training_core.py')
CONFIGS = S131.CONFIGS

# arm -> (train_k, ref_mode).  train_k = 0 means "random, U{1..n_cams}".
ARMS = {
    'rand_first': (0, 'first'),    # shipped exp142_K protocol (reproduction control)
    'rand_uni':   (0, 'uniform'),  # random count, uniform starting position
    'k1':         (1, 'uniform'),  # deployment-matched (first==uniform at K=1)
    'k2':         (2, 'uniform'),
    'k4':         (4, 'uniform'),
    'k8':         (8, 'uniform'),  # all views always -- the mismatch floor
}


def make_view_dropout(train_k: int, ref_mode: str):
    """stage131's apply_view_dropout_with_ref with the two policy knobs exposed.

    train_k = 0 -> K ~ U{1..n_cams} (stage131's behaviour); otherwise K is
    fixed at min(train_k, n_cams).
    ref_mode = 'first' -> ref = min(keep) (stage131's behaviour, which skews
    the reference camera onto low indices); 'uniform' -> ref ~ U(keep).

    The (train_k=0, ref_mode='first') path consumes exactly randint then
    sample, in stage131's order, so it reproduces the shipped runs bit-for-bit.
    """
    def apply(kp: torch.Tensor):
        B, out = kp.size(0), kp.clone()
        n_cams = kp.size(1)
        K_fixed = min(train_k, n_cams) if train_k > 0 else 0
        ref = torch.zeros(B, dtype=torch.long)
        for b in range(B):
            K = random.randint(1, n_cams) if K_fixed == 0 else K_fixed
            keep = sorted(random.sample(range(n_cams), K))
            mask = torch.zeros(n_cams, dtype=torch.bool); mask[keep] = True
            out[b, ~mask] = 0.0
            ref[b] = keep[0] if ref_mode == 'first' else random.choice(keep)
        return out, ref
    return apply


def ref_distribution(train_k: int, ref_mode: str, n_cams: int = 8,
                     n_draws: int = 200000, seed: int = 0):
    """Empirical P(ref = position) under this arm's policy, logged with each run
    so the starting-position skew is on the record next to the metrics rather
    than only in this header."""
    rng = random.Random(seed)
    counts = np.zeros(n_cams, dtype=np.int64)
    K_fixed = min(train_k, n_cams) if train_k > 0 else 0
    for _ in range(n_draws):
        K = rng.randint(1, n_cams) if K_fixed == 0 else K_fixed
        keep = sorted(rng.sample(range(n_cams), K))
        counts[keep[0] if ref_mode == 'first' else rng.choice(keep)] += 1
    return (counts / n_draws * 100).tolist()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    train_k, ref_mode = ARMS[args.arm]
    view_dropout = make_view_dropout(train_k, ref_mode)

    joint_tokens, aux, film_mode, pool_mode = CONFIGS[args.config]
    if not aux:
        args.lambda_err, args.lambda_pair = 0.0, 0.0

    model = _load('exp142', BVPNET / 'cjrrnet.py').JointTokenReprojVisRankNet(
        n_cands=8, n_joints=17, n_features=3,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout_prob, lift3d_ckpt=args.lift3d_ckpt,
        var_floor=args.var_floor, joint_tokens=joint_tokens, aux_head=aux,
        film_mode=film_mode, pool_mode=pool_mode, d_joint=args.d_joint).to(device)

    train_ds = S131.LeanMultiViewDataset(args.train_data)
    val_ds = S131.LeanMultiViewDataset(args.val_data)
    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    best_val, patience_cnt, history = float('inf'), 0, []
    t0 = time.time()
    total_p = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)

    ref_dist = ref_distribution(train_k, ref_mode)
    print(f'arm={args.arm}  train_k={"U{1..8}" if train_k == 0 else train_k}  '
          f'ref_mode={ref_mode}  config={args.config}  seed={args.seed}')
    print(f'  training P(ref=position) %: ' + ' '.join(f'{v:5.2f}' for v in ref_dist))
    print(f'  eval     P(ref=position) %: ' + ' '.join(f'{12.5:5.2f}' for _ in ref_dist))
    print(f'params={total_p:,} trainable={trainable_p:,}  '
          f'lambda_err={args.lambda_err} lambda_pair={args.lambda_pair}\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for kp, scores in train_dl:
            kp_aug, ref = view_dropout(kp)
            kp_aug, scores, ref = kp_aug.to(device), scores.to(device), ref.to(device)
            optimizer.zero_grad()
            logits, delta_hat = model(kp_aug, ref, return_aux=True)
            loss, *_ = S131.compute_losses(logits, delta_hat, scores, args)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            total_loss += loss.item() * kp.size(0); total_n += kp.size(0)
        tr_loss = total_loss / total_n

        # SAME policy on val -- early stopping must not select checkpoints
        # under a different protocol than the arm trains under (see header)
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for kp, scores in val_dl:
                kp_aug, ref = view_dropout(kp)
                kp_aug, scores, ref = kp_aug.to(device), scores.to(device), ref.to(device)
                logits, delta_hat = model(kp_aug, ref, return_aux=True)
                loss, *_ = S131.compute_losses(logits, delta_hat, scores, args)
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

    # NOTE: the eval harness is stage131's, UNCHANGED and unparameterised --
    # every arm is scored single-observation over all 8 starting positions.
    # That is the deployed condition and it is the only thing being compared.
    print('\n' + '=' * 70)
    val_m = S131.evaluate_per_position(model, args.val_data, args.batch_size, device, 'val')
    test_m = S131.evaluate_per_position(model, args.test_data, args.batch_size, device, 'test(S8)')
    meta = dict(tier=f'exp157_{args.arm}', exp='exp157', arm=args.arm,
                train_k=train_k, ref_mode=ref_mode, ref_dist_train=ref_dist,
                config=args.config, joint_tokens=joint_tokens, aux_head=aux,
                film_mode=film_mode, pool_mode=pool_mode,
                total_params=total_p, trainable_params=trainable_p,
                training_time_s=time.time() - t0, seed=args.seed,
                lambda_lr=args.lambda_lr, lambda_err=args.lambda_err,
                lambda_pair=args.lambda_pair, rare_boost=args.rare_boost,
                w_dom=args.w_dom, w_runnerup=args.w_runnerup, gap_thresh=args.gap_thresh,
                best_epoch=int(ckpt['epoch']), best_val_loss=float(ckpt['val_loss']))
    val_m.update(meta); test_m.update(meta)
    json.dump(val_m, open(save_dir / 'val_metrics.json', 'w'), indent=2)
    json.dump(test_m, open(save_dir / 'test_metrics.json', 'w'), indent=2)
    print(f'\nSaved -> {save_dir}/test_metrics.json')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=list(ARMS), required=True)
    p.add_argument('--config', choices=list(CONFIGS), default='K')
    p.add_argument('--train_data', default=str(BVPNET.parents[1] / 'data' / 'multiview_split_train'))
    p.add_argument('--val_data', default=str(BVPNET.parents[1] / 'data' / 'multiview_split_val'))
    p.add_argument('--test_data', default=str(BVPNET.parents[1] / 'data' / 'multiview_split_test'))
    p.add_argument('--save_dir', required=True)
    p.add_argument('--lift3d_ckpt', default=str(BVPNET.parents[1] / 'weights' / 'lifter_bypassed' / 'lift3d_gaussian_mlp_best.pt'))
    p.add_argument('--var_floor', type=float, default=1.0)
    # loss -- stage131 defaults, unchanged (inert for config K: aux=False)
    p.add_argument('--lambda_lr', type=float, default=0.1)
    p.add_argument('--lambda_err', type=float, default=0.5)
    p.add_argument('--lambda_pair', type=float, default=0.5)
    p.add_argument('--rare_boost', type=float, default=1.0)
    p.add_argument('--w_dom', type=float, default=2.0)
    p.add_argument('--w_runnerup', type=float, default=1.0)
    p.add_argument('--gap_thresh', type=float, default=0.15)
    # optimisation -- stage131 defaults, unchanged
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--dropout_prob', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--n_layers', type=int, default=2)
    p.add_argument('--d_joint', type=int, default=32)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
