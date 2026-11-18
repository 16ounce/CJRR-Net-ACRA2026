"""Purpose: Train paper-configuration CJRR-Net.

Main functionality: Apply the state-aware objective and evaluate every starting camera.

Usage:
    python models/cjrrnet/train.py --help
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
from torch.utils.data import DataLoader

BVPNET = Path(__file__).resolve().parent
import sys
sys.path.insert(0, str(BVPNET.parents[1]))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


S131 = _load('stage131', BVPNET / 'training_core.py')   # losses, data, eval
S143 = _load('stage143', BVPNET / 'view_dropout.py')  # dropout policy
CONTROLS = _load('nlfhead_controls', BVPNET.parents[1] / 'baselines' / 'shared_input_mlp' / 'model.py')
CONFIGS = {**S131.CONFIGS, **CONTROLS.CONFIGS}

# px. (random_px - oracle_px) on the TRAIN split -- the exact denominator norm%
# is measured against. Train-split statistic only; see header.
REGRET_SCALE = 174.818

# arm -> (dropout arm in stage143.ARMS, lambda_regret, lambda_pair)
ARMS = {
    'k1_full':   ('k1',         0.5, 0.2),   # the requested experiment
    'rand_full': ('rand_first', 0.5, 0.2),   # same loss, shipped dropout protocol
    'k1_regret': ('k1',         0.5, 0.0),   # regret term alone
    'k1_pair':   ('k1',         0.0, 0.2),   # pair term alone
    # ---- ADDED after the first four landed (pbs/186) ----
    # rand_full beat the shipped exp142_K by +1.05pp (p=0.0053, 12/15 seeds),
    # but the term split was only run at K=1, where the whole objective is null
    # (regret +0.40 n.s., pair -0.60 p=0.026 HARMFUL). That split cannot explain
    # a +1.05pp gain, so the attribution has to be redone at the dropout
    # protocol where the gain actually appears. Without these two arms the
    # headline result is unexplained and not writable.
    'rand_regret': ('rand_first', 0.5, 0.0),  # regret term alone, shipped dropout
    'rand_pair':   ('rand_first', 0.0, 0.2),  # pair term alone, shipped dropout
}


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
    loss_lr = S131.lambda_rank_loss(logits, norm_mpjpe)
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
        W = S131.build_pair_weights(mpjpe, args.gap_thresh, args.w_dom,
                                    args.w_runnerup, args.rare_boost)
        loss_pair = S131.pairwise_loss(logits, W)
        total = total + args.lambda_pair * loss_pair
    return total, loss_ce.detach(), loss_lr.detach(), loss_regret.detach(), loss_pair.detach()


# ---------------------------------------------------------------------------
# EXTERNAL LIFTER STATE (--state_npz), added 2026-08-12 for the R3 / NLF arm.
# OPT-IN AND DEFAULT OFF: with the flag absent every call site below is the
# expression it has always been, so this file's existing results are unaffected
# and the control for any --state_npz run comes out of this same script (the
# rule PBS 319 was written around -- reprojection_visibility_study.md records a
# -0.43 pp offset between two implementations of the same model).
#
# The state is a frozen per-(row, camera) (mean, var) that replaces what
# ReprojJointFeatures' own internal lifter would have produced, through the
# `external_state=` hook exp162 added. Built by export_r3_guard_state.py.
# ---------------------------------------------------------------------------
class StateMultiViewDataset(S131.LeanMultiViewDataset):
    """LeanMultiViewDataset + the per-camera lifter state for that row."""

    def __init__(self, data_dir, npz_name='r3_guard_state.npz'):
        super().__init__(data_dir)
        z = np.load(Path(data_dir) / npz_name)
        self.st_mean = z['mean'].astype(np.float32)          # (N,8,17,3)
        self.st_var = z['var'].astype(np.float32)
        assert len(self.st_mean) == len(self.kp_mv), (
            f'state has {len(self.st_mean)} rows, split has {len(self.kp_mv)} -- '
            'wrong split or a stale export')

    def __getitem__(self, idx):
        kp, sc = super().__getitem__(idx)
        return kp, sc, torch.from_numpy(self.st_mean[idx]), torch.from_numpy(self.st_var[idx])


def pick_state(st_mean, st_var, ref):
    """(B,8,17,3) x (B,) -> the CURRENT camera's state, (B,17,3)."""
    b = torch.arange(st_mean.size(0), device=st_mean.device)
    return st_mean[b, ref], st_var[b, ref]


@torch.no_grad()
def evaluate_per_position_state(model, data_dir, batch_size, device, split_name,
                                npz_name='r3_guard_state.npz'):
    """S131.evaluate_per_position with the external state. Every line that
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

    return S131.evaluate_per_position(_Wrapped(), data_dir, batch_size, device, split_name)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    dropout_arm, lam_regret, lam_pair = ARMS[args.arm]
    if args.lambda_regret is None:
        args.lambda_regret = lam_regret
    if args.lambda_pair is None:
        args.lambda_pair = lam_pair
    train_k, ref_mode = S143.ARMS[dropout_arm]
    view_dropout = S143.make_view_dropout(train_k, ref_mode)

    # regret normalisation. Default = the headline split's constant, so the
    # exp158 runs stay bit-reproducible. --regret_scale -1 refits from THIS
    # run's train split, which is mandatory for LOSO folds (see fit_regret_scale).
    if args.regret_scale is None:
        args.regret_scale_used = REGRET_SCALE
        scale_src = 'module constant (headline split)'
    elif args.regret_scale < 0:
        args.regret_scale_used = fit_regret_scale(args.train_data)
        scale_src = f'refit from {args.train_data}'
    else:
        args.regret_scale_used = float(args.regret_scale)
        scale_src = 'given on the command line'

    joint_tokens, aux, film_mode, pool_mode = CONFIGS[args.config]
    assert not aux, (f'config {args.config} has aux_head=True; exp158 varies the '
                     'objective with the architecture held at the shipped config K')

    # --camera_geom: swaps MPI's own camera.calibration-derived geometry for
    # an externally-supplied rig (2026-08-08, AIST++ transfer-learning "Branch
    # 3" -- training 142k158 from scratch on AIST++'s own 8-camera geometry
    # so the resulting checkpoint has identical shapes to an MPI-trained one,
    # including baseline_matrix and the per-candidate query table, and can
    # --init_ckpt warm-start MPI training wholesale, not just the lifter).
    # None (every existing call site) leaves cam_ids=None -> MPI's own
    # calibration, byte-identical to before this flag existed.
    camera_geom_kwargs = {}
    n_cands = 8
    if args.camera_geom:
        geom = np.load(args.camera_geom, allow_pickle=True)
        n_cands = geom['Rs'].shape[0]
        camera_geom_kwargs = dict(
            cam_ids=None, cam_Rs=geom['Rs'], cam_Ts=geom['Ts'], cam_Ks=geom['Ks'],
            cam_angles=geom['angles'], baseline_matrix=geom['baseline'])

    lifter_calib_kwargs = {}
    if args.lifter_calib:
        c = np.load(args.lifter_calib)
        lifter_calib_kwargs = dict(lifter_calib_scale=c['scale'], lifter_calib_offset=c['offset'])

    if args.config in CONTROLS.CONFIGS:
        from functools import partial
        model_class = partial(CONTROLS.build_control, args.config)
    else:
        model_class = _load('exp142', BVPNET / 'cjrrnet.py').JointTokenReprojVisRankNet
    model = model_class(
        n_cands=n_cands, n_joints=args.n_joints, n_features=args.n_features,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        dropout=args.dropout_prob, lift3d_ckpt=args.lift3d_ckpt,
        var_floor=args.var_floor, joint_tokens=joint_tokens, aux_head=aux,
        film_mode=film_mode, pool_mode=pool_mode, d_joint=args.d_joint,
        **camera_geom_kwargs, **lifter_calib_kwargs,
        # 2026-07-28: stage131 has always forwarded these two, stage144 did not.
        # Any config whose definition lives in ZERO_CHANNELS/EXTRA_KWARGS rather
        # than in the (joint_tokens, aux, film, pool) tuple -- K_semgcn,
        # K_geoquery, K_slim, K_meanq, ... -- was therefore built as PLAIN
        # config K here, silently, with no error and a plausible result. Only
        # 'K' and the tuple-defined arms were ever run through this script, so
        # no published exp158 number is affected, but the landmine is removed
        # rather than documented.
        zero_channels=S131.ZERO_CHANNELS.get(args.config),
        norm_path=(args.norm_path or None),
        **S131.EXTRA_KWARGS.get(args.config, {})).to(device)

    # 2026-08-05: optional warm start from an already-trained checkpoint (the
    # fine-tune-from-canonical experiment). Loaded AFTER construction so the
    # model's random init still consumes the usual RNG draws -- only the
    # weights are overwritten, seeding/reproducibility of everything built
    # after this point is unaffected. Requires n_features to match (asserted
    # by load_state_dict's own shape check on cam_encoder.0.weight).
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device)
        state = ck['model_state_dict']
        # 2026-08-08: exclude camera-GEOMETRY buffers (baseline_matrix,
        # all_deltas, reproj.cam_R/T/K) from the loaded state. These are
        # deterministic recomputations from cam_ids/cam_angles/camera_geom at
        # THIS construction, not learned parameters -- every prior use of
        # --init_ckpt was MPI->MPI (identical geometry both sides) so loading
        # them was a no-op; the AIST++ transfer-learning line's
        # --camera_geom introduces checkpoints whose geometry buffers belong
        # to a DIFFERENT physical rig, and blindly loading them would silently
        # overwrite this run's correct geometry with the source run's. Always
        # filtered, not just for cross-geometry loads, since it is a no-op
        # exactly when it doesn't matter and a correctness fix exactly when
        # it does.
        geom_keys = {'baseline_matrix', 'all_deltas',
                     'reproj.cam_R', 'reproj.cam_T', 'reproj.cam_K'}
        dropped = [k for k in state if k in geom_keys]
        state = {k: v for k, v in state.items() if k not in geom_keys}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, f'unexpected keys: {unexpected}'
        assert set(missing) <= geom_keys, f'unexpectedly missing keys: {set(missing) - geom_keys}'
        print(f'warm-started from {args.init_ckpt} (epoch {ck.get("epoch")}); '
              f'kept this run\'s own geometry for {sorted(dropped)}')

    use_state = bool(args.state_npz)
    if use_state:
        train_ds = StateMultiViewDataset(args.train_data, args.state_npz)
        val_ds = StateMultiViewDataset(args.val_data, args.state_npz)
        print(f'external lifter state: {args.state_npz} '
              f'({json.loads(str(np.load(Path(args.train_data) / args.state_npz)["meta"]))})')
    else:
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

    print(f'arm={args.arm}  config={args.config}  seed={args.seed}')
    print(f'  dropout: {dropout_arm}  train_k={"U{1..8}" if train_k == 0 else train_k}  ref={ref_mode}')
    print(f'  L = 1.0*CE + {args.lambda_lr}*LambdaRank + {args.lambda_regret}*regret'
          f' + {args.lambda_pair}*pair')
    print(f'  REGRET_SCALE = {args.regret_scale_used:.4f} px  [{scale_src}]')
    print(f'params={total_p:,} trainable={trainable_p:,}\n')

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums, total_n = np.zeros(5), 0
        for batch in train_dl:
            kp, scores = batch[0], batch[1]
            kp_aug, ref = view_dropout(kp)
            kp_aug, scores, ref = kp_aug.to(device), scores.to(device), ref.to(device)
            optimizer.zero_grad()
            if use_state:
                m, v = pick_state(batch[2].to(device), batch[3].to(device), ref)
                logits = model(kp_aug, ref, external_state=(m, v))
            else:
                logits = model(kp_aug, ref)
            loss, l_ce, l_lr, l_rg, l_pr = compute_losses(logits, scores, args)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            n = kp.size(0); total_n += n
            sums += np.array([loss.item(), l_ce.item(), l_lr.item(), l_rg.item(), l_pr.item()]) * n
        tr_loss, tr_ce, tr_lr, tr_rg, tr_pr = sums / total_n

        # SAME dropout policy on val -- early stopping must not select
        # checkpoints under a different protocol than the arm trains under
        model.eval()
        val_loss_sum, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_dl:
                kp, scores = batch[0], batch[1]
                kp_aug, ref = view_dropout(kp)
                kp_aug, scores, ref = kp_aug.to(device), scores.to(device), ref.to(device)
                if use_state:
                    m, v = pick_state(batch[2].to(device), batch[3].to(device), ref)
                    logits = model(kp_aug, ref, external_state=(m, v))
                else:
                    logits = model(kp_aug, ref)
                loss, *_ = compute_losses(logits, scores, args)
                val_loss_sum += loss.item() * kp.size(0); val_n += kp.size(0)
        val_loss = val_loss_sum / val_n
        scheduler.step(val_loss)
        history.append(dict(epoch=epoch, train_loss=float(tr_loss), val_loss=val_loss,
                            ce=float(tr_ce), lambdarank=float(tr_lr),
                            regret=float(tr_rg), pair=float(tr_pr)))
        if epoch % 10 == 0 or epoch == 1:
            print(f'[{epoch:>3}/{args.epochs}] train={tr_loss:.4f}  val={val_loss:.4f}   '
                  f'(ce={tr_ce:.4f} lr={tr_lr:.4f} regret={tr_rg:.4f} pair={tr_pr:.4f})')

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

    # UNCHANGED single-observation, 8-starting-position harness -- identical to
    # exp157 and to every row of docs/final_table.txt. Only the objective moved.
    print('\n' + '=' * 70)
    _eval = ((lambda mo, d, bs, dv, nm: evaluate_per_position_state(mo, d, bs, dv, nm, args.state_npz))
             if use_state else S131.evaluate_per_position)
    val_m = _eval(model, args.val_data, args.batch_size, device, 'val')
    test_m = _eval(model, args.test_data, args.batch_size, device, 'test(S8)')
    meta = dict(tier=f'exp158_{args.arm}', exp='exp158', arm=args.arm,
                dropout_arm=dropout_arm, train_k=train_k, ref_mode=ref_mode,
                config=args.config, joint_tokens=joint_tokens, aux_head=aux,
                film_mode=film_mode, pool_mode=pool_mode,
                lambda_lr=args.lambda_lr, lambda_regret=args.lambda_regret,
                lambda_pair=args.lambda_pair, regret_scale=args.regret_scale_used,
                regret_scale_source=scale_src,
                rare_boost=args.rare_boost, w_dom=args.w_dom,
                w_runnerup=args.w_runnerup, gap_thresh=args.gap_thresh,
                total_params=total_p, trainable_params=trainable_p,
                training_time_s=time.time() - t0, seed=args.seed,
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
    p.add_argument('--norm_path', default='',
                    help='reprojection-feature norm stats json; empty = canonical '
                         'docs/reproj_feature_norm.json (fit on the ViTPose-Small '
                         'lifter). Pass a matching fit when --lift3d_ckpt is not the '
                         'canonical one -- see fit_reproj_norm_camset.py / pbs/182.')
    p.add_argument('--lifter_calib', default=None,
                    help='.npz with scale,offset (3,) each -- fixed per-axis '
                         'affine correction applied to the lifter\'s output '
                         'position before reprojection (2026-08-08, fixes a '
                         'domain-shifted lifter\'s systematic depth bias, e.g. '
                         'the AIST++-trained lifter overestimating depth on '
                         'MPI by +960mm). None (default): identity, unchanged.')
    p.add_argument('--camera_geom', default=None,
                    help='path to a build_aist_camera_geometry.py .npz '
                         '(Rs,Ts,Ks,angles,baseline) to train against an '
                         'external camera rig instead of MPI\'s own '
                         'calibration; also sets n_cands from Rs.shape[0].')
    p.add_argument('--n_joints', type=int, default=17,
                    help='2D input joint count; 13 selects the nose-only-head '
                         'skeleton (skeleton_config_13j) via ReprojJointFeatures '
                         'auto-resolution. Must match --train/val/test_data and '
                         '--lift3d_ckpt.')
    p.add_argument('--var_floor', type=float, default=1.0)
    # objective -- CE is fixed at 1.0 by construction
    p.add_argument('--lambda_lr', type=float, default=0.1)
    p.add_argument('--lambda_regret', type=float, default=None, help='default: from --arm')
    p.add_argument('--lambda_pair', type=float, default=None, help='default: from --arm')
    p.add_argument('--regret_scale', type=float, default=None,
                   help='px. Omit for the headline-split constant 174.818; pass -1 to '
                        'REFIT from this run\'s train split (mandatory for LOSO folds).')
    # camera-pair ranking shape -- stage131 defaults, unchanged
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
    p.add_argument('--init_ckpt', default=None,
                    help='2026-08-05: warm-start from this checkpoint (e.g. the '
                         'canonical t-predicts-t exp158_rand_full best.pt) before '
                         'training on --train_data. None = train from scratch.')
    p.add_argument('--n_features', type=int, default=3,
                    help='2026-08-05: raw per-joint channel count fed to cam_encoder '
                         '(x,y,conf=3; +vel_x,vel_y=5 for the latency-shift velocity arm). '
                         'The frozen reprojection lifter always gets only the first 3 '
                         '(model forward() slices), so this is additive-only.')
    p.add_argument('--state_npz', default='',
                   help='filename (inside each --*_data dir) of a precomputed '
                        'lifter state to feed through external_state=, e.g. '
                        'r3_guard_state.npz. Empty (default) = use the frozen '
                        'lifter inside ReprojJointFeatures, i.e. every result '
                        'this script produced before 2026-08-12.')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
