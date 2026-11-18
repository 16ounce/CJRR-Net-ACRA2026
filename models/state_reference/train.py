"""Purpose: Train the reference state-aware ranker.

Main functionality: Use precomputed pose state and evaluate camera selection.

Usage:
    python -m models.state_reference.train --help
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import (LeanMultiViewDataset, StateMultiViewDataset, pick_state,
                   make_view_dropout)
from .evaluate import evaluate_per_position, evaluate_per_position_state
from .losses import (REGRET_SCALE, compute_losses, fit_regret_scale,
                     normalized_regret_loss, lambda_rank_loss,
                     build_pair_weights, pairwise_loss)
from .network import JointTokenReprojVisRankNet

BVPNET = Path(__file__).resolve().parents[2]

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


ZERO_CHANNELS = {
    'K_noconf': [11],           # drop cur_conf
    'K_noprob': [5, 6],         # control: drop p_in_frame + p_self_occ, the two
                                # channels exp132's scalar multiplies cur_conf
                                # BY. If noconf is null and noprob is not, the
                                # interaction story survives with cur_conf as
                                # the removable half.
}


EXTRA_KWARGS = {
    'K_geoquery': dict(learned_query=False),
    'K_lite':     dict(learned_query=False, view_mask=True),
}


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



# stage143 ARMS: dropout arm -> (train_k, ref_mode). Renamed because stage144
# has an ARMS of its own; the two used to be told apart only by module
# qualifier, which merging them into one file removes.
DROPOUT_ARMS = {
    'rand_first': (0, 'first'),    # shipped exp142_K protocol (reproduction control)
    'rand_uni':   (0, 'uniform'),  # random count, uniform starting position
    'k1':         (1, 'uniform'),  # deployment-matched (first==uniform at K=1)
    'k2':         (2, 'uniform'),
    'k4':         (4, 'uniform'),
    'k8':         (8, 'uniform'),  # all views always -- the mismatch floor
}


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(args.save_dir); save_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    dropout_arm, lam_regret, lam_pair = ARMS[args.arm]
    if args.lambda_regret is None:
        args.lambda_regret = lam_regret
    if args.lambda_pair is None:
        args.lambda_pair = lam_pair
    train_k, ref_mode = DROPOUT_ARMS[dropout_arm]
    view_dropout = make_view_dropout(train_k, ref_mode)

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

    model = JointTokenReprojVisRankNet(
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
        zero_channels=ZERO_CHANNELS.get(args.config),
        norm_path=(args.norm_path or None),
        **EXTRA_KWARGS.get(args.config, {})).to(device)

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
        train_ds = LeanMultiViewDataset(args.train_data)
        val_ds = LeanMultiViewDataset(args.val_data)
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
             if use_state else evaluate_per_position)
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
    p.add_argument('--train_data', default=str(BVPNET / 'data' / 'multiview_split_train'))
    p.add_argument('--val_data', default=str(BVPNET / 'data' / 'multiview_split_val'))
    p.add_argument('--test_data', default=str(BVPNET / 'data' / 'multiview_split_test'))
    p.add_argument('--save_dir', required=True)
    p.add_argument('--lift3d_ckpt', default=str(BVPNET / 'weights' / 'lifter_bypassed' / 'lift3d_gaussian_mlp_best.pt'))
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
