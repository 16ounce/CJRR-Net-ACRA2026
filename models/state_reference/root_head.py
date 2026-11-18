"""Purpose: Train the reference RootHead correction model.

Main functionality: Predict corrected root translation and uncertainty from NLF features.

Usage:
    python -m models.state_reference.root_head --help
"""

"""stage_nlf_root_head.py  (2026-08-12)

NLF-ONLY root correction head. Removes the Gaussian lifter from the pipeline
entirely: NLF supplies the pose, this head supplies the root, and the two
together produce the (mean, var) that ReprojJointFeatures already consumes.
CJRR-Net is untouched.

=== WHY NOT THE OBVIOUS PARAMETERISATION ===
Predicting a free (dx, dy, dz) lets the head move the subject off their own
image position, which is the one thing NLF gets right. Measured on test, the
NLF root error decomposed on the GT viewing ray:

    along the ray (depth/scale)   median 44.5 mm
    perpendicular (2D placement)  median 14.1 mm
    share of squared error, excluding the 3.9% blow-ups:  83% along the ray

and the single best per-sample SCALE applied to r_NLF cuts the median root
error 51.2 -> 14.6 mm. One degree of freedom. So the head predicts

    r = exp(delta) * r_NLF  +  lateral,      lateral small and explicitly
                                             regularised toward zero

exp(delta) preserves the root's 2D projection exactly, which is the correct
invariance; `lateral` exists only for the residual 14 mm the scale cannot reach
and is kept from absorbing depth error by an L2 penalty.

=== WHY A HEAD ALONE IS NOT ENOUGH, AND WHAT REPLACES THE GAUSSIAN LIFTER ===
The ORACLE per-sample scale still leaves a MEAN root error of 115.9 mm (median
14.6) because the ~4% of frames where NLF's global depth blows up are not a
scale error at all. The root staircase says norm% only moves clearly below
~92 mm of MEAN root error, so a scale head that ignores the tail cannot get
there no matter how well it is trained.

Detecting those blow-ups is what the Gaussian lifter was doing (the vfusion
disagreement term, which is 49% of that arm's variance). Its replacement here is
NLF-only and geometric:

    z_cue = f * H_3D / h_px

NLF's own metric root->head length H_3D divided by that segment's projected
pixel length. It is independent of NLF's TRANSLATION -- shape and projection
only -- and on train corr(log z_cue, log z_GT) = +0.475 against NLF's own
+0.549, i.e. comparably informative. The head therefore
  (a) receives log(z_cue) - log(z_NLF) as its single most important feature, and
  (b) outputs a heteroscedastic sigma, so the state carries its own uncertainty
      instead of borrowing the Gaussian lifter's.

=== INPUTS (all NLF-only; no Gaussian lifter, no ViTPose) ===
    log z_NLF, log z_cue, and their difference          the depth evidence
    root's normalised image position (u, v)             where the person is
    log H_3D                                            metric body scale
    NLF uncertainty: mean and max over joints           NLF's own confidence
    fx, fy normalised; camera one-hot (8)               the rig

=== LOSS ===
    Huber on log-depth  +  Huber on lateral (normalised)  +  Gaussian NLL for
    sigma  +  L2 on the lateral head.
Deliberately NOT a ViTPose reprojection loss: design doc sec. 1.3 measured that
solving the root against noisy ViTPose 2D gives 809.8 mm even from a PERFECT
pose, so reprojection is the wrong supervision for this quantity.

    python stage_nlf_root_head.py --seed 0
"""
import argparse, importlib.util, json, os
import numpy as np, torch, torch.nn as nn

from .nlf import load_nlf, root_of, rebase

BV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .geometry import SELECTED_CAM_IDS, parse_calibration

HEAD_J, IMG = 0, 2048.0          # COCO 0 is the head (collapsed), root = hip midpoint


def features(split, Ks):
    """-> X (N*8, F), r_nlf (N*8,3), r_gt (N*8,3). NLF-only, no labels in X."""
    nlf, unc, _, _ = load_nlf(split)
    gt = np.load(os.path.join(BV, 'data', f'multiview_split_{split}',
                              'lift3d_target_mv.npy')).astype(np.float32)
    rn, rg = root_of(nlf), root_of(gt)
    fx = Ks[None, :, 0, 0]; fy = Ks[None, :, 1, 1]
    cx = Ks[None, :, 0, 2]; cy = Ks[None, :, 1, 2]
    proj = lambda P: np.stack([P[..., 0] / np.clip(P[..., 2], 1e-3, None) * fx + cx,
                               P[..., 1] / np.clip(P[..., 2], 1e-3, None) * fy + cy], -1)
    H3 = np.linalg.norm(nlf[..., HEAD_J, :] - rn, axis=-1)
    hpx = np.linalg.norm(proj(nlf[..., HEAD_J, :]) - proj(rn), axis=-1)
    z_cue = fx * H3 / np.maximum(hpx, 1e-3)
    uv = proj(rn)
    N, C = rn.shape[:2]
    cam = np.eye(C, dtype=np.float32)[None].repeat(N, 0)
    X = np.concatenate([
        np.log(np.maximum(rn[..., 2], 1.0))[..., None],
        np.log(np.maximum(z_cue, 1.0))[..., None],
        (np.log(np.maximum(z_cue, 1.0)) - np.log(np.maximum(rn[..., 2], 1.0)))[..., None],
        (uv / IMG - 0.5),
        np.log(np.maximum(H3, 1.0))[..., None],
        unc.mean(-1)[..., None], unc.max(-1)[..., None],
        (fx / 1500.0).repeat(N, 0)[..., None], (fy / 1500.0).repeat(N, 0)[..., None],
        cam], -1).astype(np.float32)
    return X.reshape(-1, X.shape[-1]), rn.reshape(-1, 3), rg.reshape(-1, 3), (N, C)


class RootHead(nn.Module):
    """delta (log depth scale), lateral (2, normalised), log-variance (3)."""
    def __init__(self, d_in, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.delta = nn.Linear(hidden, 1)
        self.lateral = nn.Linear(hidden, 2)
        self.logvar = nn.Linear(hidden, 3)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)
        nn.init.zeros_(self.lateral.weight); nn.init.zeros_(self.lateral.bias)

    def forward(self, x, r_nlf):
        h = self.body(x)
        d = self.delta(h).clamp(-1.5, 1.5)                 # scale within [0.22, 4.5]
        lat = self.lateral(h).clamp(-1.0, 1.0) * 200.0     # mm, bounded
        r = torch.exp(d) * r_nlf
        r = torch.cat([r[:, :1] + lat[:, :1], r[:, 1:2] + lat[:, 1:2], r[:, 2:]], -1)
        return r, self.logvar(h).clamp(-4.0, 14.0), lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lam_lat', type=float, default=1e-4)
    # --- 2026-08-12: the two fixes for "the head only damps the tail" ---
    # THE GRADIENT ARITHMETIC. Typical log-depth residual is 0.016; blow-ups are
    # 2-3. With Huber delta=1.0 the typical sample sits in the QUADRATIC region
    # (grad ~0.016) and the blow-up in the LINEAR one (grad 1.0), so the 4% of
    # blow-ups carry ~10^4x the gradient of the 96%. The scale head therefore
    # never learns the typical correction -- measured: root median 51.2 -> 50.8
    # while the ORACLE per-sample scale reaches 14.6.
    #   --huber_delta   shrink it so typical residuals are linear too, making
    #                   the per-sample gradient ratio ~1:1 instead of 10^4:1
    #   --gate_train    drop the blow-ups from the SCALE loss entirely, using
    #                   NLF's own sigma as the gate (AUC 0.9969 for detecting
    #                   them). They keep contributing to the NLL so the variance
    #                   still learns to flag them.
    ap.add_argument('--huber_delta', type=float, default=1.0)
    ap.add_argument('--gate_train', type=float, default=0.0,
                    help='TRAIN percentile of mean NLF sigma above which a sample is '
                         'excluded from the scale/lateral loss; 0 disables')
    ap.add_argument('--save_dir', default=os.path.join(BV, 'runs', 'root_head'))
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal = parse_calibration(); Ks = np.stack([cal[c]['K'] for c in SELECTED_CAM_IDS])

    data = {sp: features(sp, Ks) for sp in ['train', 'val', 'test']}
    # the gate: NLF's own mean sigma, thresholded at a TRAIN percentile
    import importlib.util as _iu
    _s = _iu.spec_from_file_location('r3b', os.path.join(BV, 'stage_r3_nlf_hybrid_lifter.py'))
    _r3 = _iu.module_from_spec(_s); _s.loader.exec_module(_r3)
    UB = {sp: _r3.load_nlf(sp)[1].mean(-1).reshape(-1) for sp in ['train', 'val', 'test']}
    THR = float(np.percentile(UB['train'], a.gate_train)) if a.gate_train > 0 else np.inf
    CLEAN = {sp: (UB[sp] <= THR) for sp in UB}
    print(f'gate: sigma <= {THR:.4f} keeps {CLEAN["train"].mean()*100:.1f}% of train '
          f'for the scale loss   (huber delta {a.huber_delta})')
    Xtr, rn_tr, rg_tr, _ = data['train']
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6            # TRAIN-only standardisation
    T = lambda A: torch.tensor(A, device=dev)
    model = RootHead(Xtr.shape[1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    hub = nn.HuberLoss(delta=a.huber_delta)
    hub_none = nn.HuberLoss(delta=a.huber_delta, reduction='none')

    def batches(sp, bs=4096, shuffle=False):
        X, rn, rg, _ = data[sp]
        idx = np.random.permutation(len(X)) if shuffle else np.arange(len(X))
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            yield T((X[j] - mu) / sd), T(rn[j]), T(rg[j]), T(CLEAN[sp][j].astype(np.float32))

    def evaluate(sp):
        model.eval(); errs = []
        with torch.no_grad():
            for x, rn, rg, _ in batches(sp):
                r, _, _ = model(x, rn)
                errs.append(torch.norm(r - rg, dim=-1).cpu().numpy())
        e = np.concatenate(errs)
        return float(e.mean()), float(np.median(e))

    best = (1e9, None)
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, rn, rg, clean in batches('train', shuffle=True):
            opt.zero_grad()
            r, logv, lat = model(x, rn)
            # depth in log units, lateral in metres -- comparable magnitudes.
            # `clean` is 1 on samples the gate keeps; the SCALE loss sees only
            # those, the NLL sees everything so the variance still learns the
            # blow-ups it must flag.
            w = clean / clean.mean().clamp_min(1e-6)
            l_z = (hub_none(torch.log(r[:, 2].clamp_min(1.0)),
                            torch.log(rg[:, 2].clamp_min(1.0))) * w).mean()
            l_xy = (hub_none(r[:, :2] / 1000.0, rg[:, :2] / 1000.0).mean(-1) * w).mean()
            nll = (0.5 * (logv + (r - rg) ** 2 / logv.exp())).mean()
            loss = l_z + l_xy + 0.05 * nll + a.lam_lat * (lat ** 2).mean()
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        sch.step()
        vm, vmed = evaluate('val')
        if vm < best[0]:
            best = (vm, {k: v.clone() for k, v in model.state_dict().items()})
        if ep % 10 == 0 or ep == 1:
            print(f'[{ep:>3}/{a.epochs}] val root mean {vm:7.1f} med {vmed:6.1f} mm')

    model.load_state_dict(best[1])
    os.makedirs(a.save_dir, exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(), mu=mu, sd=sd,
                    d_in=int(Xtr.shape[1]), args=vars(a)),
               os.path.join(a.save_dir, 'best.pt'))
    print('\n            root error (mm)      NLF raw        head')
    out = {}
    for sp in ['train', 'val', 'test']:
        _, rn, rg, _ = data[sp]
        raw = np.linalg.norm(rn - rg, axis=-1)
        m, md = evaluate(sp)
        out[sp] = dict(raw_mean=float(raw.mean()), raw_med=float(np.median(raw)),
                       head_mean=m, head_med=md)
        print(f'  {sp:<6} mean {raw.mean():8.1f} -> {m:7.1f}   median {np.median(raw):6.1f} -> {md:6.1f}')
    json.dump(out, open(os.path.join(a.save_dir, 'root_metrics.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
