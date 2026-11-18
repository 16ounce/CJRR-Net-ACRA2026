"""Purpose: Build the reference NLF-S and RootHead state.

Main functionality: Export corrected pose means and variances as st_nlfonly.npz.

Usage:
    python -m models.state_reference.build_state
"""

"""export_nlfonly_state.py  (2026-08-12)

The NLF-ONLY state: NLF's pose, the root from stage_nlf_root_head's correction
head, and the variance from that head's own heteroscedastic output. The Gaussian
lifter is not loaded, not called, and contributes nothing. CJRR-Net is untouched
-- the state has the same (N,8,17,3) shape the hook already takes.

WHAT THIS COSTS, MEASURED BEFORE RUNNING IT. Root error on test:
    NLF raw                                  708.3 mm mean / 51.2 median
    z-range clamp alone (no training)        263.2 / 51.3
    THE HEAD (this file)                     256.1 / 50.8
    r3guard  (needs the Gaussian lifter)     183.0 / 51.5
    oracle per-sample scale                  115.9 / 14.6
The head kills the tail (708 -> 256) and leaves the median untouched, which is
the correct behaviour: the oracle log-scale has sd 0.50 and is dominated by the
~4% blow-ups, so regressing it directly wrecks the 96% (measured: a ridge fit
with test R^2 = +0.53 on that target drives the median 51.2 -> 211.6 mm).

TWO CUES THAT DID NOT WORK, recorded so they are not retried:
  - f*H_3D/h_px with h_px from NLF's OWN projection is CIRCULAR -- both come
    from the same 3D, so the ratio collapses to z_NLF (corr with log z_GT
    +0.475 vs NLF's own +0.549). Used as a second estimator it took the root
    mean 708 -> 946 and the max to 119 m.
  - the same ratio with h_px from ViTPose 2D IS independent (corr +0.517
    against NLF's +0.912 on conf>0.3) but far weaker, and swapping to it hurts
    monotonically with the firing rate (263 -> 308 -> 369 -> 461).
So there is currently NO NLF-only substitute for what the Gaussian lifter's
fallback root buys, and that gap is ~73 mm of mean root error.

PREDICTION, stated before the downstream is run: the root staircase moves norm%
clearly only below ~92 mm of mean root error (+1.64, p=0.004) and is already
n.s. at 137 mm (+0.83, p=0.16). 256 mm is above both, so this arm is expected to
be a null or to lose to `ng`. It is run anyway because the prediction should be
tested, and because the head's error DISTRIBUTION (median 50.8) differs from the
staircase's, which shrank both halves together.

Two arms, following the mean/variance methodology that has paid twice:
    st_nlfonly        head mean + head sigma      <- the real NLF-only system
    st_nlfonly_ggvar  head mean + Gaussian sigma  <- diagnostic: isolates the
                                                    head's variance (NOT
                                                    deployable as NLF-only)
"""
import importlib.util, json, os
import numpy as np, torch

BV = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .geometry import SELECTED_CAM_IDS, parse_calibration
from .nlf import load_nlf, root_of, rebase, moment_scale
from . import root_head as RH


def row_map(src_kp, dst_kp):
    """shift25 rows are a strict monotone subset of the canonical rows
    (4254/4266, 501/503, 500/502); matched by EXACT keypoint identity, and an
    unmatched row aborts rather than being interpolated."""
    idx_of = {src_kp[i].tobytes(): i for i in range(len(src_kp))}
    idx = np.array([idx_of.get(dst_kp[j].tobytes(), -1) for j in range(len(dst_kp))])
    assert (idx >= 0).all(), f'{int((idx < 0).sum())} target rows not in the source'
    assert np.all(np.diff(idx) > 0), 'row mapping is not monotone'
    return idx


def main():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal = parse_calibration(); Ks = np.stack([cal[c]['K'] for c in SELECTED_CAM_IDS])
    ck = torch.load(os.path.join(BV, 'weights', 'root_head', 'best.pt'),
                    map_location='cpu', weights_only=False)
    head = RH.RootHead(ck['d_in']).to(dev); head.load_state_dict(ck['model_state_dict']); head.eval()
    mu, sd = ck['mu'], ck['sd']

    for split in ['train', 'val', 'test']:
        nlf, unc, valid, _ = load_nlf(split); assert valid.all()
        gt = np.load(os.path.join(BV, 'data', f'multiview_split_{split}',
                                  'lift3d_target_mv.npy')).astype(np.float32)
        X, rn, rg, (N, C) = RH.features(split, Ks)
        with torch.no_grad():
            r, logv, _ = head(torch.tensor((X - mu) / sd, device=dev),
                              torch.tensor(rn, device=dev))
        r = r.cpu().numpy().reshape(N, C, 3)
        var_root = logv.exp().cpu().numpy().reshape(N, C, 1, 3)

        mean_nlfonly = rebase(nlf, r)                       # NLF shape, corrected root
        # per-joint variance = the head's root variance (shared, it IS a translation)
        # plus NLF's own per-joint sigma for the articulation, moment-matched on TRAIN
        if split == 'train':
            a_e = (mean_nlfonly - gt) - ((r - root_of(gt))[:, :, None, :])
            s_art = np.array([moment_scale(a_e[..., k], unc) for k in range(3)])
            np.save(os.path.join(BV, 'weights', 'root_head', 's_art.npy'), s_art)
        s_art = np.load(os.path.join(BV, 'weights', 'root_head', 's_art.npy'))
        var_art = (s_art[None, None, None, :] * unc[..., None]) ** 2
        var_nlfonly = (var_root + var_art).astype(np.float32)

        states = {'st_nlfonly': (mean_nlfonly, var_nlfonly)}

        e = np.linalg.norm(mean_nlfonly - gt, axis=-1)
        s = np.sqrt(var_nlfonly.sum(-1))
        print(f'[{split}] 3D {e.mean():7.1f} mm   root '
              f'{np.linalg.norm(r - root_of(gt), axis=-1).mean():6.1f}   '
              f'sigma/err {s.mean()/e.mean():.3f}  corr {np.corrcoef(s.ravel(), e.ravel())[0,1]:+.3f}')

        kp = np.load(os.path.join(BV, 'data', f'multiview_split_{split}', 'keypoints_mv.npy'))
        for tgt in ['multiview_split', 'latency_shift']:
            d = os.path.join(BV, 'data', f'{tgt}_{split}')
            if not os.path.isdir(d): continue
            idx = None if tgt == 'multiview_split' else row_map(
                kp, np.load(os.path.join(d, 'keypoints_mv.npy')))
            for k, (m, v) in states.items():
                mm, vv = (m, v) if idx is None else (m[idx], v[idx])
                np.savez_compressed(os.path.join(d, f'{k}.npz'),
                                    mean=mm.astype(np.float32), var=vv.astype(np.float32),
                                    fire=np.zeros(mm.shape[:2], bool),
                                    meta=json.dumps(dict(source='stage_nlf_root_head',
                                                         gaussian_lifter_used=(k.endswith('ggvar')))))
            print(f'    wrote st_nlfonly into {tgt}_{split}')


if __name__ == '__main__':
    main()
