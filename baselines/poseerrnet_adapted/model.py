"""Purpose: Define the adapted PoseErrNet camera scorer.

Main functionality: Normalize body keypoints, encode relative camera geometry, and predict view errors.
"""

import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn

# This file lives in /srv/scratch/z5511538/baseline/ but the camera-rig
# calibration it needs is bvpnet's. Path is explicit, not relative, so the
# move out of bvpnet/model_variants/ cannot silently break it.
BVPNET = Path(__file__).resolve().parents[2] / 'utils' / 'cjrrnet'
sys.path.insert(0, str(BVPNET))
from mpi_camera_geometry import build_camera_constants, BASELINE_NORM  # noqa: E402

# COCO-17 order, from skeleton_config.JOINT_NAMES
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
TORSO_IDX = [L_HIP, R_HIP, L_SHOULDER, R_SHOULDER]

# Same rig constants every geometry experiment in this line uses.
CAM_ANGLES_DEG = np.array([
    [  -5.999342, -24.232821],
    [ -36.733883, -24.627060],
    [-120.892868, -28.016209],
    [ -48.914967, -18.251417],
    [ 110.447212, -12.449697],
    [-169.133163, -37.388592],
    [ 153.881378, -18.736523],
    [  91.824913, -21.445538],
], dtype=np.float32)

HEIGHT_NORM = 2000.0   # mm -- camera-centre Y spans 823..2471 mm, so pairwise
                       # height deltas live in about +-1650 mm.


def build_relative_geometry() -> np.ndarray:
    """(8, 8, 3) -- geo[cur, cand] = [delta_azimuth, delta_height, distance].

    delta_azimuth : wrapped to (-180, 180], scaled by 1/180 -> [-1, 1]
    delta_height  : (Y_cand - Y_cur) / HEIGHT_NORM, world frame, Y is up
                    (verified: camera-centre Y = 1535/1994/1491/1123/823/
                    2471/1166/1388 mm, the only axis with that range)
    distance      : ||C_cand - C_cur|| / BASELINE_NORM, i.e. exp115's baseline
                    length, kept on the same 8000 mm scale it uses.

    All three are fixed rig calibration constants -- no privileged per-frame
    information, same clean/deployable status as exp102's az/el delta.
    """
    centers, _, baseline = build_camera_constants()
    n = len(CAM_ANGLES_DEG)
    geo = np.zeros((n, n, 3), dtype=np.float32)
    for cur in range(n):
        d_az = CAM_ANGLES_DEG[:, 0] - CAM_ANGLES_DEG[cur, 0]
        d_az = (d_az + 180.0) % 360.0 - 180.0
        geo[cur, :, 0] = d_az / 180.0
        geo[cur, :, 1] = (centers[:, 1] - centers[cur, 1]) / HEIGHT_NORM
        geo[cur, :, 2] = baseline[cur] / BASELINE_NORM
    return geo


def poseerr_normalize(xy: torch.Tensor, eps: float = 1e-6,
                      rot_sign: str = 'fixed') -> torch.Tensor:
    """PoseErrNet's centre / rotation / scale normalisation, batched.

    Follows poseerrnet/code/train_poseerrnet.py:normalize_keypoints (lines
    107-124):
        1. subtract the mean of {both hips, both shoulders}
        2. rotate so the torso vector sv = shoulder_mid - hip_mid lands on +Y
        3. divide by |sv|

    That reference runs on raw pixels; this runs on `xy / 2048 - 0.5`. Steps 1
    and 2 are translation/rotation so the 2048 factor passes straight through,
    and step 3 divides it out -- the two are identical up to the -0.5 offset,
    which step 1 also removes. So no re-extraction of the dataset is needed.

    ROTATION SIGN -- the reference has a bug, so this is a knob, not a
    constant. With a = atan2(sv_x, sv_y) the reference applies R(-a) (its
    `c, s = cos(-a), sin(-a)`). Since a = pi/2 - psi for the standard polar
    angle psi, R(-a) sends psi -> 2*psi - pi/2: it DOUBLES the tilt instead of
    removing it, and maps psi and psi+pi to the same output, so an upside-down
    pose is indistinguishable from an upright one. R(+a) is the
    canonicalisation that was intended -- it sends every torso vector to +Y
    exactly. Verified in the self-test below: R(+a) is invariant to an
    arbitrary input rotation to 8e-06, R(-a) is off by 2.6e+01.

        rot_sign='fixed'      R(+a), the correct canonicalisation  [default]
        rot_sign='reference'  R(-a), bug included -- used ONLY by the
                              `poseerr` arm, whose entire purpose is a
                              faithful reproduction of the published baseline.
                              Quoting that arm as "PoseErrNet" is only honest
                              if it runs the code PoseErrNet actually runs.

    The reference's own renders are near upright (psi ~ pi/2, so a ~ 0) which
    is why the sign error stays small there. `uav_fixed8` vs `poseerr` is not a
    clean measurement of the bug -- they also differ in the confidence channel
    -- so do not read it as one.

    Args:
        xy: (..., 17, 2)
    Returns:
        (..., 17, 2) body-centric coordinates. Rows whose torso length is
        degenerate (all-zero input, i.e. a view masked out by view dropout)
        come back as exact zeros rather than exploding through the division.
    """
    hip = xy[..., [L_HIP, R_HIP], :].mean(-2)              # (..., 2)
    sh  = xy[..., [L_SHOULDER, R_SHOULDER], :].mean(-2)    # (..., 2)
    center = xy[..., TORSO_IDX, :].mean(-2)                # (..., 2)

    out = xy - center.unsqueeze(-2)                        # (..., 17, 2)

    sv = sh - hip                                          # (..., 2)
    a = torch.atan2(sv[..., 0], sv[..., 1])                # angle of sv from +Y
    if rot_sign == 'reference':
        a = -a
    elif rot_sign != 'fixed':
        raise ValueError(f'rot_sign must be "fixed" or "reference", got {rot_sign!r}')
    c, s = torch.cos(a).unsqueeze(-1), torch.sin(a).unsqueeze(-1)
    x, y = out[..., 0], out[..., 1]
    out = torch.stack([c * x - s * y, s * x + c * y], dim=-1)

    sl = torch.linalg.norm(sv, dim=-1)                     # rotation is an isometry,
    valid = (sl > eps)                                     # so |sv| is unchanged by it
    out = out / sl.clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
    return out * valid.unsqueeze(-1).unsqueeze(-1)


class UAVGeoMLP(nn.Module):
    """Single-view pose latent + per-candidate relative geometry -> 8 MPJPE.

    pose_norm : 'body'  PoseErrNet centre/rotation/scale normalisation  [new]
                'image' the project-standard xy/2048-0.5, passed through
    head      : 'shared_geo' one decoder applied to concat(latent, geo_rel),
                             8 times, no per-camera parameter anywhere   [new]
                'fixed8'     exp21's Linear(... -> 8), depth-matched
    use_conf  : True  17x(x, y, conf) = 51-d, this project's standard input
                False 17x(x, y)       = 34-d, what PoseErrNet actually takes.
                      Its normalize_keypoints() returns coords only -- the
                      detector's confidence never enters the published model.
    rot_sign  : passed to poseerr_normalize; 'reference' reproduces the
                published rotation bug, 'fixed' corrects it.
    """

    def __init__(self, n_cands=8, n_joints=17, n_features=3, d_hidden=256,
                 d_latent=128, d_dec=128, dropout=0.1,
                 pose_norm='body', head='shared_geo', use_conf=True,
                 rot_sign='fixed'):
        super().__init__()
        assert pose_norm in ('body', 'image')
        assert head in ('shared_geo', 'fixed8')
        self.pose_norm, self.head_mode = pose_norm, head
        self.use_conf, self.rot_sign = use_conf, rot_sign
        self.n_cands = n_cands

        in_dim = n_joints * (n_features if use_conf else 2)
        self.encoder = nn.Sequential(                    # 4 layers, exp21's shape
            nn.Linear(in_dim, d_hidden), nn.LayerNorm(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_latent), nn.ReLU(),
            nn.Linear(d_latent, d_latent), nn.ReLU(),
        )

        self.register_buffer('geo', torch.from_numpy(build_relative_geometry()))  # (8,8,3)

        dec_in = d_latent + (3 if head == 'shared_geo' else 0)
        dec_out = 1 if head == 'shared_geo' else n_cands
        self.decoder = nn.Sequential(                    # 4 layers, both heads
            nn.Linear(dec_in, d_dec), nn.LayerNorm(d_dec), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_dec, d_dec), nn.LayerNorm(d_dec), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_dec, d_dec // 2), nn.ReLU(),
            nn.Linear(d_dec // 2, dec_out),
        )

    def encode(self, x: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """x: (B, N, J, F) multi-view tensor, cur: (B,) -> (B, d_latent).

        Only x[b, cur[b]] is ever read. Every other view is discarded here, by
        design -- this is a single-observation model. One consequence worth
        stating because exp160 already relied on it: view dropout of the other
        columns is a literal no-op for this architecture.
        """
        B = x.size(0)
        kp = x[torch.arange(B, device=x.device), cur]        # (B, 17, F)
        xy = poseerr_normalize(kp[..., :2], rot_sign=self.rot_sign) \
            if self.pose_norm == 'body' else kp[..., :2]
        kp = torch.cat([xy, kp[..., 2:]], dim=-1) if self.use_conf else xy
        return self.encoder(kp.flatten(1))

    def predict_error(self, x: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """(B, n_cands) predicted per-camera normalised MPJPE. Lower = better."""
        z = self.encode(x, cur)                              # (B, D)
        if self.head_mode == 'fixed8':
            return self.decoder(z)
        g = self.geo[cur]                                    # (B, n_cands, 3)
        zc = torch.cat([z.unsqueeze(1).expand(-1, self.n_cands, -1), g], dim=-1)
        return self.decoder(zc).squeeze(-1)                  # (B, n_cands)

    def forward(self, x: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
        """Returns NEGATED predicted error, so argmax == argmin. See header."""
        return -self.predict_error(x, cur)


if __name__ == '__main__':
    torch.manual_seed(0)

    geo = build_relative_geometry()
    print('relative geometry (8,8,3):')
    print(f'  d_az    range {geo[..., 0].min():+.3f} .. {geo[..., 0].max():+.3f}')
    print(f'  d_height range {geo[..., 1].min():+.3f} .. {geo[..., 1].max():+.3f}')
    print(f'  distance range {geo[..., 2].min():+.3f} .. {geo[..., 2].max():+.3f}')
    assert np.allclose(np.diagonal(geo, axis1=0, axis2=1).T, 0.0), 'self-pair must be zero'

    # --- poseerr_normalize invariance checks -------------------------------
    kp = torch.randn(6, 17, 2) * 0.2
    base = poseerr_normalize(kp)
    shifted = poseerr_normalize(kp + torch.tensor([0.13, -0.07]))
    scaled = poseerr_normalize(kp * 2.5)
    th = 0.7
    R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    rotated = poseerr_normalize(kp @ R.T)
    for name, out in [('translation', shifted), ('scale', scaled), ('rotation', rotated)]:
        d = (out - base).abs().max().item()
        print(f'  invariant to {name:12s} max|diff| = {d:.2e}')
        assert d < 1e-4, name
    zero = poseerr_normalize(torch.zeros(2, 17, 2))
    assert torch.all(zero == 0), 'degenerate (masked) view must stay zero'
    print('  degenerate view -> exact zeros  OK')

    # the torso must actually land on +Y (this is what the reference's R(-a)
    # fails to do -- see the poseerr_normalize docstring)
    torso = base[:, [L_SHOULDER, R_SHOULDER], :].mean(1) - base[:, [L_HIP, R_HIP], :].mean(1)
    assert torch.allclose(torso, torch.tensor([0.0, 1.0]).expand_as(torso), atol=1e-5), torso
    print('  canonical torso vector == (0, 1)  OK')

    # the reference sign must NOT be rotation-invariant -- that is the bug
    ref_rot = poseerr_normalize(kp @ R.T, rot_sign='reference')
    ref_base = poseerr_normalize(kp, rot_sign='reference')
    print(f'  reference sign, rotation error = {(ref_rot - ref_base).abs().max():.2e}'
          '  (expected large -- this is the published bug, reproduced on purpose)')
    assert (ref_rot - ref_base).abs().max() > 1.0

    # --- model ------------------------------------------------------------
    for head in ('shared_geo', 'fixed8'):
        for pn in ('body', 'image'):
            m = UAVGeoMLP(pose_norm=pn, head=head)
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            x = torch.randn(4, 8, 17, 3)
            cur = torch.tensor([0, 3, 5, 7])
            out = m(x, cur)
            assert out.shape == (4, 8) and not torch.isnan(out).any()
            print(f'  {head:11s} {pn:5s} conf=Y  trainable={n:,}  out={tuple(out.shape)}')

    # faithful PoseErrNet: xy only, reference rotation, fixed 8-way head
    m = UAVGeoMLP(pose_norm='body', head='fixed8', use_conf=False, rot_sign='reference')
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    out = m(torch.randn(4, 8, 17, 3), torch.tensor([0, 3, 5, 7]))
    assert out.shape == (4, 8) and not torch.isnan(out).any()
    print(f'  {"fixed8":11s} {"body":5s} conf=N  trainable={n:,}  out={tuple(out.shape)}'
          '   [poseerr arm]')
    assert m.encoder[0].in_features == 34, m.encoder[0].in_features

    # --- only the current view may reach the output -----------------------
    m = UAVGeoMLP().eval()
    x = torch.randn(4, 8, 17, 3)
    cur = torch.tensor([0, 3, 5, 7])
    with torch.no_grad():
        a = m(x, cur)
        x2 = x.clone()
        for b, c in enumerate(cur.tolist()):
            other = [i for i in range(8) if i != c]
            x2[b, other] = torch.randn(7, 17, 3)
        bb = m(x2, cur)
    print(f'  scramble non-cur views  max|diff| = {(a - bb).abs().max():.2e}')
    assert torch.allclose(a, bb, atol=1e-6)

    # --- the shared decoder has no per-camera parameter -------------------
    m = UAVGeoMLP(head='shared_geo')
    names = [n for n, _ in m.named_parameters()]
    assert not any('cam' in n or 'quer' in n for n in names), names
    print('  shared_geo carries no per-camera parameter  OK')
    print('\nOK')
