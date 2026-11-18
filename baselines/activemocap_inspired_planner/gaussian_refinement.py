"""Purpose: Refine the static Gaussian pose estimate.

Main functionality: Use projection Jacobians and a bone-length prior for pose refinement.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
import sys                                                        # noqa: E402
sys.path.insert(0, str(_HERE.parents[1] / 'utils' / 'activemocap'))
from skeleton_config import N_JOINTS, SKELETON_EDGES              # noqa: E402

IMAGE_SIZE = 2048.0
BONE_PRIOR_PATH = _HERE.parents[1] / 'config' / 'mpi_inf_3dhp' / 'bone_length_prior.json'

# ---------------------------------------------------------------------------
# defaults -- every one is a config knob, none is a hidden magic constant
# ---------------------------------------------------------------------------
PRIOR_VAR_FLOOR_MM2 = 1.0        # clamp(var, min=...) as specified
PRIOR_VAR_CEIL_MM2 = 1.0e7       # std 3.2 m. The lifter's max_log_var=10 maps
                                 # to var 2e9 mm^2 (std 45 m); left unclamped
                                 # the prior precision underflows to 5e-10 and
                                 # the Hessian loses rank in float32.
SIGMA_PIX = 20.0                 # px, base 2D measurement noise
CONF_FLOOR = 0.05                # confidence below this is treated as this
HUBER_DELTA = 5.0                # in WHITENED units (i.e. 5 * s_j px)
Z_MIN_MM = 200.0                 # joints nearer than this to the current
                                 # camera get zero reprojection weight
SIGMA_BONE_FLOOR_MM = 10.0       # matches bone_length_prior.json


def pinhole_project(X, fx, fy, cx, cy, z_min=1e-3):
    """X: (..., 3) camera-frame mm -> (uv (..., 2) px, Z (...) mm).

    Identical formula to ReprojJointFeatures (x = fx*Xc/Z + cx). The shared
    calibration itself comes from that module's buffers; only the arithmetic is
    re-expressed here so it can be differentiated analytically, and the test
    suite checks it against the extractor numerically.
    """
    Z = X[..., 2].clamp_min(z_min)
    u = fx * X[..., 0] / Z + cx
    v = fy * X[..., 1] / Z + cy
    return torch.stack([u, v], dim=-1), Z


def pinhole_jacobian(X, fx, fy, cx, cy, z_min=1e-3):
    """d(u,v)/d(X,Y,Z) of the pinhole above.  X: (...,3) -> (...,2,3).

        du/dX = fx/Z   du/dY = 0      du/dZ = -fx*X/Z^2
        dv/dX = 0      dv/dY = fy/Z   dv/dZ = -fy*Y/Z^2
    """
    Z = X[..., 2].clamp_min(z_min)
    zero = torch.zeros_like(Z)
    return torch.stack([
        torch.stack([fx / Z, zero, -fx * X[..., 0] / Z ** 2], dim=-1),
        torch.stack([zero, fy / Z, -fy * X[..., 1] / Z ** 2], dim=-1),
    ], dim=-2)                                                    # (...,2,3)


def load_bone_prior(path=BONE_PRIOR_PATH, edges=SKELETON_EDGES,
                    sigma_floor=SIGMA_BONE_FLOOR_MM):
    """TRAIN-split-only per-edge length prior, in SKELETON_EDGES order.

    Returns (edge_i, edge_j, median, lo, hi, sigma) numpy arrays.

    NOTE ON THE 5 FACE EDGES. COCO17_TO_MPI28 maps nose/eyes/ears all to MPI's
    single 'head' joint, so in lift3d_target_mv the 5 face joints are the SAME
    3D point and edges (0,1) (0,2) (1,2) (1,3) (2,4) have GT length identically
    0. They are kept -- mpjpe_per_joint_cam is scored against exactly those
    coincident targets, so pulling them together is right for the metric -- but
    their sigma is floored at sigma_floor so the constraint stays finite. They
    WILL fight the reprojection term, whose ViTPose observations put
    nose/eyes/ears at genuinely different pixels. That fight is a property of
    the label, not of this optimiser.
    """
    d = json.load(open(path))
    by_pair = {(e['i'], e['j']): e for e in d['edges']}
    med, lo, hi, sig, ei, ej, deg = [], [], [], [], [], [], []
    for (i, j) in edges:
        e = by_pair[(i, j)]
        ei.append(i); ej.append(j)
        med.append(e['median']); lo.append(e['lo']); hi.append(e['hi'])
        sig.append(max(e['std'], sigma_floor))
        deg.append(bool(e['degenerate']))
    return (np.asarray(ei, dtype=np.int64), np.asarray(ej, dtype=np.int64),
            np.asarray(med, dtype=np.float32), np.asarray(lo, dtype=np.float32),
            np.asarray(hi, dtype=np.float32), np.asarray(sig, dtype=np.float32),
            np.asarray(deg, dtype=bool))


class StaticGaussianRefiner(nn.Module):
    """Static (single-frame, no history, no motion model) 2D-3D refinement.

    forward(mu_lift, var_lift, cur_kp, K_cur) -> dict with
        X_refined     (B, J, 3)      current-camera frame, mm
        Lambda        (B, 3J, 3J)    posterior INFORMATION matrix (= H + ridge)
        Sigma_current (B, 3J, 3J)    its inverse, via Cholesky solve
        chol          (B, 3J, 3J)    lower Cholesky factor of Lambda

    Lambda is returned alongside Sigma precisely so the planner never inverts
    anything: information gain is logdet(Lambda_post) - logdet(Lambda), both
    read off Cholesky factors.
    """

    def __init__(self, n_joints=N_JOINTS, edges=SKELETON_EDGES,
                 use_reproj=True, use_bone=True, n_iters=8,
                 lambda_lift=1.0, lambda_reproj=1.0, lambda_bone=4.0,
                 sigma_pix=SIGMA_PIX, conf_floor=CONF_FLOOR,
                 huber_delta=HUBER_DELTA, z_min_mm=Z_MIN_MM,
                 prior_var_floor=PRIOR_VAR_FLOOR_MM2,
                 prior_var_ceil=PRIOR_VAR_CEIL_MM2,
                 damping_init=1e-3, damping_min=1e-8, damping_max=1e6,
                 ridge_rel=1e-6, ridge_abs=1e-9,
                 skip_degenerate_bones=False,
                 bone_prior_path=BONE_PRIOR_PATH):
        super().__init__()
        self.J = n_joints
        self.D = 3 * n_joints
        self.use_reproj = use_reproj
        self.use_bone = use_bone
        self.n_iters = n_iters
        self.lambda_lift = lambda_lift
        self.lambda_reproj = lambda_reproj
        self.lambda_bone = lambda_bone
        self.sigma_pix = sigma_pix
        self.conf_floor = conf_floor
        self.huber_delta = huber_delta
        self.z_min_mm = z_min_mm
        self.prior_var_floor = prior_var_floor
        self.prior_var_ceil = prior_var_ceil
        self.damping_init = damping_init
        self.damping_min = damping_min
        self.damping_max = damping_max
        self.ridge_rel = ridge_rel
        self.ridge_abs = ridge_abs

        ei, ej, med, lo, hi, sig, deg = load_bone_prior(bone_prior_path, edges)
        # The 5 face edges have GT length 0 (COCO17_TO_MPI28 collapses
        # nose/eyes/ears onto MPI 'head'). Keeping them is correct for the
        # METRIC but fights ViTPose, which sees 5 distinct pixels; this switch
        # exposes both readings instead of silently picking one.
        if skip_degenerate_bones:
            sig = np.where(deg, 1e9, sig).astype(np.float32)
        self.skip_degenerate_bones = skip_degenerate_bones
        self.E = len(ei)
        self.register_buffer('edge_i', torch.from_numpy(ei))
        self.register_buffer('edge_j', torch.from_numpy(ej))
        self.register_buffer('bone_med', torch.from_numpy(med))
        self.register_buffer('bone_lo', torch.from_numpy(lo))
        self.register_buffer('bone_hi', torch.from_numpy(hi))
        self.register_buffer('bone_sigma', torch.from_numpy(sig))
        self.register_buffer('bone_degenerate', torch.from_numpy(deg))
        self.register_buffer('eye_D', torch.eye(self.D))

    # -- pieces ------------------------------------------------------------

    def reference_bone_lengths(self, mu_lift):
        """(B,E) mm. Per-frame from the Gaussian MEAN, clamped into the
        TRAIN-split [p1, p99] band so a blown-up lift cannot invent a 2 m
        forearm. No temporal calibration anywhere."""
        d = mu_lift[:, self.edge_i] - mu_lift[:, self.edge_j]          # (B,E,3)
        L = torch.nan_to_num(d.norm(dim=-1), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.max(torch.min(L, self.bone_hi), self.bone_lo)

    def _prior_terms(self, X, mu, prec_sqrt):
        """Whitened prior residual; its Jacobian is diag(prec_sqrt)."""
        return (X - mu) * prec_sqrt

    def _reproj_terms(self, X, obs_px, w_pix, fx, fy, cx, cy):
        """Whitened reprojection residual + Jacobian + true robust cost.

        w_pix : (B,J) = 1/s_j, confidence already folded in.
        r (B,J,2) and Jr (B,J,2,3) are BOTH pre-multiplied by
        sqrt(huber_weight) * w_pix so the caller just forms J^T J / J^T r.
        """
        uv, Z = pinhole_project(X, fx, fy, cx, cy)
        e = (uv - obs_px) * w_pix.unsqueeze(-1)                        # (B,J,2)
        t = e.norm(dim=-1).clamp_min(1e-12)                            # (B,J)
        d = self.huber_delta
        omega = torch.where(t <= d, torch.ones_like(t), d / t)         # IRLS
        rho = torch.where(t <= d, 0.5 * t ** 2, d * (t - 0.5 * d))     # true cost
        valid = (Z > self.z_min_mm).to(X.dtype)   # behind the camera -> no vote
        omega = omega * valid
        rho = rho * valid
        Jr = pinhole_jacobian(X, fx, fy, cx, cy) * w_pix[..., None, None]
        s = omega.sqrt().unsqueeze(-1)                                 # (B,J,1)
        return e * s, Jr * s.unsqueeze(-1), rho

    def _bone_terms(self, X, Lref):
        """r (B,E) whitened, Jb (B,E,3J) dense (E=19, D=51 -> tiny)."""
        d = X[:, self.edge_i] - X[:, self.edge_j]                      # (B,E,3)
        L = d.norm(dim=-1).clamp_min(1e-6)                             # (B,E)
        inv_sig = (1.0 / self.bone_sigma).unsqueeze(0)                 # (1,E)
        r = (L - Lref) * inv_sig
        u = d / L.unsqueeze(-1) * inv_sig.unsqueeze(-1)                # (B,E,3)
        B = X.size(0)
        Jb = X.new_zeros(B, self.E, self.J, 3)
        ar = torch.arange(self.E, device=X.device)
        Jb[:, ar, self.edge_i, :] = u
        Jb[:, ar, self.edge_j, :] = -u
        return r, Jb.reshape(B, self.E, self.D)

    def _energy(self, X, mu, prec_sqrt, obs_px, w_pix, fx, fy, cx, cy, Lref):
        """TRUE (robust) energy, used for LM accept/reject."""
        rp = self._prior_terms(X, mu, prec_sqrt)
        E = self.lambda_lift * 0.5 * (rp ** 2).flatten(1).sum(1)
        if self.use_reproj:
            _, _, rho = self._reproj_terms(X, obs_px, w_pix, fx, fy, cx, cy)
            E = E + self.lambda_reproj * rho.sum(1)
        if self.use_bone:
            rb, _ = self._bone_terms(X, Lref)
            E = E + self.lambda_bone * 0.5 * (rb ** 2).sum(1)
        return E                                                       # (B,)

    def _normal_equations(self, X, mu, prec_sqrt, obs_px, w_pix,
                          fx, fy, cx, cy, Lref):
        """Gauss-Newton H (B,D,D) and gradient g (B,D)."""
        B, D = X.size(0), self.D
        rp = self._prior_terms(X, mu, prec_sqrt)                       # (B,J,3)
        H = torch.diag_embed((self.lambda_lift * prec_sqrt ** 2).reshape(B, D))
        g = (self.lambda_lift * prec_sqrt * rp).reshape(B, D)

        if self.use_reproj:
            rr, Jr, _ = self._reproj_terms(X, obs_px, w_pix, fx, fy, cx, cy)
            Jfull = X.new_zeros(B, self.J, 2, self.J, 3)   # block diagonal
            ar = torch.arange(self.J, device=X.device)
            Jfull[:, ar, :, ar, :] = Jr.permute(1, 0, 2, 3)
            Jfull = Jfull.reshape(B, 2 * self.J, D)
            H = H + self.lambda_reproj * (Jfull.transpose(1, 2) @ Jfull)
            g = g + self.lambda_reproj * (
                Jfull.transpose(1, 2) @ rr.reshape(B, -1, 1)).squeeze(-1)

        if self.use_bone:
            rb, Jb = self._bone_terms(X, Lref)
            H = H + self.lambda_bone * (Jb.transpose(1, 2) @ Jb)
            g = g + self.lambda_bone * (
                Jb.transpose(1, 2) @ rb.unsqueeze(-1)).squeeze(-1)

        return 0.5 * (H + H.transpose(1, 2)), g       # exact symmetry

    # -- main --------------------------------------------------------------

    def forward(self, mu_lift, var_lift, cur_kp, K_cur):
        """
        Args:
            mu_lift  : (B, J, 3) mm, CURRENT camera frame
            var_lift : (B, J, 3) mm^2, per-axis, same frame
            cur_kp   : (B, J, 3) observed (x_norm, y_norm, conf)
            K_cur    : (B, 3, 3) the CURRENT camera's intrinsics
        """
        B, J = mu_lift.shape[0], self.J
        dev, dt = mu_lift.device, mu_lift.dtype

        mu = torch.nan_to_num(mu_lift, nan=0.0, posinf=0.0, neginf=0.0)
        var = torch.nan_to_num(var_lift, nan=self.prior_var_ceil,
                               posinf=self.prior_var_ceil,
                               neginf=self.prior_var_floor)
        var = var.clamp(self.prior_var_floor, self.prior_var_ceil)
        prec_sqrt = var.rsqrt()                                        # (B,J,3)

        kp = torch.nan_to_num(cur_kp, nan=0.0)
        obs_px = (kp[..., :2] + 0.5) * IMAGE_SIZE                      # (B,J,2)
        conf = kp[..., 2].clamp(self.conf_floor, 1.0)
        w_pix = conf.sqrt() / self.sigma_pix                           # (B,J)

        fx = K_cur[:, 0, 0].view(B, 1); fy = K_cur[:, 1, 1].view(B, 1)
        cx = K_cur[:, 0, 2].view(B, 1); cy = K_cur[:, 1, 2].view(B, 1)

        Lref = self.reference_bone_lengths(mu) if self.use_bone \
            else torch.zeros(B, self.E, device=dev, dtype=dt)

        X = mu.clone()
        px_before = self._mean_px_residual(X, obs_px, conf, fx, fy, cx, cy)

        if self.use_reproj or self.use_bone:
            lm = torch.full((B,), self.damping_init, device=dev, dtype=dt)
            Ecur = self._energy(X, mu, prec_sqrt, obs_px, w_pix,
                                fx, fy, cx, cy, Lref)
            for _ in range(self.n_iters):
                H, g = self._normal_equations(X, mu, prec_sqrt, obs_px, w_pix,
                                              fx, fy, cx, cy, Lref)
                A = H + lm.view(B, 1, 1) * self.eye_D
                Lc, info = torch.linalg.cholesky_ex(A)
                ok = (info == 0) & torch.isfinite(Lc).flatten(1).all(1)
                Lc = torch.where(ok.view(B, 1, 1), Lc, self.eye_D.expand(B, -1, -1))
                delta = torch.cholesky_solve((-g).unsqueeze(-1), Lc).squeeze(-1)
                delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
                Xn = X + delta.view(B, J, 3)
                En = self._energy(Xn, mu, prec_sqrt, obs_px, w_pix,
                                  fx, fy, cx, cy, Lref)
                acc = (ok & torch.isfinite(En) & (En < Ecur)
                       & torch.isfinite(Xn).flatten(1).all(1))
                X = torch.where(acc.view(B, 1, 1), Xn, X)
                Ecur = torch.where(acc, En, Ecur)
                lm = torch.where(acc, (lm * 0.5).clamp_min(self.damping_min),
                                 (lm * 4.0).clamp_max(self.damping_max))

        px_after = self._mean_px_residual(X, obs_px, conf, fx, fy, cx, cy)

        # ---- Laplace posterior at the optimum ------------------------------
        H, _ = self._normal_equations(X, mu, prec_sqrt, obs_px, w_pix,
                                      fx, fy, cx, cy, Lref)
        ridge = (self.ridge_rel * H.diagonal(dim1=1, dim2=2).mean(1).clamp_min(0.0)
                 + self.ridge_abs)                                     # (B,)
        Lambda, chol = self._spd(H, ridge)
        Sigma = torch.cholesky_solve(self.eye_D.expand(B, -1, -1), chol)
        Sigma = 0.5 * (Sigma + Sigma.transpose(1, 2))
        logdet = 2.0 * torch.log(chol.diagonal(dim1=1, dim2=2)).sum(1)

        return dict(X_refined=X, Lambda=Lambda, Sigma_current=Sigma, chol=chol,
                    logdet_prior=logdet, reproj_px_before=px_before,
                    reproj_px_after=px_after, bone_ref=Lref,
                    prior_var=var, prior_mu=mu)

    def _spd(self, H, ridge):
        """Cholesky with an escalating ridge -- never a bare inverse, never a
        silent NaN. Escalates 6 times, then falls back to the diagonal, which is
        strictly > 0 by construction (the prior precision is floored by
        prior_var_ceil)."""
        B = H.size(0)
        A = H + ridge.view(B, 1, 1) * self.eye_D
        for _ in range(6):
            Lc, info = torch.linalg.cholesky_ex(A)
            bad = (info != 0) | ~torch.isfinite(Lc).flatten(1).all(1)
            if not bool(bad.any()):
                return A, Lc
            bump = A.diagonal(dim1=1, dim2=2).abs().amax(1) * 1e-4 + 1e-9
            A = torch.where(bad.view(B, 1, 1),
                            A + bump.view(B, 1, 1) * self.eye_D, A)
        Adiag = torch.diag_embed(A.diagonal(dim1=1, dim2=2).clamp_min(1e-9))
        Lc, info = torch.linalg.cholesky_ex(A)
        bad = (info != 0) | ~torch.isfinite(Lc).flatten(1).all(1)
        A = torch.where(bad.view(B, 1, 1), Adiag, A)
        return A, torch.linalg.cholesky(A)

    @torch.no_grad()
    def _mean_px_residual(self, X, obs_px, conf, fx, fy, cx, cy):
        """Confidence-weighted mean pixel reprojection error (diagnostic)."""
        uv, Z = pinhole_project(X, fx, fy, cx, cy)
        e = (uv - obs_px).norm(dim=-1)                                 # (B,J)
        w = conf * (Z > self.z_min_mm).to(X.dtype)
        return (e * w).sum(1) / w.sum(1).clamp_min(1e-6)


if __name__ == '__main__':
    torch.manual_seed(0)
    ref = StaticGaussianRefiner()
    B = 5
    mu = torch.randn(B, 17, 3) * 200 + torch.tensor([0., 0., 3400.])
    var = torch.rand(B, 17, 3) * 1e5 + 1e3
    kp = torch.rand(B, 17, 3) - 0.5
    kp[..., 2] = torch.rand(B, 17)
    K = torch.tensor([[1500., 0., 1024.], [0., 1500., 1024.],
                      [0., 0., 1.]]).expand(B, 3, 3)
    out = ref(mu, var, kp, K)
    print('X', tuple(out['X_refined'].shape), 'Sigma', tuple(out['Sigma_current'].shape))
    print('px before %.1f -> after %.1f' % (out['reproj_px_before'].mean(),
                                            out['reproj_px_after'].mean()))
    ev = torch.linalg.eigvalsh(out['Sigma_current'])
    print('Sigma min eig %.3e  asym %.2e' % (
        ev.min(), (out['Sigma_current'] - out['Sigma_current'].transpose(1, 2)).abs().max()))
