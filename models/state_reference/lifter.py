"""Purpose: Define the reference Gaussian pose lifter.

Main functionality: Predict joint means and variances from 2D keypoints.
"""

import torch
import torch.nn as nn


class Lift3DGaussianMLP(nn.Module):
    def __init__(self, n_joints=17, n_features=3, d_hidden=256, dropout=0.1,
                 output_scale=3000.0, min_log_var=-4.0, max_log_var=10.0):
        super().__init__()
        self.output_scale = output_scale
        self.n_joints = n_joints
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        in_dim = n_joints * n_features
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, d_hidden), nn.LayerNorm(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden // 2), nn.ReLU(),
        )
        self.mean_head    = nn.Linear(d_hidden // 2, n_joints * 3)
        self.logvar_head  = nn.Linear(d_hidden // 2, n_joints * 3)

    def forward(self, x: torch.Tensor):
        """
        x: (B, 17, 3) current-view 2D keypoints (x_norm, y_norm, conf)
        Returns:
            mean    : (B, 17, 3) camera-frame 3D pose, mm
            var     : (B, 17, 3) per-axis variance, mm^2 (in the SAME
                      camera-frame axes as mean -- axis 2 = depth/Z)
        """
        B = x.shape[0]
        h = self.trunk(x.view(B, -1))
        mean = self.mean_head(h).view(B, self.n_joints, 3) * self.output_scale
        log_var_raw = self.logvar_head(h).view(B, self.n_joints, 3)
        log_var = torch.clamp(log_var_raw, self.min_log_var, self.max_log_var)
        # variance in mm^2 -- scale log_var's effective range with output_scale^2
        # so a log_var=0 means ~1mm std at the raw head's natural scale, and the
        # clamp range comfortably covers "very confident" to "very uncertain"
        # relative to the pose's own mm scale.
        var = torch.exp(log_var) * (self.output_scale ** 2) / (10.0 ** 2)
        return mean, var


def gaussian_nll_loss(mean, var, target, eps=1e-6):
    """Per-axis heteroscedastic Gaussian NLL, averaged over joints/axes/batch.
    mean, var, target: (B, J, 3)."""
    var = var.clamp_min(eps)
    nll = 0.5 * (torch.log(var) + (target - mean) ** 2 / var)
    return nll.mean()


if __name__ == '__main__':
    model = Lift3DGaussianMLP()
    total = sum(p.numel() for p in model.parameters())
    print(f'params={total:,}')
    x = torch.randn(4, 17, 3)
    mean, var = model(x)
    assert mean.shape == (4, 17, 3) and var.shape == (4, 17, 3)
    assert (var > 0).all()
    tgt = torch.randn(4, 17, 3) * 1000
    loss = gaussian_nll_loss(mean, var, tgt)
    print(f'mean shape {mean.shape}  var range [{var.min():.2f}, {var.max():.2f}]  NLL={loss.item():.3f}')
