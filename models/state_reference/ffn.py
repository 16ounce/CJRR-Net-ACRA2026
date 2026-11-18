"""Purpose: Provide reference ranker feed-forward layers.

Main functionality: Define the ranking block and pairwise loss used by the reference network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class _FFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        return self.net(x)


class MultiViewRankNet(nn.Module):
    """
    Multi-view quality ranking via cross-attention.

    Predicts a quality score for each of n_cands candidate camera slots.
    Each slot has its own learnable query token that attends to all N input cameras.

    Args:
        n_input_cams : N — number of input cameras (72 teacher, 8 student)
        n_cands      : number of candidate camera slots to score (8)
        n_joints     : 17 (COCO)
        n_features   : 3  (x, y, conf)
        d_model      : transformer width
        n_heads      : attention heads
        n_layers     : cross-attention depth (3 teacher, 2 student)
    """

    def __init__(
        self,
        n_input_cams: int   = 72,
        n_cands:      int   = 8,
        n_joints:     int   = 17,
        n_features:   int   = 3,
        d_model:      int   = 128,
        n_heads:      int   = 4,
        n_layers:     int   = 3,
        dropout:      float = 0.1,
    ):
        super().__init__()
        in_dim = n_joints * n_features   # 51

        # Shared per-camera encoder (same architecture as exp42b)
        # cam_encoder converts each camera’s raw 51-dimensional keypoint input into a d_model-dimensional feature token that can be used by the attention layers.
        self.cam_encoder = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU(),
        )

        # n_cands independent learnable query tokens (one per camera slot)
        self.queries = nn.Parameter(torch.randn(1, n_cands, d_model) * 0.02)

        # Cross-attention blocks: all n_cands queries attend to all N camera tokens
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn = nn.ModuleList([_FFN(d_model, dropout) for _ in range(n_layers)])

        self.norm1 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        # Score head: d_model → scalar per slot
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B = batch size
        # N = number of input cameras
        # J = number of joints
        # F = number of features for each joint
        """
        Args:
            x : (B, N_cams, 17, 3)
        Returns:
            (B, n_cands)  quality scores — higher means better camera
        """
        B, N, J, F = x.shape
        tokens = self.cam_encoder(x.view(B * N, J * F)).view(B, N, -1)  # (B, N, D)
        q = self.queries.expand(B, -1, -1)                                # (B, n_cands, D)
        for cross, ffn, n1, n2 in zip(self.cross_attn, self.ffn, self.norm1, self.norm2):
            attn_out, _ = cross(query=q, key=tokens, value=tokens)
            q = n1(q + attn_out)
            q = n2(q + ffn(q))
        return self.score_head(q).squeeze(-1)   # (B, n_cands)


def pairwise_ranking_loss(
    scores:       torch.Tensor,
    mpjpe_labels: torch.Tensor,
    margin:       float = 0.1,
) -> torch.Tensor:
    """
    Hinge ranking loss over all C(n_cands, 2) pairs.

    For every pair (i, j) where mpjpe_i < mpjpe_j (camera i is better),
    penalise when score_i is not at least `margin` above score_j:
        loss += max(0, margin - (score_i - score_j))

    Args:
        scores       : (B, C)  predicted quality (higher = better)
        mpjpe_labels : (B, C)  MPJPE per candidate in pixels (lower = better);
                               NaN entries are skipped
        margin       : hinge margin (default 0.1)

    Returns:
        scalar loss averaged over all valid pairs and frames
    """
    B, C = scores.shape
    pair_losses = []
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            valid   = ~(torch.isnan(mpjpe_labels[:, i]) | torch.isnan(mpjpe_labels[:, j]))
            if valid.sum() == 0:
                continue
            better  = valid & (mpjpe_labels[:, i] < mpjpe_labels[:, j])
            if better.sum() == 0:
                continue
            hinge = torch.clamp(margin - (scores[better, i] - scores[better, j]), min=0.0)
            pair_losses.append(hinge.mean())
    if not pair_losses:
        return scores.sum() * 0.0   # zero with gradient
    return torch.stack(pair_losses).mean()


if __name__ == '__main__':
    teacher = MultiViewRankNet(n_input_cams=72, n_cands=8, n_layers=3)
    student = MultiViewRankNet(n_input_cams=8,  n_cands=8, n_layers=2)
    pt = sum(p.numel() for p in teacher.parameters())
    ps = sum(p.numel() for p in student.parameters())
    print(f'Teacher params : {pt:,}')
    print(f'Student params : {ps:,}')

    x72 = torch.randn(4, 72, 17, 3)
    x8  = torch.randn(4, 8,  17, 3)
    st  = teacher(x72)
    ss  = student(x8)
    print(f'Teacher output : {st.shape}  {st.detach().numpy().round(3)}')
    print(f'Student output : {ss.shape}  {ss.detach().numpy().round(3)}')

    mpjpe_fake = torch.rand(4, 8) * 50
    loss = pairwise_ranking_loss(st, mpjpe_fake)
    print(f'Ranking loss   : {loss.item():.4f}')
    loss.backward()
    print('Backward OK')
