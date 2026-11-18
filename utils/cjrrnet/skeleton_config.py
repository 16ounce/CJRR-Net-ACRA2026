"""Purpose: Define the COCO-17 skeleton configuration.

Main functionality: Provide joint and edge constants plus adjacency matrices.
"""

import numpy as np

N_JOINTS = 17

JOINT_NAMES = [
    'nose',           # 0
    'left_eye',       # 1
    'right_eye',      # 2
    'left_ear',       # 3
    'right_ear',      # 4
    'left_shoulder',  # 5
    'right_shoulder', # 6
    'left_elbow',     # 7
    'right_elbow',    # 8
    'left_wrist',     # 9
    'right_wrist',    # 10
    'left_hip',       # 11
    'right_hip',      # 12
    'left_knee',      # 13
    'right_knee',     # 14
    'left_ankle',     # 15
    'right_ankle',    # 16
]

# Undirected edges — exact COCO skeleton_links from mmpose/ViTPose
# Note: (3,5) and (4,6) connect ears to shoulders, keeping the graph connected.
SKELETON_EDGES = [
    (0,  1), (0,  2),    # nose → eyes
    (1,  2),             # left_eye ↔ right_eye
    (1,  3), (2,  4),    # eyes → ears
    (3,  5), (4,  6),    # ears → shoulders  (face-to-body link)
    (5,  6),             # left_shoulder ↔ right_shoulder
    (5,  7), (7,  9),    # left arm: shoulder → elbow → wrist
    (6,  8), (8, 10),    # right arm
    (5, 11), (6, 12),    # torso: shoulders → hips
    (11, 12),            # left_hip ↔ right_hip
    (11, 13), (13, 15),  # left leg: hip → knee → ankle
    (12, 14), (14, 16),  # right leg
]  # 19 undirected edges


def build_adjacency_norm(n_joints=N_JOINTS, edges=SKELETON_EDGES):
    """
    Normalized adjacency with self-loops: A_norm = D^(-1/2) (A + I) D^(-1/2)
    Returns float32 ndarray of shape (n_joints, n_joints).
    """
    A = np.zeros((n_joints, n_joints), dtype=np.float32)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0     # symmetric
    np.fill_diagonal(A, 1.0)                                    # self-loops
    D = A.sum(axis=1)
    D_inv_sqrt = np.where(D > 0, 1.0 / np.sqrt(D), 0.0)
    A_norm = A * D_inv_sqrt[:, None] * D_inv_sqrt[None, :]     # D^(-1/2) A D^(-1/2)
    return A_norm.astype(np.float32)


def build_adjacency_mask(n_joints=N_JOINTS, edges=SKELETON_EDGES):
    """
    Boolean adjacency mask with self-loops for masked graph attention.
    Returns bool ndarray of shape (n_joints, n_joints).
    """
    A = np.zeros((n_joints, n_joints), dtype=bool)
    for i, j in edges:
        A[i, j] = True
        A[j, i] = True
    np.fill_diagonal(A, True)
    return A


def print_skeleton_info():
    """Print joint names and edges — call this to verify the topology."""
    print(f'COCO {N_JOINTS}-joint order (from ViTPose keypoints[:17]):')
    for i, name in enumerate(JOINT_NAMES):
        print(f'  {i:2d}: {name}')
    print(f'\nSkeleton edges ({len(SKELETON_EDGES)} undirected, + self-loops):')
    for i, j in SKELETON_EDGES:
        print(f'  ({i:2d},{j:2d})  {JOINT_NAMES[i]} ↔ {JOINT_NAMES[j]}')
    A_norm = build_adjacency_norm()
    nonzero = int((A_norm > 0).sum())
    print(f'\nA_norm: shape={A_norm.shape}  non-zero entries={nonzero}')
    print(f'        min={A_norm[A_norm > 0].min():.4f}  '
          f'max={A_norm.max():.4f}  '
          f'mean(non-zero)={A_norm[A_norm > 0].mean():.4f}')


if __name__ == '__main__':
    print_skeleton_info()
