"""Purpose: Provide reference camera and skeleton geometry.

Main functionality: Parse calibration and define joint mappings and adjacency matrices.
"""

import re
from pathlib import Path

import numpy as np

CALIB_PATH = Path('/srv/scratch/z5511538/nbvnet/data/mpi_inf_3dhp/data/S1/Seq1/camera.calibration')
# Standalone/exported copies of this tree ship camera.calibration next to this
# file. The dataset path above still wins when it exists, so in-place behaviour
# on Katana is unchanged; this only rescues a package moved off the scratch fs.
if not CALIB_PATH.exists():
    _local = Path(__file__).resolve().parents[2] / 'config' / 'mpi_inf_3dhp' / 'camera.calibration'
    if _local.exists():
        CALIB_PATH = _local
SELECTED_CAM_IDS = [0, 1, 2, 4, 5, 6, 7, 8]

# MPI-INF-3DHP 28-joint 'all' joint set (util/mpii_get_joint_set.m, 0-indexed here)
MPI28_JOINT_NAMES = [
    'spine3', 'spine4', 'spine2', 'spine', 'pelvis',                              # 0-4
    'neck', 'head', 'head_top',                                                    # 5-7
    'left_clavicle', 'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand',    # 8-12
    'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', 'right_hand', # 13-17
    'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe',                # 18-22
    'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe',           # 23-27
]
PELVIS_IDX = MPI28_JOINT_NAMES.index('pelvis')   # 4
NECK_IDX   = MPI28_JOINT_NAMES.index('neck')      # 5

# COCO-17 (skeleton_config.py order) -> MPI-28 index. MPI has no eyes/ears/nose
# (only head/head_top), so all 5 COCO face joints map to MPI's 'head' as the
# closest available occlusion/position proxy -- documented approximation, not
# a biomechanical claim.
COCO17_TO_MPI28 = [
    MPI28_JOINT_NAMES.index('head'),           # 0  nose      -> head (approx)
    MPI28_JOINT_NAMES.index('head'),           # 1  left_eye  -> head (approx)
    MPI28_JOINT_NAMES.index('head'),           # 2  right_eye -> head (approx)
    MPI28_JOINT_NAMES.index('head'),           # 3  left_ear  -> head (approx)
    MPI28_JOINT_NAMES.index('head'),           # 4  right_ear -> head (approx)
    MPI28_JOINT_NAMES.index('left_shoulder'),  # 5
    MPI28_JOINT_NAMES.index('right_shoulder'), # 6
    MPI28_JOINT_NAMES.index('left_elbow'),     # 7
    MPI28_JOINT_NAMES.index('right_elbow'),    # 8
    MPI28_JOINT_NAMES.index('left_wrist'),     # 9
    MPI28_JOINT_NAMES.index('right_wrist'),    # 10
    MPI28_JOINT_NAMES.index('left_hip'),       # 11
    MPI28_JOINT_NAMES.index('right_hip'),      # 12
    MPI28_JOINT_NAMES.index('left_knee'),      # 13
    MPI28_JOINT_NAMES.index('right_knee'),     # 14
    MPI28_JOINT_NAMES.index('left_ankle'),     # 15
    MPI28_JOINT_NAMES.index('right_ankle'),    # 16
]


def parse_calibration(path=CALIB_PATH):
    """Returns dict[cam_id] -> {'K': (3,3), 'R': (3,3), 'T': (3,)}"""
    cams, cur = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('name'):
            cur = int(line.split()[1])
            cams[cur] = {}
        elif line.startswith('intrinsic') and cur is not None:
            vals = [float(x) for x in line.split()[1:]]
            cams[cur]['K'] = np.array(vals, dtype=np.float64).reshape(4, 4)[:3, :3]
        elif line.startswith('extrinsic') and cur is not None:
            vals = [float(x) for x in line.split()[1:]]
            E = np.array(vals, dtype=np.float64).reshape(4, 4)
            cams[cur]['R'] = E[:3, :3]
            cams[cur]['T'] = E[:3, 3]
    return cams


def build_camera_constants(cam_ids=SELECTED_CAM_IDS):
    """
    Returns (all in the SELECTED_CAM_IDS order, world-frame, float32):
        camera_centers  (8, 3)   world-frame camera positions
        optical_axis    (8, 3)   unit vectors, world-frame forward direction
        baseline_matrix (8, 8)   pairwise Euclidean distance between camera
                                  centers (mm, same units as annot3)
    """
    cams = parse_calibration()
    centers, axes = [], []
    for cid in cam_ids:
        R, T = cams[cid]['R'], cams[cid]['T']
        C = -R.T @ T                    # camera center in world coords
        axis = R.T @ np.array([0.0, 0.0, 1.0])   # forward (+Z) direction, world coords
        centers.append(C)
        axes.append(axis / np.linalg.norm(axis))
    centers = np.stack(centers).astype(np.float32)               # (8,3)
    axes    = np.stack(axes).astype(np.float32)                  # (8,3)
    baseline = np.linalg.norm(
        centers[:, None, :] - centers[None, :, :], axis=-1
    ).astype(np.float32)                                          # (8,8)
    return centers, axes, baseline


def world_points_from_cam0(annot3_cam0_frame: np.ndarray, cam0_R: np.ndarray,
                           cam0_T: np.ndarray) -> np.ndarray:
    """annot3_cam0_frame: (28,3) or (N,28,3) camera-0-space points -> world-space."""
    return (annot3_cam0_frame - cam0_T) @ cam0_R   # (R^T @ x^T)^T == x @ R


BASELINE_NORM = 8000.0     # mm -- same normalisation constant as exp115_geo_vis_camagnostic.py (measured max ~7841mm)
INTRINSICS_NORM = 100.0   # px -- rough scale of cross-camera intrinsics variation (fx/fy/cx/cy diffs span ~+-76px)


def build_full_extrinsics_features(cam_ids=SELECTED_CAM_IDS):
    """
    Full relative-extrinsics conditioning feature -- a SOTA, lossless
    alternative to the hand-crafted az/el delta + baseline length used by
    exp102/115+. Answers: does that hand-crafted 3-d summary throw away
    real information (camera roll/tilt, translation DIRECTION not just
    magnitude, per-camera intrinsics differences)?

    Returns (n_cams, n_cams, 13) float32, feat[cur, cand] = concat(
        rot6d(6), trans_local(3), intrinsics_diff(4)
    ) where:
      - rot6d (6,): first two columns of R_rel = R_cand @ R_cur.T, the
        continuous 6D rotation representation (Zhou et al. 2019, CVPR) --
        avoids the discontinuities of Euler/axis-angle, and captures the
        FULL relative orientation (including camera roll around its own
        optical axis) that az/el (2 DOF, viewing-angle-only) cannot.
      - trans_local (3,): (C_cand - C_cur) expressed in the CURRENT
        camera's own local frame (R_cur @ delta_world), normalized by
        BASELINE_NORM. Unlike scalar baseline length (|delta|), this keeps
        the DIRECTION of the offset (left/right, up/down, fwd/back
        relative to what "current" is looking at) -- baseline length is
        exactly this vector's magnitude, so this feature is a strict
        superset.
      - intrinsics_diff (4,): (fx,fy,cx,cy)_cand - (fx,fy,cx,cy)_cur,
        normalized by INTRINSICS_NORM. Small but real per-physical-camera
        manufacturing/mounting differences (~+-76px raw), never
        represented in any prior experiment in this project.

    All 100% clean / deployable -- purely a function of the (fixed,
    known-in-advance) camera rig calibration, no per-frame observation
    required, exactly like az/el delta + baseline length.
    """
    cams = parse_calibration()
    n = len(cam_ids)
    Rs = np.stack([cams[c]['R'] for c in cam_ids])   # (n,3,3)
    Ts = np.stack([cams[c]['T'] for c in cam_ids])   # (n,3)
    Ks = np.stack([cams[c]['K'] for c in cam_ids])   # (n,3,3)
    centers = np.stack([-Rs[i].T @ Ts[i] for i in range(n)])   # (n,3) world

    feat = np.zeros((n, n, 13), dtype=np.float32)
    for cur in range(n):
        for cand in range(n):
            R_rel = Rs[cand] @ Rs[cur].T
            rot6d = R_rel[:, :2].flatten(order='F')                  # (6,)
            t_rel_world = centers[cand] - centers[cur]
            t_rel_local = (Rs[cur] @ t_rel_world) / BASELINE_NORM     # (3,)
            intr_diff = np.array([
                Ks[cand, 0, 0] - Ks[cur, 0, 0], Ks[cand, 1, 1] - Ks[cur, 1, 1],
                Ks[cand, 0, 2] - Ks[cur, 0, 2], Ks[cand, 1, 2] - Ks[cur, 1, 2],
            ]) / INTRINSICS_NORM                                       # (4,)
            feat[cur, cand] = np.concatenate([rot6d, t_rel_local, intr_diff])
    return feat


IMAGE_SIZE = 2048   # px -- see build_keypoint_splits.py: x_norm = x_px/IMAGE_SIZE - 0.5


def build_fundamental_matrices(cam_ids=SELECTED_CAM_IDS):
    """
    Fundamental matrix F[cur, cand] (3,3) such that for a point x_cur in the
    CURRENT camera's image (homogeneous pixel coords), l_cand = F @ x_cur is
    the epipolar line (a,b,c), a*x+b*y+c=0, that the corresponding point in
    the CANDIDATE camera's image must lie on. Camera-rig calibration only
    (100% clean/fixed) -- the actual epipolar LINE for a given frame still
    needs that frame's real current-view keypoints (computed at runtime in
    the model's forward pass, not here).

    F[cur, cur] is degenerate (zero matrix, t_rel=0) -- guarded downstream
    with an epsilon norm, since "candidate == current" has no epipolar
    constraint (same camera, same point).

    Returns (n_cams, n_cams, 3, 3) float32.
    """
    cams = parse_calibration()
    n = len(cam_ids)
    Rs = np.stack([cams[c]['R'] for c in cam_ids])   # (n,3,3)
    Ts = np.stack([cams[c]['T'] for c in cam_ids])   # (n,3)
    Ks = np.stack([cams[c]['K'] for c in cam_ids])   # (n,3,3)
    Ks_inv = np.linalg.inv(Ks)

    F = np.zeros((n, n, 3, 3), dtype=np.float32)
    for cur in range(n):
        for cand in range(n):
            R_rel = Rs[cand] @ Rs[cur].T
            t_rel = Ts[cand] - R_rel @ Ts[cur]
            t_x = np.array([
                [0, -t_rel[2], t_rel[1]],
                [t_rel[2], 0, -t_rel[0]],
                [-t_rel[1], t_rel[0], 0],
            ])
            E = t_x @ R_rel
            F_mat = Ks_inv[cand].T @ E @ Ks_inv[cur]
            F[cur, cand] = F_mat.astype(np.float32)
    return F


def estimate_capture_volume_center(cam_ids=SELECTED_CAM_IDS):
    """
    Least-squares intersection point of all 8 cameras' optical-axis rays --
    a principled, calibration-only estimate of "where the subject roughly
    stands" (the rays don't perfectly converge since real footage has the
    subject moving around, but a well-aimed rig's rays cluster near the
    true capture-volume center; residual perpendicular distances here are
    190-710mm against ~2500-5000mm camera distances, i.e. rays converge to
    within ~10-20% of range, consistent with a rig aimed at a common area).

    Returns (3,) float32, world-frame mm.
    """
    centers, axes, _ = build_camera_constants(cam_ids=cam_ids)
    A = np.zeros((3, 3)); b = np.zeros(3)
    for C, d in zip(centers, axes):
        d = d / np.linalg.norm(d)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ C
    return np.linalg.solve(A, b).astype(np.float32)


def build_vergence_angles(cam_ids=SELECTED_CAM_IDS):
    """
    Vergence (parallax) angle -- the angle subtended AT THE SUBJECT'S
    LOCATION by the two camera centers, i.e. angle(C_cur - P, C_cand - P)
    where P is the (fixed, calibration-only) estimated capture-volume
    center. This is the quantity that actually determines triangulation
    precision (a small vergence angle gives poor depth resolution
    regardless of how far apart the cameras physically are) -- baseline
    length (exp115) is only a proxy for it, conflating "cameras far apart"
    with "cameras that see the subject from very different angles" (the
    two coincide only if the subject-to-camera distance is also similar
    across cameras). Assumes a FIXED capture-volume center (per the user's
    framing) rather than the true per-frame subject position, keeping this
    100% clean/deployable -- same status as az/el delta and baseline length.

    Empirically correlated with baseline length (r=0.95 on this rig) but
    not identical -- genuinely different, if overlapping, information.

    Returns (n_cams, n_cams) float32, radians / pi (normalized to ~[0,1]).
    """
    centers, _, _ = build_camera_constants(cam_ids=cam_ids)
    P = estimate_capture_volume_center(cam_ids=cam_ids)
    n = len(cam_ids)
    vergence = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            vi, vj = centers[i] - P, centers[j] - P
            cosang = np.clip((vi @ vj) / (np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8), -1, 1)
            vergence[i, j] = np.arccos(cosang)
    return (vergence / np.pi).astype(np.float32)


if __name__ == '__main__':
    centers, axes, baseline = build_camera_constants()
    print('camera_centers (world, mm):')
    print(centers)
    print('\noptical_axis (world, unit):')
    print(axes)
    print('\nbaseline_matrix (mm):')
    print(baseline.round(1))
    print('\nbaseline range:', baseline[baseline > 0].min(), '-', baseline.max(), 'mm')


# ==================== was skeleton_config.py ====================

# skeleton_config.py
#
# COCO 17-joint skeleton configuration used by all skeleton-graph experiments.
#
# Joint order matches ViTPose-Small output (confirmed in stage2_build_vit_labels.py):
#   joints_2d = keypoints[:17]   — first 17 joints, COCO body order
#
# Edges: official COCO skeleton_links used by mmpose / ViTPose for visualization.
# Source: mmpose dataset_zoo/2d_body_keypoint.html

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
