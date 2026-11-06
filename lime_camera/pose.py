"""Pose conventions for the 9-D output of the flow-matching head.

The head emits ``pose9 = [tx, ty, tz, r1..r6]`` where the six rotation entries
are the first two columns of the rotation matrix (Zhou et al. 2019) and are
turned back into a valid 3x3 by Gram-Schmidt.

Coordinate convention:

    T_rel = T_start_w2c @ inv(T_end_w2c)

in OpenCV camera convention (x-right, y-down, z-forward). So ``T_rel`` is the
pose of the end camera in the start-camera frame, and the translation ``t_rel``
is where the end camera sits as seen from the start camera (``tz > 0`` moves
forward). To turn a prediction into a target pose from a known start pose:

    T_world_end = T_world_start @ T_rel

For visualization we put the start camera at the world origin
(``T_start_w2c = I``), which makes the predicted end camera's camera-to-world
matrix exactly ``T_e_c2w = T_rel`` — i.e. ``pose9_to_mat(pose9)`` can be
dropped straight into the scene as the predicted camera frame.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def r6d_to_R(r6) -> np.ndarray:
    """First-two-columns 6D rotation -> 3x3 rotation matrix (Gram-Schmidt)."""
    a1 = np.asarray(r6[:3], dtype=np.float64)
    a2 = np.asarray(r6[3:], dtype=np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-9)
    b2 = a2 - (b1 @ a2) * b1
    b2 /= np.linalg.norm(b2) + 1e-9
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


def pose9_to_mat(p9) -> np.ndarray:
    """[tx, ty, tz, r1..r6] -> 4x4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = r6d_to_R(p9[3:])
    T[:3, 3] = p9[:3]
    return T


def mat_to_pose9(T) -> np.ndarray:
    """4x4 homogeneous transform -> [tx, ty, tz, r1..r6]."""
    T = np.asarray(T, dtype=np.float64)
    return np.concatenate([T[:3, 3], T[:3, 0], T[:3, 1]])


def pose7_to_mat(pose7) -> np.ndarray:
    """[qw, qx, qy, qz, tx, ty, tz] -> 4x4 homogeneous transform."""
    qw, qx, qy, qz, tx, ty, tz = pose7
    norm = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ]
    T[:3, 3] = tx, ty, tz
    return T


def relative_pose(T_start_w2c, T_end_w2c) -> np.ndarray:
    """T_rel from two world-to-camera poses: the convention above."""
    return np.asarray(T_start_w2c) @ np.linalg.inv(T_end_w2c)


@dataclass
class PoseNorm:
    """Per-axis mean and std the head's poses are standardized by."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def from_file(cls, path) -> "PoseNorm":
        stats = json.loads(Path(path).read_text())
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.maximum(np.asarray(stats["std"], dtype=np.float64),
                         float(stats.get("std_floor", 0.01)))
        return cls(mean, std)

    def unnormalize(self, pose9) -> np.ndarray:
        """Map standardized poses, (9,) or (K, 9), back to metres."""
        return np.asarray(pose9, dtype=np.float64) * self.std + self.mean
