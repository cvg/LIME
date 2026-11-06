import numpy as np
import pytest

from lime_camera.pose import PoseNorm, mat_to_pose9, pose9_to_mat, r6d_to_R, relative_pose


def random_transform(rng) -> np.ndarray:
    Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    Q[:, 2] *= np.linalg.det(Q)
    T = np.eye(4)
    T[:3, :3] = Q
    T[:3, 3] = rng.normal(size=3)
    return T


def test_r6d_always_gives_a_rotation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = r6d_to_R(rng.normal(size=6))
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-7)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_pose9_round_trip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        T = random_transform(rng)
        assert np.allclose(pose9_to_mat(mat_to_pose9(T)), T, atol=1e-9)


def test_moving_forward_gives_positive_tz():
    # OpenCV axes: a camera 2 m ahead of the start camera, same orientation.
    T_world_end = np.eye(4)
    T_world_end[2, 3] = 2.0
    T_rel = relative_pose(np.eye(4), np.linalg.inv(T_world_end))
    assert np.allclose(T_rel[:3, 3], [0.0, 0.0, 2.0])


def test_target_pose_composes_on_the_right():
    rng = np.random.default_rng(2)
    for _ in range(20):
        T_world_start, T_world_end = random_transform(rng), random_transform(rng)
        T_rel = relative_pose(np.linalg.inv(T_world_start), np.linalg.inv(T_world_end))
        assert np.allclose(T_world_start @ T_rel, T_world_end, atol=1e-9)


def test_pose_norm_applies_the_std_floor_and_handles_batches(tmp_path):
    stats = tmp_path / "stats.json"
    stats.write_text('{"mean": [1, 0, 0, 0, 0, 0, 0, 0, 0],'
                     ' "std": [2, 0.001, 1, 1, 1, 1, 1, 1, 1], "std_floor": 0.01}')
    norm = PoseNorm.from_file(stats)
    assert norm.std[1] == pytest.approx(0.01)
    assert norm.unnormalize(np.ones(9))[0] == pytest.approx(3.0)
    assert norm.unnormalize(np.zeros((4, 9))).shape == (4, 9)
