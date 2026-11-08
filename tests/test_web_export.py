import numpy as np

from lang3d.colmap import qvec_to_rotmat
from lang3d.web_export import quat_multiply, rotation_between, rotmat_to_quat, to_splat_bytes


def test_rotation_between_aligns_vectors():
    for a in ([0.2, 0.9, -0.3], [0, 1.0, 0], [0, -1.0, 0]):
        a = np.array(a) / np.linalg.norm(a)
        R = rotation_between(a, np.array([0.0, -1, 0]))
        np.testing.assert_allclose(R @ a, [0, -1, 0], atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)


def test_quaternion_roundtrip_and_composition():
    q1 = np.array([0.9, 0.1, -0.3, 0.2]); q1 /= np.linalg.norm(q1)
    q2 = np.array([0.5, 0.5, 0.5, -0.5])
    np.testing.assert_allclose(qvec_to_rotmat(rotmat_to_quat(qvec_to_rotmat(q1))), qvec_to_rotmat(q1), atol=1e-9)
    # le produit de quaternions correspond au produit des matrices
    np.testing.assert_allclose(qvec_to_rotmat(quat_multiply(q1, q2)), qvec_to_rotmat(q1) @ qvec_to_rotmat(q2),
                               atol=1e-9)


def test_splat_format_is_32_bytes_per_gaussian():
    n = 5
    raw = to_splat_bytes(np.ones((n, 3)), np.full((n, 3), 0.1), np.full((n, 4), 200, np.uint8),
                         np.tile([1.0, 0, 0, 0], (n, 1)))
    assert len(raw) == 32 * n
    row = np.frombuffer(raw[:32], np.uint8)
    assert np.frombuffer(raw[:12], "<f4").tolist() == [1, 1, 1]
    assert row[24:28].tolist() == [200] * 4         # RGBA
    assert row[28:32].tolist() == [255, 128, 128, 128]  # quaternion identité : w = 1 → 255
