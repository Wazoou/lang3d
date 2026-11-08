import struct

import cv2
import numpy as np

from lang3d.colmap import import_colmap, qvec_to_rotmat, read_cameras, read_images, read_points3d


def write_reconstruction(sparse, W=200, H=100):
    sparse.mkdir(parents=True)
    with open(sparse / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, W, H))  # PINHOLE
        f.write(struct.pack("<4d", 150.0, 160.0, W / 2, H / 2))
    with open(sparse / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 2))
        for img_id, name, tx in [(1, "b.jpg", 1.0), (2, "a.jpg", -1.0)]:
            f.write(struct.pack("<idddddddi", img_id, 1, 0, 0, 0, tx, 0, 3.0, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ddq", 10.0, 20.0, -1))
    with open(sparse / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 2))
        for pid, xyz, rgb in [(1, (0, 0, 1), (255, 0, 0)), (2, (1, 2, 3), (0, 255, 0))]:
            f.write(struct.pack("<QdddBBBd", pid, *xyz, *rgb, 0.5))
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<ii", 1, 0))


def test_read_binary_files(tmp_path):
    sparse = tmp_path / "sparse" / "0"
    write_reconstruction(sparse)
    cams = read_cameras(sparse / "cameras.bin")
    assert cams[1][0] == "PINHOLE" and cams[1][1:3] == (200, 100)
    images = read_images(sparse / "images.bin")
    assert [i[0] for i in images] == ["b.jpg", "a.jpg"]
    xyz, rgb = read_points3d(sparse / "points3D.bin")
    assert xyz.shape == (2, 3) and rgb[1].tolist() == [0, 255, 0]


def test_import_scales_intrinsics_to_actual_images(tmp_path):
    write_reconstruction(tmp_path / "sparse" / "0")
    (tmp_path / "images").mkdir()
    for name in ("a.jpg", "b.jpg"):  # images deux fois plus petites que celles vues par COLMAP
        cv2.imwrite(str(tmp_path / "images" / name), np.zeros((50, 100, 3), np.uint8))
    import_colmap(tmp_path)
    d = np.load(tmp_path / "poses.npz")
    assert d["names"].tolist() == ["a.jpg", "b.jpg"]  # triées par nom
    np.testing.assert_allclose(d["K"][0], [[75, 0, 50], [0, 80, 25], [0, 0, 1]])
    np.testing.assert_allclose(d["viewmats"][0][:3, 3], [-1, 0, 3])


def test_qvec_to_rotmat_is_a_rotation():
    q = np.array([0.9, 0.1, -0.3, 0.2])
    R = qvec_to_rotmat(q)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1)
