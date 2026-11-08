"""Import d'une reconstruction COLMAP existante (format binaire) vers `poses.npz`.

Sert à l'évaluation : les datasets de référence (LERF, 3D-OVS) fournissent
leurs poses COLMAP, et les réutiliser rend la comparaison avec LangSplat
équitable. Lecteur en Python pur, car Windows bloque la DLL de pycolmap.
"""
import struct
from pathlib import Path

import cv2
import numpy as np

# model_id → (nom, nombre de paramètres)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3), 1: ("PINHOLE", 4), 2: ("SIMPLE_RADIAL", 4), 3: ("RADIAL", 5),
    4: ("OPENCV", 8), 5: ("OPENCV_FISHEYE", 8), 6: ("FULL_OPENCV", 12),
}


def _read(f, fmt: str):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))


def read_cameras(path: Path) -> dict[int, tuple[str, int, int, np.ndarray]]:
    cams = {}
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            cam_id, model_id, w, h = _read(f, "iiQQ")
            name, n = CAMERA_MODELS[model_id]
            cams[cam_id] = (name, w, h, np.array(_read(f, "d" * n)))
    return cams


def read_images(path: Path) -> list[tuple[str, int, np.ndarray, np.ndarray]]:
    """[(nom, camera_id, qvec wxyz, tvec)] ; pose monde → caméra."""
    images = []
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            _, qw, qx, qy, qz, tx, ty, tz, cam_id = _read(f, "idddddddi")
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            n_pts = _read(f, "Q")[0]
            f.seek(n_pts * 24, 1)  # (x, y, point3D_id) par point 2D, inutile ici
            images.append((name.decode(), cam_id, np.array([qw, qx, qy, qz]), np.array([tx, ty, tz])))
    return images


def read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    xyz, rgb = [], []
    with open(path, "rb") as f:
        for _ in range(_read(f, "Q")[0]):
            vals = _read(f, "QdddBBBd")
            xyz.append(vals[1:4])
            rgb.append(vals[4:7])
            f.seek(_read(f, "Q")[0] * 8, 1)  # piste (image_id, point2D_idx)
    return np.array(xyz, np.float32), np.array(rgb, np.uint8)


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def intrinsics(model: str, params: np.ndarray) -> np.ndarray:
    if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
        fx = fy = params[0]
        cx, cy = params[1:3]
    else:
        fx, fy, cx, cy = params[:4]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)


def import_colmap(scene_dir: Path, sparse: str = "sparse/0"):
    sp = scene_dir / sparse
    cams = read_cameras(sp / "cameras.bin")
    images = sorted(read_images(sp / "images.bin"))
    xyz, rgb = read_points3d(sp / "points3D.bin")

    # Les images fournies peuvent être plus petites que celles vues par COLMAP.
    H, W = cv2.imread(str(scene_dir / "images" / images[0][0])).shape[:2]
    Ks, viewmats = [], []
    for name, cam_id, q, t in images:
        model, cw, ch, params = cams[cam_id]
        first_dist = 4 if "OPENCV" in model else 3  # index du premier coefficient de distorsion
        if model not in ("SIMPLE_PINHOLE", "PINHOLE") and np.abs(params[first_dist:]).max() > 1e-3:
            print(f"Attention : distorsion {model} ignorée pour {name}")
        K = intrinsics(model, params)
        K[0] *= W / cw
        K[1] *= H / ch
        V = np.eye(4, dtype=np.float32)
        V[:3, :3] = qvec_to_rotmat(q)
        V[:3, 3] = t
        Ks.append(K)
        viewmats.append(V)

    np.savez(
        scene_dir / "poses.npz",
        names=np.array([n for n, *_ in images]),
        K=np.stack(Ks),
        viewmats=np.stack(viewmats),
        width=W,
        height=H,
        xyz=xyz,
        rgb=rgb,
    )
    print(f"{len(images)} caméras COLMAP, {len(xyz)} points, images {W}x{H} → {scene_dir / 'poses.npz'}")
