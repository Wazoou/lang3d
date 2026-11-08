"""GIF de démo : la caméra suit la trajectoire filmée ; la première moitié montre
la scène, la seconde la requête surlignée."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from .gaussians import load_splats, render, sh_to_rgb
from .query import SceneQuery, clean_query, highlight
from .scene import load_scene


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _trajectory(frames, steps_between: int) -> list[np.ndarray]:
    """Interpolation entre caméras successives : centre linéaire, rotation par nlerp."""
    views = []
    for a, b in zip(frames[:-1], frames[1:]):
        qa, qb = _rot_to_quat(a.viewmat[:3, :3].T), _rot_to_quat(b.viewmat[:3, :3].T)
        if qa @ qb < 0:
            qb = -qb
        for t in np.linspace(0, 1, steps_between, endpoint=False):
            R_wc = _quat_to_rot((1 - t) * qa + t * qb)  # rotation caméra → monde
            c = (1 - t) * a.cam_center + t * b.cam_center
            V = np.eye(4)
            V[:3, :3] = R_wc.T
            V[:3, 3] = -R_wc.T @ c
            views.append(V)
    return views


def make_demo(scene_dir: Path, text: str, out: Path, max_side: int = 320, steps_between: int = 2,
              threshold: float = 0.5, device: str = "cuda"):
    splats, sh_degree, _ = load_splats(scene_dir / "gs" / "splats.pt", device)
    frames, _, _ = load_scene(scene_dir)
    scores, _ = SceneQuery(scene_dir, device).scores(text)
    colors, _ = highlight(sh_to_rgb(splats["sh0"][:, 0]), scores, threshold)

    s = max_side / max(frames[0].width, frames[0].height)
    w, h = round(frames[0].width * s), round(frames[0].height * s)
    K = torch.from_numpy(frames[0].K).float().to(device)
    K[0] *= w / frames[0].width
    K[1] *= h / frames[0].height

    views = _trajectory(frames, steps_between)
    images = []
    for i, V in enumerate(views):
        hl = i >= len(views) // 2
        with torch.no_grad():
            r, _, _ = render(splats, torch.from_numpy(V).float().to(device), K, w, h,
                             colors=colors if hl else None, sh_degree=sh_degree)
        img = (r[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        if hl:
            label = f"« {clean_query(text)} »"
            img = cv2.rectangle(img.copy(), (0, 0), (w, 26), (0, 0, 0), -1)
            # OpenCV ne gère pas les accents : on passe par PIL pour le texte.
            pil = Image.fromarray(img)
            ImageDraw.Draw(pil).text((8, 6), label, fill=(255, 220, 0))
            img = np.asarray(pil)
        images.append(Image.fromarray(img))

    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(out, save_all=True, append_images=images[1:], duration=60, loop=0, optimize=True)
    print(f"GIF de {len(images)} images ({w}x{h}) → {out} ({out.stat().st_size / 1e6:.1f} Mo)")
