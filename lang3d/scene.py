"""Chargement d'une scène : images + caméras (K, world→caméra) + nuage de points initial.

Tout est stocké dans `<scene>/poses.npz`, écrit par l'étape `poses`.
Conventions OpenCV : x à droite, y vers le bas, z vers l'avant.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass
class Frame:
    name: str
    path: Path
    K: np.ndarray        # (3, 3) intrinsèques à la résolution native de l'image
    viewmat: np.ndarray  # (4, 4) transformation monde → caméra
    width: int
    height: int

    @property
    def cam_center(self) -> np.ndarray:
        R, t = self.viewmat[:3, :3], self.viewmat[:3, 3]
        return -R.T @ t


def load_scene(scene_dir: Path):
    d = np.load(scene_dir / "poses.npz")
    frames = [
        Frame(
            name=str(name),
            path=scene_dir / "images" / str(name),
            K=d["K"][i],
            viewmat=d["viewmats"][i],
            width=int(d["width"]),
            height=int(d["height"]),
        )
        for i, name in enumerate(d["names"])
    ]
    return frames, d["xyz"], d["rgb"]


def load_frame_image(frame: Frame, downscale: int = 1) -> torch.Tensor:
    """Image RGB uint8 (H, W, 3), éventuellement réduite d'un facteur entier."""
    img = cv2.cvtColor(cv2.imread(str(frame.path)), cv2.COLOR_BGR2RGB)
    if downscale > 1:
        img = cv2.resize(img, (frame.width // downscale, frame.height // downscale), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img)
