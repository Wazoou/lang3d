"""Représentation d'une scène en gaussiennes 3D : init, rendu, sauvegarde, export.

Chaque gaussienne a :
  - means      (N, 3)      position du centre
  - scales     (N, 3)      log de l'écart-type sur chaque axe local
  - quats      (N, 4)      orientation (quaternion, normalisé par gsplat)
  - opacities  (N,)        logit de l'opacité
  - sh0, shN   (N, K, 3)   couleur en harmoniques sphériques (dépend de l'angle de vue)
"""
from pathlib import Path

import numpy as np
import torch
from gsplat import rasterization

SH_C0 = 0.28209479177387814  # constante de l'harmonique sphérique d'ordre 0


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh0: torch.Tensor) -> torch.Tensor:
    return (sh0 * SH_C0 + 0.5).clamp(0, 1)


def knn_mean_dist(points: torch.Tensor, k: int = 3, chunk: int = 4096) -> torch.Tensor:
    """Distance moyenne de chaque point à ses k plus proches voisins."""
    out = torch.empty(len(points), device=points.device)
    for i in range(0, len(points), chunk):
        d = torch.cdist(points[i : i + chunk], points)
        out[i : i + chunk] = d.topk(k + 1, largest=False).values[:, 1:].mean(1)
    return out


def init_splats(xyz: np.ndarray, rgb: np.ndarray, sh_degree: int, device="cuda") -> torch.nn.ParameterDict:
    means = torch.from_numpy(xyz).float().to(device)
    colors = torch.from_numpy(rgb).float().to(device) / 255.0
    n = len(means)
    # Une gaussienne isotrope par point, de la taille de son voisinage.
    dist = knn_mean_dist(means).clamp_min(1e-7)
    scales = torch.log(dist)[:, None].repeat(1, 3)
    quats = torch.zeros(n, 4, device=device)
    quats[:, 0] = 1.0
    opacities = torch.logit(torch.full((n,), 0.1, device=device))
    sh0 = rgb_to_sh(colors)[:, None, :]
    shN = torch.zeros(n, (sh_degree + 1) ** 2 - 1, 3, device=device)
    return torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "sh0": torch.nn.Parameter(sh0),
            "shN": torch.nn.Parameter(shN),
        }
    )


def render(splats, viewmat: torch.Tensor, K: torch.Tensor, width: int, height: int,
           colors: torch.Tensor | None = None, sh_degree: int | None = None):
    """Rend une vue. Par défaut la couleur RGB (SH) ; sinon `colors` (N, D) quelconque.

    `colors` permet de « splatter » n'importe quel vecteur par gaussienne :
    c'est ce qu'on utilise pour les features langage.
    """
    if colors is None:
        colors = torch.cat([splats["sh0"], splats["shN"]], 1)
    else:
        sh_degree = None
    return rasterization(
        means=splats["means"],
        quats=splats["quats"],
        scales=torch.exp(splats["scales"]),
        opacities=torch.sigmoid(splats["opacities"]),
        colors=colors,
        viewmats=viewmat[None],
        Ks=K[None],
        width=width,
        height=height,
        sh_degree=sh_degree,
        packed=False,
    )


def save_splats(path: Path, splats, sh_degree: int, up: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"splats": {k: v.detach().cpu() for k, v in splats.items()}, "sh_degree": sh_degree, "up": up},
        path,
    )


def load_splats(path: Path, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    splats = {k: v.to(device) for k, v in ckpt["splats"].items()}
    return splats, ckpt["sh_degree"], ckpt["up"]


def export_ply(path: Path, splats, sh0_override: torch.Tensor | None = None,
               opacity_override: torch.Tensor | None = None):
    """Export au format .ply standard du 3DGS (lisible par SuperSplat, etc.)."""
    from plyfile import PlyData, PlyElement

    means = splats["means"].detach().cpu().numpy()
    sh0 = (sh0_override if sh0_override is not None else splats["sh0"]).detach().cpu()
    f_dc = sh0.reshape(len(means), -1).numpy()
    # Le format attend les coefficients groupés par canal : (N, 3, K) aplati.
    f_rest = splats["shN"].detach().cpu().transpose(1, 2).reshape(len(means), -1).numpy()
    opac = (opacity_override if opacity_override is not None else splats["opacities"]).detach().cpu().numpy()
    scales = splats["scales"].detach().cpu().numpy()
    quats = torch.nn.functional.normalize(splats["quats"].detach(), dim=-1).cpu().numpy()

    names = ["x", "y", "z", "nx", "ny", "nz"]
    names += [f"f_dc_{i}" for i in range(f_dc.shape[1])]
    names += [f"f_rest_{i}" for i in range(f_rest.shape[1])]
    names += ["opacity"] + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)]
    data = np.concatenate(
        [means, np.zeros_like(means), f_dc, f_rest, opac[:, None], scales, quats], axis=1
    ).astype(np.float32)
    elements = np.ascontiguousarray(data).view(dtype=[(n, "f4") for n in names]).reshape(-1)
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(elements, "vertex")]).write(str(path))
    print(f"Export .ply ({len(means)} gaussiennes) → {path}")
