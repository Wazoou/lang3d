"""Étape 3 : entraînement du Gaussian Splatting avec gsplat.

À chaque itération : on tire une caméra au hasard, on rend l'image avec les
gaussiennes actuelles, on compare à la vraie photo (L1 + D-SSIM) et on
rétropropage. La `DefaultStrategy` de gsplat densifie (clone / split) les
gaussiennes là où le gradient est fort et élague les gaussiennes transparentes.
"""
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.strategy import DefaultStrategy
from tqdm import tqdm

from .gaussians import export_ply, init_splats, render, save_splats
from .scene import load_frame_image, load_scene


def ssim(x: torch.Tensor, y: torch.Tensor, window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """SSIM moyen entre deux images (B, C, H, W) dans [0, 1]."""
    coords = torch.arange(window, device=x.device, dtype=x.dtype) - window // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = (g / g.sum())[:, None] @ (g / g.sum())[None, :]
    c = x.shape[1]
    kernel = g.expand(c, 1, window, window)

    def blur(t):
        return F.conv2d(t, kernel, padding=window // 2, groups=c)

    mu_x, mu_y = blur(x), blur(y)
    sxx = blur(x * x) - mu_x**2
    syy = blur(y * y) - mu_y**2
    sxy = blur(x * y) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.mean()


def train(scene_dir: Path, steps: int = 7000, downscale: int = 1, sh_degree: int = 3,
          ssim_weight: float = 0.2, device: str = "cuda"):
    frames, xyz, rgb = load_scene(scene_dir)
    print(f"{len(frames)} caméras, {len(xyz)} points COLMAP")

    # Images gardées en uint8 sur le CPU, envoyées au GPU une par une.
    images = [load_frame_image(f, downscale) for f in frames]
    Ks = [torch.from_numpy(f.K).float().to(device) for f in frames]
    for K in Ks:
        K[:2] /= downscale
    viewmats = [torch.from_numpy(f.viewmat).float().to(device) for f in frames]

    centers = np.stack([f.cam_center for f in frames])
    scene_scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    # « Haut » de la scène : moyenne des axes -Y des caméras (convention OpenCV).
    up = -np.mean([f.viewmat[1, :3] for f in frames], axis=0)
    up /= np.linalg.norm(up)

    splats = init_splats(xyz, rgb, sh_degree, device)
    lrs = {
        "means": 1.6e-4 * scene_scale,
        "scales": 5e-3,
        "quats": 1e-3,
        "opacities": 5e-2,
        "sh0": 2.5e-3,
        "shN": 2.5e-3 / 20,
    }
    optimizers = {
        name: torch.optim.Adam([{"params": splats[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }
    # Le pas d'apprentissage des positions décroît exponentiellement jusqu'à 1 % de sa valeur.
    means_sched = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / steps))

    strategy = DefaultStrategy(
        refine_start_iter=500,
        refine_stop_iter=int(steps * 0.6),
        reset_every=3000,
        refine_every=100,
        verbose=False,
    )
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)

    pbar = tqdm(range(steps), desc="Entraînement 3DGS")
    for step in pbar:
        i = np.random.randint(len(frames))
        gt = images[i].to(device, non_blocking=True).float() / 255.0  # (H, W, 3)
        h, w = gt.shape[:2]
        # On débloque un degré d'harmoniques sphériques tous les 1000 pas.
        cur_sh = min(step // 1000, sh_degree)

        renders, _, info = render(splats, viewmats[i], Ks[i], w, h, sh_degree=cur_sh)
        pred = renders[0].clamp(0, 1)
        strategy.step_pre_backward(splats, optimizers, state, step, info)

        l1 = (pred - gt).abs().mean()
        d_ssim = 1 - ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None])
        loss = (1 - ssim_weight) * l1 + ssim_weight * d_ssim
        loss.backward()

        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)

        if step % 100 == 0:
            psnr = -10 * math.log10(F.mse_loss(pred, gt).item() + 1e-10)
            pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}", n=len(splats["means"]))

    out = scene_dir / "gs"
    save_splats(out / "splats.pt", splats, sh_degree, up)
    export_ply(out / "scene.ply", splats)
    print(f"Terminé : {len(splats['means'])} gaussiennes → {out}")
