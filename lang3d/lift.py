"""Étape 5 : report des embeddings 2D sur les gaussiennes 3D, sans entraînement.

Le rendu d'un pixel p est linéaire en la « couleur » c_i des gaussiennes :
    rendu(p) = Σ_i w_ip · c_i      avec w_ip = α_i · T_i (opacité × transmittance)

Donc, pour une carte cible F(p), le gradient de L = Σ_p <rendu(p), F(p)> par
rapport à c_i vaut exactement Σ_p w_ip · F(p) : la somme des features des
pixels que la gaussienne i couvre, pondérée par sa contribution. Un seul
rendu + backward par image donne cette somme pour toutes les gaussiennes à la
fois. On divise ensuite par Σ_p w_ip (canal supplémentaire rempli de 1).

Pour limiter la mémoire, on compresse d'abord les embeddings (768 dims) par
une SVD non centrée : la projection P conserve les produits scalaires avec le
texte (<t, f> ≈ <Pᵀt, Pᵀf>) tant que f est bien représenté dans la base.
"""
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .gaussians import load_splats, render
from .scene import load_scene


def splat_backward(splats, viewmat: torch.Tensor, K: torch.Tensor, w: int, h: int,
                   target: torch.Tensor) -> torch.Tensor:
    """Σ_p w_ip · target(p) pour chaque gaussienne i, en un rendu + un backward. target : (h, w, D)."""
    colors = torch.zeros(len(splats["means"]), target.shape[-1], device=target.device, requires_grad=True)
    out, _, _ = render(splats, viewmat, K, w, h, colors=colors)
    (out[0] * target).sum().backward()
    return colors.grad


def lift_features(scene_dir: Path, dim: int = 128, device: str = "cuda"):
    splats, _, _ = load_splats(scene_dir / "gs" / "splats.pt", device)
    frames = {f.name: f for f in load_scene(scene_dir)[0]}
    data = torch.load(scene_dir / "lang" / "features.pt", weights_only=False)

    all_feats = torch.cat([f for f in data["feats"] if len(f)]).float().to(device)
    dim = min(dim, len(all_feats))
    _, S, Vh = torch.linalg.svd(all_feats, full_matrices=False)
    P = Vh[:dim].T  # (768, dim)
    kept = (S[:dim] ** 2).sum() / (S**2).sum()
    print(f"SVD : {all_feats.shape[1]} → {dim} dimensions, {kept:.1%} de l'énergie conservée")

    n = len(splats["means"])
    feats_3d, valid = {}, {}
    for level, indices in data["index"].items():
        acc = torch.zeros(n, dim + 1, device=device)
        for name, index, feats in tqdm(list(zip(data["names"], indices, data["feats"])), desc=f"Lifting [{level}]"):
            if not len(feats):
                continue
            f = frames[name]
            h, w = index.shape
            K = torch.from_numpy(f.K).float().to(device)
            K[0] *= w / f.width
            K[1] *= h / f.height
            viewmat = torch.from_numpy(f.viewmat).float().to(device)

            # Table (features projetées | 1) + une ligne de zéros en dernier : l'index -1
            # (pixel sans masque) tombe dessus et ne contribue ni aux features ni aux poids.
            m = len(feats)
            table = torch.zeros(m + 1, dim + 1, device=device)
            table[:m, :dim] = feats.float().to(device) @ P
            table[:m, dim] = 1.0
            target = table[index.long().to(device)]  # (h, w, dim + 1)
            acc += splat_backward(splats, viewmat, K, w, h, target)

        weight = acc[:, dim]
        feats_3d[level] = F.normalize(acc[:, :dim] / weight.clamp_min(1e-8)[:, None], dim=-1).half().cpu()
        valid[level] = (weight > 1e-2).cpu()
        print(f"[{level}] {valid[level].sum().item()} / {n} gaussiennes ont reçu une feature")

    out_path = scene_dir / "lang" / "gaussians_lang.pt"
    torch.save({"P": P.cpu(), "feats": feats_3d, "valid": valid, "model": data["model"]}, out_path)
    print(f"→ {out_path}")
