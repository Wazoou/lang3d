"""Export d'une démo web statique (GitHub Pages) : la scène + des requêtes précalculées.

Fichiers produits dans `out_dir` :
  scene.splat    gaussiennes au format .splat (32 octets chacune : position, échelle, RGBA, rotation)
  queries.bin    pour chaque requête, un score 0-255 par gaussienne (même ordre que scene.splat)
  queries.json   libellés, niveaux, seuil par défaut et point de vue initial
  features.bin   un vecteur de sens compressé par gaussienne (int8), pour la sélection au clic
  index.html, app.js   la page (gsplat.js, chargé depuis un CDN)

SigLIP 2 ne tourne pas dans le navigateur : les requêtes sont donc calculées ici,
à l'avance. Pour une requête libre, il faut le viewer local (`lang3d view`).
"""
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .gaussians import load_splats, sh_to_rgb
from .query import SceneQuery, clean_query
from .scene import load_scene

WEB_DIR = Path(__file__).resolve().parent / "web"


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation 3×3 qui envoie le vecteur unitaire a sur b (formule de Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:  # vecteurs opposés : demi-tour autour d'un axe perpendiculaire
        axis = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) d'une matrice de rotation (méthode de Shepperd, stable)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(t + 1)
        return np.array([s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = s / 4
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Produit de quaternions (w, x, y, z) ; q2 peut être un tableau (N, 4)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def to_splat_bytes(pos, scales, rgba, quats) -> bytes:
    rows = np.zeros(len(pos), dtype=[("pos", "<f4", 3), ("scale", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    rows["pos"], rows["scale"], rows["rgba"] = pos, scales, rgba
    q = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    rows["rot"] = np.clip(q * 128 + 128, 0, 255).astype(np.uint8)
    return rows.tobytes()


def export_web(scene_dir: Path, queries: list[str], out_dir: Path, max_gaussians: int = 250_000,
               threshold: float = 0.5, feature_dim: int = 16, device: str = "cuda"):
    splats, _, up = load_splats(scene_dir / "gs" / "splats.pt", device)
    frames, xyz, _ = load_scene(scene_dir)
    opacity = torch.sigmoid(splats["opacities"])
    scales = torch.exp(splats["scales"])

    # Garder les gaussiennes qui comptent le plus à l'écran (opacité × taille), triées par importance.
    importance = opacity * scales.prod(dim=1) ** (1 / 3)
    importance[opacity < 0.05] = -1
    order = torch.argsort(importance, descending=True)[:max_gaussians]
    order = order[importance[order] > 0]

    # Repère de la démo : scène centrée, « haut » de la scène aligné sur -y (convention des scènes 3DGS).
    center = np.median(xyz, axis=0)
    R = rotation_between(np.asarray(up, np.float64), np.array([0.0, -1.0, 0.0]))
    q_align = rotmat_to_quat(R)
    sel = order.cpu().numpy()
    pos = (splats["means"].detach().cpu().numpy()[sel] - center) @ R.T
    quats = quat_multiply(q_align, splats["quats"].detach().cpu().numpy()[sel])
    rgb = sh_to_rgb(splats["sh0"][order, 0]).detach().cpu().numpy()
    rgba = np.concatenate([rgb, opacity[order, None].detach().cpu().numpy()], 1) * 255
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scene.splat").write_bytes(
        to_splat_bytes(pos, scales[order].detach().cpu().numpy(), rgba.round().astype(np.uint8), quats))

    q = SceneQuery(scene_dir, device)
    blocks, meta = [], []
    for text in queries:
        # « rideau rouge* » : l'étoile active l'ensemble de prompts (option « objet entier »).
        ensemble = text.endswith("*")
        text = text.rstrip("*").strip()
        s, level = q.scores(text, ensemble=ensemble)
        blocks.append((s[order].clamp(0, 1) * 255).round().byte().cpu().numpy())
        meta.append({"label": clean_query(text), "level": level})
        print(f"  « {text} » [{level}] : {(s[order] > threshold).sum().item()} gaussiennes au-dessus du seuil")
    (out_dir / "queries.bin").write_bytes(np.concatenate(blocks).tobytes())

    # Sélection au clic dans le navigateur : un vecteur de sens par gaussienne, compressé en
    # `feature_dim` dimensions (SVD des features du niveau « large ») et quantifié sur 1 octet.
    feats = q.feats["large"][order]
    valid = q.valid["large"][order]
    _, _, Vh = torch.linalg.svd(feats[valid], full_matrices=False)
    low = torch.nn.functional.normalize(feats @ Vh[:feature_dim].T, dim=-1)
    low[~valid] = 0
    (out_dir / "features.bin").write_bytes((low * 127).round().to(torch.int8).cpu().numpy().tobytes())
    # Fidélité de la compression : corrélation des similarités avant / après, sur un échantillon.
    idx = torch.randperm(int(valid.sum()), device=feats.device)[:2000]
    ref, sub = feats[valid][idx[:20]], feats[valid][idx]
    fid = torch.corrcoef(torch.stack([(ref @ sub.T).flatten(), (low[valid][idx[:20]] @ low[valid][idx].T).flatten()]))[0, 1]
    print(f"  features {feature_dim} dims pour la sélection au clic (corrélation des similarités : {fid:.3f})")

    cam0 = frames[0].cam_center
    forward = frames[0].viewmat[2, :3]
    target = cam0 + forward * np.linalg.norm(np.median(xyz, 0) - cam0)
    (out_dir / "queries.json").write_text(json.dumps({
        "count": int(len(sel)), "threshold": threshold, "queries": meta, "feature_dim": feature_dim,
        "camera": {"position": ((cam0 - center) @ R.T).round(4).tolist(),
                   "target": ((target - center) @ R.T).round(4).tolist()},
    }, ensure_ascii=False, indent=1))
    for f in ("index.html", "app.js"):
        shutil.copy(WEB_DIR / f, out_dir / f)
    size = sum(p.stat().st_size for p in out_dir.iterdir()) / 1e6
    print(f"Démo web : {len(sel)} gaussiennes, {len(queries)} requêtes, {size:.1f} Mo → {out_dir}")
