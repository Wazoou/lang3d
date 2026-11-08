"""Étape 2 : poses de caméra et nuage de points avec VGGT.

VGGT (Meta, CVPR 2025) est un transformer « feed-forward » : en une seule
passe sur toutes les images, il prédit les caméras (intrinsèques +
extrinsèques) et une carte de profondeur par image. Pas de matching de
points SIFT ni d'optimisation itérative comme dans COLMAP.

Sur 8 Go de VRAM, on passe le modèle en fp16 et on se limite à ~40 images.
"""
from pathlib import Path

import cv2
import numpy as np
import torch

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _prepare_batch(imgs: list[np.ndarray], target: int = 518):
    """Grand côté ramené à 518 px puis complété en carré blanc (mode « pad » de VGGT).

    Sans ce padding, VGGT prédit des champs de vision horizontal et vertical
    identiques même sur des images portrait, d'où fx ≠ fy. Sur un carré,
    c'est correct, et on ramène ensuite K dans le repère de l'image d'origine.
    """
    H, W = imgs[0].shape[:2]
    s = target / max(H, W)
    h, w = round(H * s), round(W * s)
    top, left = (target - h) // 2, (target - w) // 2
    canvas = np.full((len(imgs), target, target, 3), 255, np.uint8)
    for c, im in zip(canvas, imgs):
        c[top : top + h, left : left + w] = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    valid = np.zeros((target, target), bool)
    valid[top : top + h, left : left + w] = True
    batch = torch.from_numpy(canvas).permute(0, 3, 1, 2).float() / 255.0
    return batch, valid, (s, top, left)


def estimate_poses(scene_dir: Path, max_frames: int = 40, conf_quantile: float = 0.5,
                   max_points: int = 200_000, device: str = "cuda"):
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    paths = sorted(p for p in (scene_dir / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    if len(paths) > max_frames:
        idx = np.linspace(0, len(paths) - 1, max_frames).round().astype(int)
        paths = [paths[i] for i in idx]
    imgs = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]
    H, W = imgs[0].shape[:2]
    batch, valid, (s, top, left) = _prepare_batch(imgs)
    h, w = batch.shape[-2:]
    print(f"VGGT sur {len(paths)} images ({W}x{H} → {w}x{h} avec padding)")

    model = VGGT.from_pretrained("facebook/VGGT-1B")
    model.point_head = None  # inutile : on reconstruit les points depuis la profondeur
    model.track_head = None
    model = model.half().to(device).eval()

    # Pas de bf16 sur Turing (RTX 20xx) : poids en fp16 + autocast pour les tenseurs créés en interne.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        x = batch.half().to(device)[None]
        tokens, ps_idx = model.aggregator(x)
        pose_enc = model.camera_head(tokens)[-1]
        depth, conf = model.depth_head(tokens, images=x, patch_start_idx=ps_idx)
        if not torch.isfinite(depth).all():
            # Débordement fp16 dans la tête de profondeur : on la repasse en fp32.
            print("Profondeur non finie en fp16, recalcul de la tête en fp32")
            model.depth_head.float()
            tokens = [t.float() if t is not None else None for t in tokens]
            depth, conf = model.depth_head(tokens, images=x.float(), patch_start_idx=ps_idx)
    print(f"Mémoire GPU max : {torch.cuda.max_memory_allocated() / 1e9:.2f} Go")

    extr, intr = pose_encoding_to_extri_intri(pose_enc.float(), (h, w))
    extr, intr = extr[0].cpu().numpy(), intr[0].cpu().numpy()
    depth = depth[0].float().cpu().numpy()
    conf = conf[0].float().cpu().numpy()
    del model, tokens
    torch.cuda.empty_cache()

    # Nuage de points : on rétroprojette les pixels les plus fiables.
    points = unproject_depth_map_to_point_map(depth, extr, intr)
    colors = (batch.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    in_image = np.broadcast_to(valid, conf.shape)
    keep = in_image & (conf >= np.quantile(conf[in_image], conf_quantile))
    xyz, rgb = points[keep], colors[keep]
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]

    # Intrinsèques ramenées à l'image native : u_orig = (u_pad - left) / s.
    K = intr.copy()
    K[:, 0, 2] -= left
    K[:, 1, 2] -= top
    K[:, :2] /= s
    viewmats = np.tile(np.eye(4), (len(paths), 1, 1))
    viewmats[:, :3, :4] = extr

    # Profondeurs gardées pour régulariser l'entraînement (`train --depth-weight`) :
    # on ne garde que la zone utile, sans les bandes de padding.
    rows, cols = np.nonzero(valid.any(1))[0], np.nonzero(valid.any(0))[0]
    crop = (slice(None), slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    np.savez(
        scene_dir / "poses.npz",
        depth=depth[..., 0][crop].astype(np.float16),
        depth_conf=conf[crop].astype(np.float16),
        names=np.array([p.name for p in paths]),
        K=K.astype(np.float32),
        viewmats=viewmats.astype(np.float32),
        width=W,
        height=H,
        xyz=xyz.astype(np.float32),
        rgb=rgb,
    )
    print(f"{len(paths)} caméras, {len(xyz)} points → {scene_dir / 'poses.npz'}")
