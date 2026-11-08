"""Étape 4 : features langage 2D (approche LangSplat).

Pour chaque image :
  1. SAM découpe l'image en masques d'objets / parties d'objets ;
  2. chaque masque est isolé (fond noir), recadré en carré et encodé par
     SigLIP 2 → un embedding par masque ;
  3. deux cartes d'index (niveaux, comme la hiérarchie de LangSplat) :
       - "small" : chaque pixel pointe vers le plus petit masque qui le couvre
         (parties d'objets : « oreille », « fleur ») ;
       - "large" : vers le plus grand (objets entiers, surfaces : « papier peint »).

On stocke les cartes d'index + une table d'embeddings partagée plutôt qu'une
carte (H, W, 768) par image : c'est des centaines de fois plus léger.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .encoder import Encoder
from .scene import load_scene

SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_sam(device: str):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    ckpt = MODELS_DIR / Path(SAM_URL).name
    if not ckpt.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        torch.hub.download_url_to_file(SAM_URL, str(ckpt))
    sam = sam_model_registry["vit_b"](checkpoint=str(ckpt)).to(device)
    # Réglages proches de LangSplat, sans les crops multi-échelles (trop lents ici).
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.85,
        box_nms_thresh=0.7,
        min_mask_region_area=100,
    )


def leftover_masks(masks: list[dict], h: int, w: int, min_frac: float = 0.01) -> list[dict]:
    """Zones que SAM n'a couvertes par aucun masque (souvent les grands fonds texturés :
    papier peint, sol, ciel). Chaque composante connexe assez grande devient un masque."""
    covered = np.zeros((h, w), bool)
    for m in masks:
        covered |= m["segmentation"]
    n, labels, stats, _ = cv2.connectedComponentsWithStats((~covered).astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        x, y, bw, bh, area = stats[k]
        if area >= min_frac * h * w:
            out.append({"segmentation": labels == k, "area": int(area), "bbox": [x, y, bw, bh]})
    return out


def mask_crops(img: np.ndarray, masks: list[dict], size: int) -> np.ndarray:
    """Un crop carré par masque : pixels hors masque en noir, padding noir."""
    crops = np.zeros((len(masks), size, size, 3), np.uint8)
    for k, m in enumerate(masks):
        x, y, w, h = (int(v) for v in m["bbox"])
        w, h = max(w, 1), max(h, 1)
        crop = img[y : y + h, x : x + w] * m["segmentation"][y : y + h, x : x + w, None]
        side = max(w, h)
        square = np.zeros((side, side, 3), np.uint8)
        oy, ox = (side - h) // 2, (side - w) // 2
        square[oy : oy + h, ox : ox + w] = crop
        crops[k] = cv2.resize(square, (size, size), interpolation=cv2.INTER_AREA)
    return crops


def extract_features(scene_dir: Path, max_side: int = 640, device: str = "cuda"):
    frames, _, _ = load_scene(scene_dir)
    mask_gen = load_sam(device)
    enc = Encoder(device=device)

    W, H = frames[0].width, frames[0].height
    s = max_side / max(W, H)
    w, h = round(W * s), round(H * s)

    names, feats = [], []
    indices = {"small": [], "large": []}
    for f in tqdm(frames, desc="SAM + SigLIP 2"):
        img = cv2.cvtColor(cv2.imread(str(f.path)), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        masks = mask_gen.generate(img)
        masks = sorted(masks + leftover_masks(masks, h, w), key=lambda m: m["area"], reverse=True)
        # Le dernier masque peint l'emporte : du plus grand au plus petit pour "small", l'inverse pour "large".
        for level, order in (("small", range(len(masks))), ("large", reversed(range(len(masks))))):
            index = np.full((h, w), -1, np.int16)
            for k in order:
                index[masks[k]["segmentation"]] = k
            indices[level].append(torch.from_numpy(index))
        emb = enc.encode_images(mask_crops(img, masks, enc.size)) if masks else torch.zeros(0, 1)
        names.append(f.name)
        feats.append(emb.half().cpu())

    out = scene_dir / "lang" / "features.pt"
    out.parent.mkdir(exist_ok=True)
    torch.save({"names": names, "index": indices, "feats": feats, "model": enc.name}, out)
    n_masks = sum(len(x) for x in feats)
    print(f"{n_masks} masques sur {len(frames)} images ({n_masks / len(frames):.0f}/image) → {out}")
