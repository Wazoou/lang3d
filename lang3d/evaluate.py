"""Évaluation sur le benchmark LERF (version annotée « lerf_ovs » de LangSplat).

Reproduit le protocole de `LangSplat/eval/evaluate_iou_loc.py` :
  - on rend, depuis chaque vue annotée, la carte de features de chaque niveau
    et on calcule la pertinence LERF de chaque pixel pour chaque requête ;
  - mIoU : carte lissée (moyenne avec un flou 30×30), normalisée min-max,
    ramenée de [-1, 1] à [0, 1], seuillée à 0.4, vote majoritaire 7×7 ;
    on garde le niveau dont la carte lissée a le plus grand maximum ;
  - localisation : le maximum de la carte floutée tombe-t-il dans une boîte
    annotée ? (niveau choisi de la même façon sur la carte floutée)
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .gaussians import load_splats, render
from .query import OBJECT_TEMPLATES, TEMPERATURE, TEMPLATES, SceneQuery
from .scene import load_scene

BLUR = np.ones((30, 30), np.float32) / 900


def load_labels(label_dir: Path) -> dict[str, tuple[int, int, dict]]:
    """{nom d'image: (h, w, {requête: {"mask": union des polygones, "boxes": [x1 y1 x2 y2]}})}"""
    out = {}
    for path in sorted(label_dir.glob("frame_*.json")):
        d = json.loads(path.read_text())
        h, w = d["info"]["height"], d["info"]["width"]
        ann = {}
        for obj in d["objects"]:
            a = ann.setdefault(obj["category"], {"mask": np.zeros((h, w), np.uint8), "boxes": []})
            poly = np.zeros((h, w), np.uint8)
            cv2.fillPoly(poly, [np.asarray(obj["segmentation"], np.int32)], 1)
            a["mask"] |= poly
            a["boxes"].append(np.asarray(obj["bbox"], np.float32).reshape(4))
        out[d["info"]["name"]] = (h, w, ann)
    return out


def majority_smooth(mask: np.ndarray, r: int = 3) -> np.ndarray:
    """Vote majoritaire sur une fenêtre 7×7 (version vectorisée du `smooth` de LangSplat).

    Leur boucle borne les fenêtres à `h-1` / `w-1` exclus : la dernière ligne et la
    dernière colonne ne votent jamais. On reproduit ce détail pour des scores identiques."""
    k = np.ones((2 * r + 1, 2 * r + 1), np.float32)
    voters = np.ones(mask.shape, np.float32)
    voters[-1, :] = 0
    voters[:, -1] = 0
    ones = cv2.filter2D(mask.astype(np.float32) * voters, -1, k, borderType=cv2.BORDER_CONSTANT)
    count = cv2.filter2D(voters, -1, k, borderType=cv2.BORDER_CONSTANT)
    return (ones > count / 2).astype(np.uint8)


def iou_and_score(rel: np.ndarray, gt: np.ndarray, thresh: float) -> tuple[float, float]:
    v = 0.5 * (cv2.filter2D(rel, -1, BLUR) + rel)
    out = v - v.min()
    out = np.clip(out / (out.max() + 1e-9) * 2 - 1, 0, 1)
    pred = majority_smooth(out > thresh)
    union = np.logical_or(gt, pred).sum()
    return float(np.logical_and(gt, pred).sum() / max(union, 1)), float(v.max())


def localization(rel: np.ndarray) -> tuple[np.ndarray, float]:
    """Coordonnées (x, y) des maxima de la carte floutée, et la valeur de ce maximum."""
    avg = cv2.filter2D(rel, -1, BLUR)
    ys, xs = np.nonzero(avg == avg.max())
    return np.stack([xs, ys], 1), float(avg.max())


@torch.no_grad()
def evaluate_scene(scene_dir: Path, label_dir: Path, thresh: float = 0.4, ensemble: bool = False,
                   device: str = "cuda") -> dict:
    splats, _, _ = load_splats(scene_dir / "gs" / "splats.pt", device)
    frames = {f.name: f for f in load_scene(scene_dir)[0]}
    q = SceneQuery(scene_dir, device)
    templates = OBJECT_TEMPLATES if ensemble else TEMPLATES
    negs = q.neg[ensemble]

    ious, hits, per_query = [], 0, []
    for name, (h, w, ann) in load_labels(label_dir).items():
        f = frames[name]
        K = torch.from_numpy(f.K).float().to(device)
        K[0] *= w / f.width
        K[1] *= h / f.height
        V = torch.from_numpy(f.viewmat).float().to(device)
        # Carte de features par niveau, normalisée pixel par pixel.
        fmaps = {}
        for lvl, feats in q.feats.items():
            out, _, _ = render(splats, V, K, w, h, colors=feats * q.valid[lvl][:, None])
            fmaps[lvl] = F.normalize(out[0], dim=-1)

        labels = list(ann)
        pos = q.embed(labels, templates)  # (n_labels, dim)
        for k, label in enumerate(labels):
            rels = {}
            for lvl, fm in fmaps.items():
                s_pos = fm @ pos[k]                      # (h, w)
                s_neg = fm @ negs.T                      # (h, w, n_neg)
                rels[lvl] = torch.sigmoid(TEMPERATURE * (s_pos[..., None] - s_neg)).min(-1).values.cpu().numpy()

            gt = ann[label]["mask"]
            iou, _ = max((iou_and_score(r, gt, thresh) for r in rels.values()), key=lambda t: t[1])
            coords, _ = max((localization(r) for r in rels.values()), key=lambda t: t[1])
            hit = any(
                ((coords[:, 0] >= min(x1, x2)) & (coords[:, 0] <= max(x1, x2))
                 & (coords[:, 1] >= min(y1, y2)) & (coords[:, 1] <= max(y1, y2))).any()
                for x1, y1, x2, y2 in ann[label]["boxes"]
            )
            ious.append(iou)
            hits += hit
            per_query.append({"image": name, "query": label, "iou": round(iou, 4), "localized": bool(hit)})

    return {"miou": float(np.mean(ious)), "loc_acc": hits / len(ious), "n": len(ious), "queries": per_query}


def evaluate(root: Path, scenes: list[str], thresh: float = 0.4, ensemble: bool = False):
    results = {}
    for s in scenes:
        r = evaluate_scene(root / s, root / "label" / s, thresh, ensemble)
        results[s] = r
        print(f"{s:15s} mIoU {r['miou'] * 100:5.1f}   localisation {r['loc_acc'] * 100:5.1f}   ({r['n']} requêtes)")
    miou = np.mean([r["miou"] for r in results.values()])
    acc = np.mean([r["loc_acc"] for r in results.values()])
    print(f"{'moyenne':15s} mIoU {miou * 100:5.1f}   localisation {acc * 100:5.1f}")
    out = root / ("eval_results_objet_entier.json" if ensemble else "eval_results.json")
    out.write_text(json.dumps({"mean": {"miou": miou, "loc_acc": acc}, "scenes": results}, indent=1))
    print(f"→ {out}")
