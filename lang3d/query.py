"""Requêtes en langage naturel sur les gaussiennes.

Score de pertinence façon LERF : pour une requête q et des négatifs
canoniques n_j ("object", "things", ...), on calcule pour chaque gaussienne
    min_j  sigmoid(τ · (<f, q> - <f, n_j>))
soit la probabilité que la gaussienne ressemble plus à la requête qu'au
négatif le plus proche. > 0.5 = plus proche de la requête que de « un truc ».
"""
import re
from pathlib import Path

import torch

from .encoder import Encoder
from .gaussians import export_ply, load_splats, rgb_to_sh, sh_to_rgb

NEGATIVES = ["object", "things", "stuff", "texture"]
TEMPERATURE = 10.0

_FILLERS = re.compile(
    r"^(où\s+(est|sont|se\s+trouvent?|se\s+situent?)|montre[-\s]?moi|trouve[-\s]?moi|cherche|"
    r"where\s+(is|are)|find|show\s+me)\s+",
    re.IGNORECASE,
)
_ARTICLES = re.compile(r"^((le|la|les|un|une|des|du|the|a|an)\s+|l['’]\s*)", re.IGNORECASE)


def clean_query(text: str) -> str:
    """« Où est la chaise rouge ? » → « chaise rouge »."""
    t = text.strip().rstrip("?!. ").strip()
    t = _FILLERS.sub("", t)
    t = _ARTICLES.sub("", t)
    return t or text.strip()


class SceneQuery:
    def __init__(self, scene_dir: Path, device: str = "cuda"):
        d = torch.load(scene_dir / "lang" / "gaussians_lang.pt", weights_only=False)
        self.P = d["P"].to(device)
        self.feats = {lvl: f.float().to(device) for lvl, f in d["feats"].items()}
        self.valid = {lvl: v.to(device) for lvl, v in d["valid"].items()}
        self.enc = Encoder(tuple(d["model"]), device)
        self.neg = self.enc.encode_text(NEGATIVES) @ self.P

    def relevancy(self, text: str) -> tuple[torch.Tensor, str]:
        """Pertinence par gaussienne, au niveau de masques le plus pertinent (comme LangSplat)."""
        q = self.enc.encode_text([clean_query(text)]) @ self.P
        best = None
        for lvl, feats in self.feats.items():
            pos = feats @ q.T          # (N, 1)
            neg = feats @ self.neg.T   # (N, len(NEGATIVES))
            rel = torch.sigmoid(TEMPERATURE * (pos - neg)).min(dim=1).values
            rel[~self.valid[lvl]] = 0
            if best is None or rel.max() > best[0].max():
                best = (rel, lvl)
        return best

    def scores(self, text: str) -> tuple[torch.Tensor, str]:
        """Pertinence renormalisée dans [0, 1] : 0 = pas mieux qu'un négatif, 1 = meilleure gaussienne."""
        rel, lvl = self.relevancy(text)
        return ((rel - 0.5) / (rel.max() - 0.5).clamp_min(1e-6)).clamp(0, 1), lvl


def highlight(rgb: torch.Tensor, scores: torch.Tensor, threshold: float):
    """Surligne en rouge les gaussiennes au-dessus du seuil, grise le reste."""
    hit = scores > threshold
    strength = (0.5 + 0.5 * (scores - threshold) / max(1 - threshold, 1e-6)).clamp(0, 1)[:, None]
    red = torch.tensor([1.0, 0.1, 0.1], device=rgb.device)
    gray = rgb.mean(1, keepdim=True).expand(-1, 3) * 0.5
    colors = torch.where(hit[:, None], rgb * (1 - strength) + red * strength, gray)
    return colors, hit


def export_query(scene_dir: Path, text: str, threshold: float = 0.5, out: Path | None = None):
    splats, _, _ = load_splats(scene_dir / "gs" / "splats.pt")
    scores, level = SceneQuery(scene_dir).scores(text)
    colors, hit = highlight(sh_to_rgb(splats["sh0"][:, 0]), scores, threshold)
    # Sans les harmoniques d'ordre supérieur, la couleur ne dépend plus de l'angle : la surbrillance reste visible.
    flat = dict(splats, shN=torch.zeros_like(splats["shN"]))
    slug = re.sub(r"\W+", "_", clean_query(text)).strip("_").lower()
    out = out or scene_dir / "gs" / f"query_{slug}.ply"
    print(f"« {text} » → « {clean_query(text)} » [niveau {level}] : {hit.sum().item()} gaussiennes surlignées")
    export_ply(out, flat, sh0_override=rgb_to_sh(colors)[:, None])
