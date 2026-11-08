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
# Texte brut par défaut (comme LERF). L'ensemble de prompts façon CLIP, testé sur
# fox, aide les objets entiers (« rideau rouge » : 168 → 1080 gaussiennes) mais
# noie les parties (« oreille du renard » finit par surligner tout le renard).
TEMPLATES = ["{}"]
OBJECT_TEMPLATES = ["{}", "une photo de {}", "a photo of {}"]

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
        self.neg = {
            False: self.embed(NEGATIVES, TEMPLATES),
            True: self.embed(NEGATIVES, OBJECT_TEMPLATES),
        }

    def embed(self, texts: list[str], templates: list[str]) -> torch.Tensor:
        """Embedding moyen sur les templates, projeté dans la base SVD → (len(texts), dim)."""
        prompts = [t.format(x) for x in texts for t in templates]
        emb = self.enc.encode_text(prompts).view(len(texts), len(templates), -1).mean(1)
        return torch.nn.functional.normalize(emb, dim=-1) @ self.P

    def relevancy(self, text: str, level: str | None = None, ensemble: bool = False) -> tuple[torch.Tensor, str]:
        """Pertinence par gaussienne, au niveau imposé ou, par défaut, au niveau de masques
        le plus pertinent (comme LangSplat). `ensemble` active OBJECT_TEMPLATES."""
        q = self.embed([clean_query(text)], OBJECT_TEMPLATES if ensemble else TEMPLATES)
        negs = self.neg[ensemble]
        best = None
        for lvl in [level] if level else self.feats:
            feats = self.feats[lvl]
            pos = feats @ q.T          # (N, 1)
            neg = feats @ negs.T       # (N, len(NEGATIVES))
            rel = torch.sigmoid(TEMPERATURE * (pos - neg)).min(dim=1).values
            rel[~self.valid[lvl]] = 0
            if best is None or rel.max() > best[0].max():
                best = (rel, lvl)
        return best

    def similar(self, idx: int, level: str = "large") -> torch.Tensor:
        """Ressemblance de chaque gaussienne à la gaussienne `idx` (requête par l'exemple), dans [0, 1].

        0 = pas plus proche que la gaussienne médiane de la scène, 1 = identique."""
        feats = self.feats[level]
        cos = feats @ feats[idx]
        valid = self.valid[level]
        base = cos[valid].median()
        sim = ((cos - base) / (1 - base).clamp_min(1e-6)).clamp(0, 1)
        sim[~valid] = 0
        return sim

    def scores(self, text: str, level: str | None = None, ensemble: bool = False) -> tuple[torch.Tensor, str]:
        """Pertinence renormalisée dans [0, 1] : 0 = pas mieux qu'un négatif, 1 = meilleure gaussienne."""
        rel, lvl = self.relevancy(text, level, ensemble)
        return ((rel - 0.5) / (rel.max() - 0.5).clamp_min(1e-6)).clamp(0, 1), lvl


def highlight(rgb: torch.Tensor, scores: torch.Tensor, threshold: float):
    """Surligne en rouge les gaussiennes au-dessus du seuil, grise le reste."""
    hit = scores > threshold
    strength = (0.5 + 0.5 * (scores - threshold) / max(1 - threshold, 1e-6)).clamp(0, 1)[:, None]
    red = torch.tensor([1.0, 0.1, 0.1], device=rgb.device)
    gray = rgb.mean(1, keepdim=True).expand(-1, 3) * 0.5
    colors = torch.where(hit[:, None], rgb * (1 - strength) + red * strength, gray)
    return colors, hit


def heatmap(rgb: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Carte de chaleur sans seuil : du gris (score 0) au rouge (score 1), une seule teinte."""
    gray = rgb.mean(1, keepdim=True).expand(-1, 3) * 0.5
    red = torch.tensor([1.0, 0.1, 0.1], device=rgb.device)
    s = scores.clamp(0, 1)[:, None] ** 0.7  # un peu de contraste pour les scores faibles
    return gray * (1 - s) + red * s


def export_query(scene_dir: Path, text: str, threshold: float = 0.5, out: Path | None = None,
                 level: str | None = None, ensemble: bool = False):
    splats, _, _ = load_splats(scene_dir / "gs" / "splats.pt")
    scores, level = SceneQuery(scene_dir).scores(text, level, ensemble)
    colors, hit = highlight(sh_to_rgb(splats["sh0"][:, 0]), scores, threshold)
    # Sans les harmoniques d'ordre supérieur, la couleur ne dépend plus de l'angle : la surbrillance reste visible.
    flat = dict(splats, shN=torch.zeros_like(splats["shN"]))
    slug = re.sub(r"\W+", "_", clean_query(text)).strip("_").lower()
    out = out or scene_dir / "gs" / f"query_{slug}.ply"
    print(f"« {text} » → « {clean_query(text)} » [niveau {level}] : {hit.sum().item()} gaussiennes surlignées")
    export_ply(out, flat, sh0_override=rgb_to_sh(colors)[:, None])
