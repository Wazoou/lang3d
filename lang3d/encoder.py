"""Encodeur image/texte partagé (SigLIP 2 via open_clip).

SigLIP 2 projette images et textes dans le même espace : un crop de chaise
rouge et la phrase « chaise rouge » ont des embeddings proches (cosinus élevé).
Son tokenizer est multilingue, ce qui permet de taper les requêtes en français.
"""
import numpy as np
import open_clip
import torch
import torch.nn.functional as F

DEFAULT_MODEL = ("ViT-B-16-SigLIP2", "webli")


class Encoder:
    def __init__(self, model: tuple[str, str] = DEFAULT_MODEL, device: str = "cuda"):
        self.name = model
        self.device = device
        self.model, _, _ = open_clip.create_model_and_transforms(model[0], pretrained=model[1], device=device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model[0])
        cfg = self.model.visual.preprocess_cfg
        size = cfg["size"]
        self.size = size if isinstance(size, int) else size[0]
        self.mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def encode_images(self, crops: np.ndarray, batch: int = 64) -> torch.Tensor:
        """crops : (B, S, S, 3) uint8 déjà carrés et à la bonne taille → (B, D) normalisés."""
        out = []
        for i in range(0, len(crops), batch):
            x = torch.from_numpy(crops[i : i + batch]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
            x = (x - self.mean) / self.std
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(self.model.encode_image(x).float())
        return F.normalize(torch.cat(out), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> torch.Tensor:
        tokens = self.tokenizer(texts).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            emb = self.model.encode_text(tokens).float()
        return F.normalize(emb, dim=-1)
