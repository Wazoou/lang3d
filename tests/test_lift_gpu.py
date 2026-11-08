import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="gsplat a besoin d'un GPU CUDA")


def test_gradient_trick_recovers_each_gaussian_feature():
    from lang3d.lift import splat_backward

    d = "cuda"
    # Deux gaussiennes opaques, l'une à gauche, l'autre à droite de l'image.
    splats = {
        "means": torch.tensor([[-0.5, 0.0, 3.0], [0.5, 0.0, 3.0]], device=d),
        "quats": torch.tensor([[1.0, 0, 0, 0]] * 2, device=d),
        "scales": torch.log(torch.full((2, 3), 0.12, device=d)),
        "opacities": torch.full((2,), 5.0, device=d),  # sigmoïde(5) ≈ 0,99
    }
    w, h = 64, 32
    K = torch.tensor([[60.0, 0, w / 2], [0, 60.0, h / 2], [0, 0, 1]], device=d)
    V = torch.eye(4, device=d)
    # Carte de sens : vecteur A à gauche, B à droite, + un canal de 1 pour les poids.
    A, B = torch.tensor([1.0, 0, 0], device=d), torch.tensor([0.0, 1, 0], device=d)
    target = torch.zeros(h, w, 4, device=d)
    target[:, : w // 2, :3] = A
    target[:, w // 2:, :3] = B
    target[..., 3] = 1

    acc = splat_backward(splats, V, K, w, h, target)
    feats = acc[:, :3] / acc[:, 3:]
    assert torch.allclose(feats[0], A, atol=1e-3)
    assert torch.allclose(feats[1], B, atol=1e-3)
    assert (acc[:, 3] > 1).all()  # chaque gaussienne couvre plusieurs pixels
