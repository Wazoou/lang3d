import torch

from lang3d.query import heatmap


def test_heatmap_goes_from_gray_to_red():
    rgb = torch.tensor([[0.2, 0.6, 0.4]] * 3)
    out = heatmap(rgb, torch.tensor([0.0, 0.5, 1.0]))
    gray = rgb[0].mean() * 0.5
    assert torch.allclose(out[0], gray.expand(3))                 # score 0 : gris
    assert torch.allclose(out[2], torch.tensor([1.0, 0.1, 0.1]))  # score 1 : rouge
    assert out[0, 0] < out[1, 0] < out[2, 0]                      # monotone en rouge
