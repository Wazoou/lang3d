import torch

from lang3d.viewer import pick_gaussian


def test_pick_returns_closest_opaque_gaussian_on_the_ray():
    centers = torch.tensor([[0.0, 0, 5], [0.0, 0, 2], [0.0, 0, 1], [3.0, 0, 1]])
    radius = torch.full((4,), 0.2)
    opacity = torch.tensor([0.9, 0.9, 0.05, 0.9])  # la plus proche sur le rayon est transparente
    j = pick_gaussian(torch.zeros(3), torch.tensor([0.0, 0, 1]), centers, radius, opacity)
    assert j == 1


def test_pick_returns_none_when_nothing_is_hit():
    centers = torch.tensor([[2.0, 0, 5]])
    assert pick_gaussian(torch.zeros(3), torch.tensor([0.0, 0, 1]), centers, torch.tensor([0.1]),
                         torch.tensor([1.0])) is None
