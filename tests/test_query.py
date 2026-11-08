import torch

from lang3d.query import clean_query, highlight


def test_clean_query_french():
    assert clean_query("Où est la chaise rouge ?") == "chaise rouge"
    assert clean_query("où sont les lampes") == "lampes"
    assert clean_query("Montre-moi l'ordinateur") == "ordinateur"
    assert clean_query("trouve moi une plante verte") == "plante verte"


def test_clean_query_keeps_words_starting_like_articles():
    # « la » ne doit pas être retiré de « lampe »
    assert clean_query("lampe") == "lampe"
    assert clean_query("la lampe") == "lampe"


def test_clean_query_english():
    assert clean_query("Where is the red chair?") == "red chair"


def test_highlight_only_above_threshold():
    rgb = torch.full((3, 3), 0.5)
    scores = torch.tensor([0.1, 0.6, 1.0])
    colors, hit = highlight(rgb, scores, 0.5)
    assert hit.tolist() == [False, True, True]
    # au-dessus du seuil : plus rouge que vert ; en dessous : gris
    assert colors[2, 0] > colors[2, 1]
    assert torch.allclose(colors[0], colors[0, :1].expand(3))
