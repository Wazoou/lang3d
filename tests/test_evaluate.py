import numpy as np

from lang3d.evaluate import iou_and_score, localization, majority_smooth


def smooth_reference(mask):
    """Copie exacte du `smooth` de LangSplat/eval/utils.py (boucle Python)."""
    h, w = mask.shape[:2]
    out = mask.copy()
    for i in range(h):
        for j in range(w):
            sq = mask[max(0, i - 3): min(i + 4, h - 1), max(0, j - 3): min(j + 4, w - 1)]
            out[i, j] = np.argmax(np.bincount(sq.reshape(-1)))
    return out


def test_majority_smooth_matches_langsplat_exactly():
    rng = np.random.default_rng(0)
    for p in (0.3, 0.5, 0.7):
        m = (rng.random((40, 50)) < p).astype(np.uint8)
        np.testing.assert_array_equal(majority_smooth(m), smooth_reference(m))


def test_iou_perfect_prediction():
    gt = np.zeros((100, 100), np.uint8)
    gt[30:70, 30:70] = 1
    rel = np.where(gt, 0.9, 0.1).astype(np.float32)
    iou, _ = iou_and_score(rel, gt, 0.4)
    assert iou > 0.85  # < 1 : le lissage 30×30 du protocole arrondit les coins du masque


def test_localization_finds_peak():
    # Objet loin des bords : le flou d'OpenCV reflète l'image sur ses bords (BORDER_REFLECT_101,
    # comme chez LangSplat), ce qui décale le maximum d'un objet collé au bord vers ce bord.
    rel = np.zeros((80, 80), np.float32)
    rel[35:45, 30:40] = 1
    coords, _ = localization(rel)
    x, y = coords.mean(0)
    assert 30 <= x <= 40 and 35 <= y <= 45
