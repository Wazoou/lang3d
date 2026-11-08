import cv2
import numpy as np

from lang3d.frames import extract_frames, sharpness
from lang3d.poses import _prepare_batch


def textured(seed, h=120, w=160):
    return (np.random.default_rng(seed).random((h, w, 3)) * 255).astype(np.uint8)


def test_sharpness_prefers_sharp_image():
    img = textured(0)
    assert sharpness(img) > 5 * sharpness(cv2.GaussianBlur(img, (0, 0), 3))


def test_extract_frames_keeps_the_sharp_copy(tmp_path):
    video = tmp_path / "v.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    sharp_pos = [2, 0, 3]
    originals = [textured(k) for k in range(3)]
    for img, k in zip(originals, sharp_pos):
        for j in range(4):
            vw.write(img if j == k else cv2.GaussianBlur(img, (0, 0), 4))
    vw.release()
    n = extract_frames(video, tmp_path / "out", num_frames=3, max_side=1280)
    assert n == 3
    for k, orig in enumerate(originals):
        got = cv2.imread(str(tmp_path / "out" / f"{k:05d}.jpg"))
        assert sharpness(got) > 0.5 * sharpness(orig)  # la copie nette, pas une floue


def test_padding_geometry_for_portrait_images():
    imgs = [np.zeros((1920, 1080, 3), np.uint8)]
    batch, valid, (s, top, left) = _prepare_batch(imgs)
    assert batch.shape[-2:] == (518, 518)
    assert top == 0 and left > 0
    # la zone valide a le rapport d'aspect de l'image d'origine
    rows, cols = np.nonzero(valid)
    assert abs((cols.max() - cols.min() + 1) / (rows.max() - rows.min() + 1) - 1080 / 1920) < 0.01
    # un pixel de l'image d'origine retombe au bon endroit : u_pad = u * s + left
    assert np.isclose(1080 * s + left, left + valid[0].sum(), atol=1)
