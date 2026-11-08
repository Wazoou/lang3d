"""Étape 1 : vidéo → images.

On découpe la vidéo en `num_frames` fenêtres régulières et on garde, dans
chaque fenêtre, l'image la plus nette (variance du Laplacien). Le flou de
bougé est l'ennemi numéro un de COLMAP comme du Gaussian Splatting.
"""
from pathlib import Path

import cv2


def sharpness(img) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def resize_max_side(img, max_side: int):
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def extract_frames(video: Path, out_dir: Path, num_frames: int = 40, max_side: int = 1280) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo {video}")
    # Les téléphones stockent souvent la vidéo couchée avec une métadonnée de
    # rotation ; OpenCV 4.11 ne l'applique pas par défaut.
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window = max(1, total // num_frames)
    out_dir.mkdir(parents=True, exist_ok=True)

    best, best_score, saved, idx = None, -1.0, 0, 0
    while saved < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_max_side(frame, max_side)
        score = sharpness(frame)
        if score > best_score:
            best, best_score = frame, score
        idx += 1
        if idx % window == 0:
            cv2.imwrite(str(out_dir / f"{saved:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1
            best, best_score = None, -1.0
    if best is not None and saved < num_frames:
        cv2.imwrite(str(out_dir / f"{saved:05d}.jpg"), best, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved += 1
    cap.release()
    print(f"{saved} images extraites de {total} frames → {out_dir}")
    return saved
