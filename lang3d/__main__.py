"""Ligne de commande : `python -m lang3d <étape> <dossier_scène> [options]`.

Pipeline complet :
    frames   vidéo → images nettes
    poses    caméras + nuage de points (VGGT)
    train    Gaussian Splatting (gsplat)
    features masques SAM + embeddings SigLIP 2 par image
    lift     report des embeddings 2D sur les gaussiennes 3D
    view     viewer web interactif (requêtes en langage naturel)
    query    export .ply avec la requête surlignée
"""
import argparse
import sys
from pathlib import Path


def main():
    # La console Windows est en cp1252 par défaut : on force l'UTF-8 pour les accents et flèches.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="lang3d")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("frames", help="vidéo → images")
    s.add_argument("scene", type=Path)
    s.add_argument("--video", type=Path, required=True)
    s.add_argument("--num-frames", type=int, default=40)
    s.add_argument("--max-side", type=int, default=1280)

    s = sub.add_parser("poses", help="caméras + points avec VGGT")
    s.add_argument("scene", type=Path)
    s.add_argument("--max-frames", type=int, default=40)

    s = sub.add_parser("colmap", help="importer des poses COLMAP existantes (au lieu de VGGT)")
    s.add_argument("scene", type=Path)
    s.add_argument("--sparse", default="sparse/0")

    s = sub.add_parser("train", help="entraînement Gaussian Splatting")
    s.add_argument("scene", type=Path)
    s.add_argument("--steps", type=int, default=7000)
    s.add_argument("--downscale", type=int, default=1)
    s.add_argument("--test-every", type=int, default=0,
                   help="met de côté 1 vue sur N pour mesurer PSNR/SSIM (modèle rangé dans gs_eval*/)")
    s.add_argument("--depth-weight", type=float, default=None,
                   help="poids de la régularisation par la profondeur VGGT (défaut : 0,1 si disponible ; 0 = désactivée)")

    s = sub.add_parser("features", help="masques SAM + embeddings SigLIP 2")
    s.add_argument("scene", type=Path)
    s.add_argument("--max-side", type=int, default=640)

    s = sub.add_parser("lift", help="embeddings 2D → gaussiennes 3D")
    s.add_argument("scene", type=Path)
    s.add_argument("--dim", type=int, default=128)

    s = sub.add_parser("view", help="viewer web interactif")
    s.add_argument("scene", type=Path)
    s.add_argument("--port", type=int, default=8080)

    s = sub.add_parser("query", help="export .ply avec la requête surlignée")
    s.add_argument("scene", type=Path)
    s.add_argument("text")
    s.add_argument("--threshold", type=float, default=0.5)
    s.add_argument("--out", type=Path, default=None)
    s.add_argument("--level", choices=["small", "large"], default=None,
                   help="niveau de masques (défaut : le plus pertinent)")
    s.add_argument("--objet-entier", action="store_true",
                   help="ensemble de prompts : mieux pour les objets, moins bien pour les parties")

    s = sub.add_parser("evaluate", help="mIoU et localisation sur le benchmark LERF (lerf_ovs)")
    s.add_argument("root", type=Path, help="dossier lerf_ovs (contient label/ et une scène par dossier)")
    s.add_argument("--scenes", nargs="+", default=["figurines", "ramen", "teatime", "waldo_kitchen"])
    s.add_argument("--threshold", type=float, default=0.4)
    s.add_argument("--objet-entier", action="store_true")

    s = sub.add_parser("export-web", help="démo web statique (GitHub Pages) avec requêtes précalculées")
    s.add_argument("scene", type=Path)
    s.add_argument("queries", nargs="+", help="requêtes ; suffixe * = option « objet entier »")
    s.add_argument("--out", type=Path, default=Path("docs/demo"))
    s.add_argument("--max-gaussians", type=int, default=250_000)

    s = sub.add_parser("demo", help="GIF de démo le long de la trajectoire filmée")
    s.add_argument("scene", type=Path)
    s.add_argument("text")
    s.add_argument("--out", type=Path, default=Path("docs/demo.gif"))

    a = p.parse_args()
    if a.cmd == "frames":
        from .frames import extract_frames
        extract_frames(a.video, a.scene / "images", a.num_frames, a.max_side)
    elif a.cmd == "poses":
        from .poses import estimate_poses
        estimate_poses(a.scene, a.max_frames)
    elif a.cmd == "colmap":
        from .colmap import import_colmap
        import_colmap(a.scene, a.sparse)
    elif a.cmd == "train":
        from .train import train
        train(a.scene, a.steps, a.downscale, test_every=a.test_every, depth_weight=a.depth_weight)
    elif a.cmd == "features":
        from .features import extract_features
        extract_features(a.scene, a.max_side)
    elif a.cmd == "lift":
        from .lift import lift_features
        lift_features(a.scene, a.dim)
    elif a.cmd == "view":
        from .viewer import run_viewer
        run_viewer(a.scene, a.port)
    elif a.cmd == "query":
        from .query import export_query
        export_query(a.scene, a.text, a.threshold, a.out, a.level, a.objet_entier)
    elif a.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(a.root, a.scenes, a.threshold, a.objet_entier)
    elif a.cmd == "export-web":
        from .web_export import export_web
        export_web(a.scene, a.queries, a.out, a.max_gaussians)
    elif a.cmd == "demo":
        from .demo import make_demo
        make_demo(a.scene, a.text, a.out)


if __name__ == "__main__":
    main()
