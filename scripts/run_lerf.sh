#!/usr/bin/env bash
# Reproduit l'évaluation sur le benchmark LERF (version annotée lerf_ovs de LangSplat).
# Prérequis : data/lerf_ovs/ extrait depuis le lien Google Drive du README de LangSplat.
set -euo pipefail
PY=${PY:-.venv/Scripts/python.exe}
SCENES=${SCENES:-"figurines ramen teatime waldo_kitchen"}

for s in $SCENES; do
  dir=data/lerf_ovs/$s
  echo "=== $s"
  [ -f "$dir/poses.npz" ]                || $PY -m lang3d colmap "$dir"
  [ -f "$dir/gs/splats.pt" ]             || $PY -m lang3d train "$dir"
  [ -f "$dir/lang/features.pt" ]         || $PY -m lang3d features "$dir"
  [ -f "$dir/lang/gaussians_lang.pt" ]   || $PY -m lang3d lift "$dir"
done

$PY -m lang3d evaluate data/lerf_ovs --scenes $SCENES
