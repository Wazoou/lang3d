# lang3d — Scène 3D interrogeable en langage naturel

À partir d'une simple vidéo filmée au téléphone, lang3d reconstruit la scène en
**Gaussian Splatting** et attache à chaque gaussienne un **embedding
vision-langage** (SigLIP 2). On peut ensuite taper « où est la chaise rouge ? »
dans un viewer web et voir l'objet surligné en 3D.

Approche inspirée de [LERF](https://www.lerf.io/) et
[LangSplat](https://langsplat.github.io/), adaptée pour tourner sur un GPU
grand public (RTX 2070, 8 Go).

![Viewer : « Où est le renard ? »](docs/viewer_renard.jpg)

## Résultats sur la scène « fox » (instant-ngp, 50 photos)

| Étape | Temps (RTX 2070) | Résultat |
|---|---|---|
| Poses VGGT | 18 s | 50 caméras, 200 k points, 7,1 Go VRAM max |
| Gaussian Splatting | 3 min 43 s (7000 itérations, 540×960) | 419 k gaussiennes, PSNR entraînement 29,7 dB |
| SAM + SigLIP 2 | 3 min | 4127 masques (83 par image) |
| Lifting 2D → 3D | 16 s (2 niveaux) | 402 k gaussiennes avec une feature, SVD 768 → 128 dims (98,3 % de l'énergie) |
| Requête | 0,02 à 0,3 s | |

Rendu (droite) vs photo d'origine (gauche) :

![Rendu vs photo](docs/rendu_vs_photo.jpg)

Requêtes en français (gaussiennes au-dessus du seuil en rouge) :

![Requêtes](docs/requetes_fox.jpg)

- ✅ « renard », « rideau rouge », « oreille du renard » (pointes des oreilles), « cadre photo » (en bord de vue)
- ❌ « papier peint à fleurs », « plaque en bois » : seuls quelques fragments ressortent (voir limites)

## Pipeline

```
vidéo ──► frames ──► VGGT ──► Gaussian Splatting ──┐
                    (poses +    (gsplat)           │
                    points)                         ▼
          frames ──► SAM ──► SigLIP 2 ──► lifting 2D→3D ──► viewer web + requêtes texte
                    (masques)  (1 embedding    (astuce du gradient)
                                par masque)
```

| Étape | Module | Ce qui se passe |
|---|---|---|
| 1. `frames` | [frames.py](lang3d/frames.py) | Découpe la vidéo, garde l'image la plus nette de chaque fenêtre (variance du Laplacien). |
| 2. `poses` | [poses.py](lang3d/poses.py) | **VGGT** (Meta, CVPR 2025) prédit en une passe les caméras et une profondeur par image ; on en tire le nuage de points initial. |
| 3. `train` | [train.py](lang3d/train.py) | **3D Gaussian Splatting** avec gsplat : perte L1 + D-SSIM, densification adaptative, harmoniques sphériques progressives. |
| 4. `features` | [features.py](lang3d/features.py) | **SAM** découpe chaque image en masques (+ les zones non couvertes) ; chaque masque est encodé par **SigLIP 2** (multilingue). Deux niveaux : plus petit masque par pixel (parties) et plus grand (objets entiers). |
| 5. `lift` | [lift.py](lang3d/lift.py) | Report des embeddings sur les gaussiennes **sans entraînement** (voir ci-dessous). |
| 6. `view` / `query` | [viewer.py](lang3d/viewer.py), [query.py](lang3d/query.py) | Score de pertinence façon LERF, surbrillance dans un viewer web (viser) ou export `.ply`. |

## Les idées clés

**Lifting 2D → 3D par le gradient.** Le rendu d'un pixel est linéaire en la
« couleur » des gaussiennes : `rendu(p) = Σᵢ wᵢₚ · cᵢ` avec `wᵢₚ = αᵢ · Tᵢ`.
Le gradient de `Σₚ ⟨rendu(p), F(p)⟩` par rapport à `cᵢ` vaut donc exactement
`Σₚ wᵢₚ · F(p)` : la moyenne (non normalisée) des features des pixels que la
gaussienne couvre. **Un seul rendu + backward par image** suffit, là où LangSplat
entraîne un autoencodeur puis optimise les features pendant des milliers d'itérations.

**Compression qui préserve les requêtes.** Les embeddings (768 dims) sont
projetés sur une base obtenue par SVD *non centrée* des embeddings de la
scène. Le produit scalaire avec le texte est préservé (`⟨t, f⟩ ≈ ⟨Pᵀt, Pᵀf⟩`),
contrairement à l'autoencodeur 3D de LangSplat.

**Pertinence relative (LERF).** Un cosinus brut ne veut rien dire seul. On
compare la requête à des négatifs génériques (« object », « things »,
« stuff », « texture ») : `min_j sigmoid(τ · (⟨f,q⟩ − ⟨f,n_j⟩))`.

**Requêtes en français.** SigLIP 2 a un tokenizer multilingue
(similarité « renard » / « fox » : 0,95), et les tournures du type
« où est la… ? » sont retirées avant l'encodage.

## Installation (Windows, GPU NVIDIA)

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --index-strategy unsafe-best-match
uv pip install --python .venv/Scripts/python.exe --no-deps "vggt @ git+https://github.com/facebookresearch/vggt"
```

Python 3.10 est imposé par la wheel gsplat précompilée pour Windows (pas besoin
du CUDA Toolkit ni de Visual Studio).

## Utilisation

```bash
python -m lang3d frames   scenes/salon --video salon.mp4
python -m lang3d poses    scenes/salon
python -m lang3d train    scenes/salon
python -m lang3d features scenes/salon
python -m lang3d lift     scenes/salon
python -m lang3d view     scenes/salon        # → http://localhost:8080
python -m lang3d query    scenes/salon "où est la chaise rouge ?"   # export .ply
```

Pour un dossier d'images existant, mettre les images dans `scenes/<nom>/images/`
et commencer à l'étape `poses`.

Conseils de capture : 30 à 60 s, tourner lentement autour de la scène, éviter
le flou, les reflets et les zones très sombres.

## Limites connues

- VGGT tient environ 40 à 50 images sur 8 Go de VRAM : moins de vues que COLMAP, donc un splatting moins net hors des points de vue filmés.
- Grands fonds texturés (papier peint, sol) : SAM ne les segmente pas d'un bloc. Les zones non couvertes sont ajoutées comme masques, mais ces crops troués ne ressemblent pas à « du papier peint » pour SigLIP.
- Objets en grande partie cachés (la plaque derrière la tête du renard) : leurs crops sont trop petits pour être reconnus.
- Requêtes très courtes et sans contexte (« les oreilles ») : moins fiables que « oreille du renard ».
- COLMAP n'est pas utilisable ici : Smart App Control (Windows) bloque la DLL non signée de pycolmap.
- Poids VGGT-1B sous licence non commerciale.

Suivi d'avancement : [TODO.md](TODO.md).
