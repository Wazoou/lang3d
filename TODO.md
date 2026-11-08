# TODO — Scène 3D interrogeable en langage naturel

Légende : `[x]` fait · `[~]` en cours · `[ ]` à faire

## Phase 0 — Environnement
- [x] Vérifier le matériel : RTX 2070 (8 Go VRAM, Turing sm_75), 31 Go RAM, pas de CUDA Toolkit ni de MSVC
- [x] Trouver une wheel gsplat précompilée Windows → gsplat 1.5.3 / Python 3.10 / torch 2.4 / CUDA 12.4
- [x] Projet déplacé dans `Documents\lang3d` (le chemin d'origine dépassait la limite Windows de 260 caractères et cassait le chargement des DLL CUDA de PyTorch)
- [x] Créer le venv Python 3.10 et installer les dépendances (`requirements.txt`)
- [x] Vérifier que gsplat tourne sur le GPU (rendu de 129 canaux OK)
- [x] Épingler `transformers==4.49.0` (les versions récentes désactivent PyTorch < 2.6)
- [x] Forcer l'UTF-8 dans la console Windows (accents et flèches dans les logs)

## Phase 1 — Vidéo → images
- [~] Extraction des frames (OpenCV), une par fenêtre temporelle — code écrit, à tester sur une vraie vidéo
- [~] Garder la frame la plus nette de chaque fenêtre (variance du Laplacien) — code écrit, à tester

## Phase 2 — Poses de caméra
- [x] ~~COLMAP via pycolmap~~ → abandonné : Windows (Smart App Control) bloque la DLL non signée de pycolmap
- [x] **VGGT** (Meta, CVPR 2025) à la place : poses + profondeur en une passe, PyTorch pur — 50 images en 18 s
- [x] Tenir dans 8 Go : poids en fp16 + autocast (pas de bf16 sur Turing), têtes `point` et `track` retirées → 50 images = 7,1 Go
- [x] Bug fx ≠ fy sur images portrait → mode « pad » (image complétée en carré 518×518) puis K ramené au repère natif
- [x] Nuage de points initial : rétroprojection des profondeurs les plus fiables (50 % meilleures confiances), 200 k points max

## Phase 3 — Gaussian Splatting
- [x] Initialisation des gaussiennes depuis le nuage de points (échelle = distance aux 3 plus proches voisins)
- [x] Boucle d'entraînement gsplat : L1 + D-SSIM, densification, SH progressives — fox : 3 min 43 s, 419 k gaussiennes, PSNR 29,7 dB
- [x] Sauvegarde `splats.pt` et export `.ply` standard (visualisable dans SuperSplat)
- [x] Vérification visuelle rendu vs photo (`docs/rendu_vs_photo.jpg`)

## Phase 4 — Features langage 2D
- [x] Vérifier que SigLIP 2 est multilingue (« renard » ↔ « fox » : 0,95 ; « chaise rouge » ↔ « red chair » : 0,95)
- [x] SAM ViT-B : masques automatiques par image (~83 par image, 3,5 s/image)
- [x] SigLIP 2 : un embedding par masque (crop masqué, carré, 224 px)
- [x] Deux niveaux de carte d'index : plus petit masque (parties) et plus grand masque (objets entiers)
- [x] Zones non couvertes par SAM ajoutées comme masques (fonds texturés)

## Phase 5 — Lifting 2D → 3D (sans entraînement)
- [x] Compression par SVD non centrée : 768 → 128 dims, 98,3 % de l'énergie conservée
- [x] Astuce du gradient : feature d'une gaussienne = moyenne pondérée (α·T) des pixels qu'elle couvre — 8 s par niveau
- [x] Sauvegarde des features par gaussienne et par niveau

## Phase 6 — Requête et visualisation
- [x] Score de pertinence façon LERF (requête vs négatifs "object", "things", "stuff", "texture"), meilleur niveau par requête
- [x] Nettoyage des requêtes en français ("Où est la chaise rouge ?" → "chaise rouge") — testé
- [x] Viewer web (viser) : champ texte, seuil, surbrillance en rouge, mode "isoler" — testé dans le navigateur
- [x] Export `.ply` avec surbrillance (`python -m lang3d query`) — testé (`scenes/fox/gs/query_renard.ply`)

## Phase 7 — Test et démo
- [x] Test de bout en bout sur une scène publique (fox d'instant-ngp, 50 images)
- [ ] Test sur une vidéo filmée au téléphone
- [~] README avec explications, résultats et captures — fait ; GIF de démo à ajouter
- [x] Repo GitHub public pour le CV : https://github.com/Wazoou/lang3d

## Pistes d'amélioration (à ajouter au fil de l'eau)
- [ ] Fonds texturés (papier peint, sol) : encoder la zone non couverte sans trous (inpainting ou crop de la bounding box entière)
- [ ] Requêtes courtes (« les oreilles ») : prompts multiples (« une photo de {} ») et moyenne des embeddings texte
- [ ] Plus de 50 vues : VGGT par blocs qui se chevauchent + alignement Sim(3) entre blocs
- [ ] Rendu de la carte de pertinence en 2D (heatmap) en plus de la surbrillance 3D
- [ ] Évaluation quantitative (mIoU sur le dataset LERF / 3D-OVS)
