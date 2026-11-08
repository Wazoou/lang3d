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
- [x] Extraction des frames (OpenCV), une par fenêtre temporelle — testé sur une vidéo synthétique (300 frames → 50)
- [x] Garder la frame la plus nette de chaque fenêtre (variance du Laplacien) — 50/50 frames nettes retrouvées parmi 5 copies floutées
- [x] Bug trouvé sur une vraie vidéo : frames couchées, car OpenCV 4.11 n'applique pas la métadonnée de rotation des téléphones → `CAP_PROP_ORIENTATION_AUTO`
- [x] Vraie vidéo de téléphone (« bottle », 4K, 24,6 s, tour complet) : 40 frames 720×1280 en 22 s

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
- [x] Choix du niveau de masques (auto / small / large) dans le viewer et `--level` en CLI
- [x] Option « objet entier » (ensemble de prompts) dans le viewer et `--objet-entier` en CLI — testé dans le navigateur

## Phase 7 — Test et démo
- [x] Test de bout en bout sur une scène publique (fox d'instant-ngp, 50 images)
- [x] Test sur une vidéo filmée au téléphone (« bottle », 40 vues, tour complet) : VGGT 6,6 Go, 3DGS 5 min 17 s, 797 k gaussiennes, PSNR 35,6 dB
  - ✅ « gourde violette » (isolée parfaitement), « chaussures », « couvercle »
  - ❌ « carrelage », « parquet », « mur blanc » (surfaces) ; « bouchon » (compris comme un bouchon de liège)
- [x] README avec explications, résultats, captures et tableau « ce qui a été essayé »
- [x] GIF de démo (`python -m lang3d demo`) : trajectoire interpolée, 98 images, 3,8 Mo
- [x] Repo GitHub public pour le CV : https://github.com/Wazoou/lang3d

## Phase 8 — Évaluation chiffrée (benchmark LERF)
- [x] Dataset `lerf_ovs` (LERF annoté par LangSplat) : 4 scènes, 22 vues annotées
- [x] Lecteur COLMAP binaire en Python pur (`python -m lang3d colmap`) pour réutiliser les poses du benchmark
- [x] Protocole de LangSplat reproduit à l'identique (lissage vérifié identique à 100 % sur des masques aléatoires)
- [x] Chiffres de référence vérifiés dans le papier LangSplat (tableaux 1 et 2)
- [x] figurines : localisation 83,9 % / IoU 49,0 % (LERF 75,0 / 38,6 ; LangSplat 80,4 / 44,7)
- [x] Contrôle avec requêtes mélangées : 1,8 % / 1,7 % → l'évaluation est saine
- [x] Variante *objet entier* : IoU 52,9 % (indicatif, non retenue comme chiffre principal)
- [ ] Évaluer les 3 autres scènes (ramen, teatime, waldo_kitchen) pour une moyenne comparable aux papiers
- [x] Mesurer la qualité RGB sur des vues mises de côté (`train --test-every 8`) — fox : PSNR test 25,0 dB contre 31,2 dB en entraînement
- [~] Régularisation par la profondeur VGGT (`train --depth-weight`) : +0,18 dB au poids 0,1 sur un seul run → répétitions en cours pour mesurer le bruit

## Phase 9 — Qualité logicielle et démo publique
- [x] 19 tests pytest (requêtes, COLMAP, protocole d'évaluation, frames, padding, clic, format .splat, astuce du gradient sur GPU)
- [x] GitHub Actions : tests sur Linux CPU à chaque push
- [x] Viewer : « Sélectionner en cliquant » — requête par l'exemple à partir de la gaussienne cliquée
- [x] Démo web statique (`export-web`, gsplat.js) : scène fox + 5 requêtes précalculées, 9,3 Mo, testée dans le navigateur
- [ ] Activer GitHub Pages (Settings → Pages → `main` / `docs`) pour publier la démo

## Pistes d'amélioration (à ajouter au fil de l'eau)
- [ ] Objets fins (pieds de chaise) et petits objets (« pirate hat ») : principales erreurs sur figurines
- [x] ~~Fonds texturés : carrés intérieurs au lieu du crop troué~~ → testé, aucun gain (le fond reste au niveau du négatif « texture »), retiré
- [x] ~~Requêtes courtes : ensemble de prompts~~ → testé : aide les objets (« rideau rouge » 168 → 1080) mais noie les parties (« oreille du renard » → tout le renard) ; gardé en option
- [x] ~~Fonds texturés : retirer « texture » des négatifs~~ → testé sur bottle, aucun gain visible sur « carrelage », « parquet », « mur blanc »
- [ ] Fonds texturés, piste suivante : vérifier en 2D si SAM produit un masque propre pour le sol, sinon segmenter les surfaces planes à part (plans estimés depuis la profondeur VGGT)
- [ ] Requêtes courtes, piste suivante : compléter automatiquement le contexte de la scène (« oreilles » → « oreilles du renard ») avec l'objet le plus pertinent
- [ ] Plus de 50 vues : VGGT par blocs qui se chevauchent + alignement Sim(3) entre blocs
- [ ] Rendu de la carte de pertinence en 2D (heatmap) en plus de la surbrillance 3D
- [x] ~~Évaluation quantitative~~ → faite sur LERF figurines (Phase 8)
