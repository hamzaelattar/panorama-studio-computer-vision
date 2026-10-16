# Computer Vision — Panorama Studio

Application Streamlit pédagogique et fonctionnelle de création de panoramas. Elle exécute et compare trois chaînes de détection/description :

- **SIFT** : SIFT + descripteur flottant 128D, distance L2 ;
- **FAST + BRIEF** : FAST détecte, BRIEF décrit, distance de Hamming ;
- **ORB** : FAST multi-échelle + rBRIEF, Hamming (ou Hamming2 lorsque `WTA_K` vaut 3 ou 4).

L'interface expose les points d'intérêt, les descripteurs, le KNN matching, le test du ratio de Lowe, les inliers RANSAC, les homographies, la fusion et les métriques comparatives.

## Installation avec l'environnement Conda existant

Le projet cible l'environnement `OPENCV` et **ne crée aucun nouvel environnement**.

```powershell
conda activate OPENCV
cd H:\Stitching_project\panorama-studio
```

OpenCV publie plusieurs paquets qui installent tous le même module `cv2`. Pour éviter un mélange de fichiers — et notamment garantir la présence de BRIEF — nettoyez une fois les distributions OpenCV puis installez les dépendances :

```powershell
python -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless
python -m pip install -r requirements.txt
```

Vérification rapide :

```powershell
python -c "import cv2; print(cv2.__version__); print(cv2.xfeatures2d.BriefDescriptorExtractor_create())"
```

> Ne réinstallez pas `opencv-python` par-dessus `opencv-contrib-python`. Le paquet contrib contient déjà les modules principaux d'OpenCV.

## Lancement

```powershell
conda activate OPENCV
cd H:\Stitching_project\panorama-studio
python -m streamlit run app.py
```

Chargez les images **de gauche à droite**. Chaque paire voisine doit partager une zone suffisamment texturée. Choisissez l'algorithme et ses paramètres dans la barre latérale, puis cliquez sur **Lancer le traitement**. Le panorama final est téléchargeable en PNG.

L'onglet **Comparaison SIFT vs FAST + BRIEF vs ORB** relance trois pipelines indépendants sur les mêmes images. Un échec local est affiché sans interrompre les autres méthodes.

## Pipeline multi-images

Pour `N` images, l'application :

1. calcule les caractéristiques de chaque image ;
2. associe chaque paire voisine par BFMatcher KNN (`k=2`) et ratio de Lowe ;
3. estime les `N-1` homographies par RANSAC ;
4. compose ces transformations vers l'image centrale ;
5. transforme une seule fois chaque image dans le repère commun ;
6. projette des masques de validité indépendants des couleurs des pixels ;
7. réalise une moyenne ou un feather blending, puis recadre l'union valide.

Un pixel noir réel n'est donc jamais interprété comme une zone vide. La limite de taille du canevas protège aussi l'application des homographies aberrantes.

## Tests

Les tests génèrent leurs propres scènes texturées avec un recouvrement connu :

```powershell
python -m pytest -q
```

Ils contrôlent les trois méthodes, les types de distance, le matching, l'estimation géométrique, la composition et le recadrage.

## Architecture

```text
panorama-studio/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── detectors.py
│   ├── descriptors.py
│   ├── matching.py
│   ├── homography.py
│   ├── stitching.py
│   ├── comparison.py
│   └── visualization.py
└── tests/
    ├── test_detectors.py
    ├── test_matching.py
    └── test_stitching.py
```

## Limites connues

- **Parallaxe** : une homographie unique convient surtout à une scène approximativement plane ou à une caméra qui pivote autour de son centre optique. Des objets proches à des profondeurs différentes produisent des dédoublements.
- **Faible recouvrement** : moins de correspondances fiables signifie une homographie instable ou impossible.
- **Scènes sans texture** : murs unis, ciel ou flou fournissent peu de points distinctifs.
- **Objets mobiles et exposition** : ils peuvent créer des fantômes. Le feather blending atténue la couture mais ne remplace pas une compensation photométrique ou un seam finder avancé.
- **Ordre des images** : cette version suppose une séquence ordonnée, avec recouvrement entre images adjacentes.

Les visualisations de l'onglet pédagogique illustrent les idées mathématiques ; elles sont explicitement distinguées de l'implémentation optimisée d'OpenCV.
