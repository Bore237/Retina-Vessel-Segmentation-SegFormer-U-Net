# Projet : Analyse Intelligente d'Images Médicales

## 1. Introduction à l'ICPR

L'**ICPR (International Conference on Pattern Recognition)** est l'une des conférences mondiales les plus prestigieuses en intelligence artificielle et reconnaissance de formes. Organisée par l'**IAPR (International Association for Pattern Recognition)**, elle est le lieu de présentation de ruptures technologiques majeures, notamment dans le domaine de l'analyse d'images complexes.

L'ICPR ne se contente pas de présenter des algorithmes ; elle est le moteur de création de **benchmarks** (jeux de données de référence) qui permettent à la communauté scientifique de comparer l'efficacité des modèles d'IA.

## 2. Panorama des Datasets Médicaux

L'imagerie médicale est un pilier historique de l'ICPR. On y retrouve généralement trois grandes familles de datasets :

| Type de Dataset    | Exemple de Source | Objectif |
|------------------|-----------------|----------|
| **Radiologie (CT/IRM)** | LIDC-IDRI | Détection de nodules pulmonaires ou tumeurs |
| **Histopathologie** | PCAM / BACH | Classification de tissus cancéreux (coloration H&E) |
| **Ophtalmologie** | DRIVE / STARE | Analyse de la vascularisation et du nerf optique |

## 3. Focus : Segmentation Automatisée des Vaisseaux Rétiniens

### Contexte Médical
L'analyse de la structure vasculaire de la rétine est un indicateur crucial pour le diagnostic précoce de plusieurs pathologies :
* **Rétinopathie diabétique :** Détection de micro-anévrismes et de néovaisseaux.
* **Hypertension artérielle :** Modification du ratio artère/veine.
* **Glaucome et DMLA :** Suivi de la vascularisation papillaire.

###  Pourquoi le choix des Vaisseaux Rétiniens ?
[cite_start]Contrairement à d'autres types d'imagerie (comme la segmentation 3D de tumeurs cérébrales sur laquelle j'ai également travaillé [cite: 2]), les vaisseaux rétiniens présentent des défis uniques en **Vision par Ordinateur** :
* **Complexité structurelle :** Arborescences fines et jonctions complexes.
* **Faible contraste :** Difficulté de distinction entre les vaisseaux et le fond de la rétine (bruit de fond).
* **Multi-modalité :** Nécessité d'algorithmes robustes capables de traiter des images issues de différents dispositifs de fond d'œil.

###  Approche Technique & Méthodologie
[cite_start]Fort de mon expérience en **Deep Learning** (TensorFlow, PyTorch), l'approche retenue repose sur :
* **Prétraitement :** Égalisation d'histogramme adaptative (CLAHE) et filtrage de Gabor pour accentuer les structures filaires.
* **Architecture :** Utilisation de réseaux de neurones convolutifs de type **U-Net**, particulièrement performants pour la segmentation médicale précise.
* **Validation :** Évaluation des performances via les métriques de sensibilité, spécificité et score F1 sur des bases de données de référence (type DRIVE ou STARE).

### Compétences Transverses Appliquées
Ce choix s'inscrit dans une démarche globale d'ingénierie biomédicale :
* [cite_start]**Maîtrise des standards :** Manipulation de fichiers haute résolution et respect des flux DICOM.
* [cite_start]**Rigueur Scientifique :** Développement de pipelines reproductibles en Python.
* [cite_start]**Conformité :** Sensibilisation aux normes ISO 13485 pour le développement de logiciels en tant que dispositif médical (SaMD)[cite: 2].