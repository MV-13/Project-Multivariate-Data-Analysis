# Projet AMDA — Analyse du jeu de données JODA

Analyse multi-bloc et tensorielle du jeu de données JODA (28 échantillons,
5 constituants chimiques) mesuré par NMR (tenseur), EEM (tenseur),
LC-MS (matrice), avec concentrations connues.

Ce projet met en œuvre les méthodes vues en cours "analyse de données multivariées avancées" et reproduit
partiellement les résultats des articles :

- Acar, Kolda, Dunlavy (2011) — *All-at-once Optimization for Coupled
  Matrix and Tensor Factorizations* (CMTF-OPT)
- Acar et al. (2014) — *Structure-revealing data fusion* (ACMTF)

## Structure du projet

```
joda_project/
├── data/                      # placer ici les fichiers .mat et concentrations
├── data_loader.py             # chargement NMR / EEM / LC-MS / Y
├── preprocessing.py           # centrage, scaling, normalisation de blocs
├── methods/
│   ├── pca_sparse.py          # (s)PCA par bloc déplié
│   ├── parafac.py             # CP/PARAFAC sur tenseurs (NMR, EEM)
│   ├── rgcca.py               # (s)GCCA multi-bloc (wrapper)
│   ├── cmtf.py                # CMTF-OPT (Acar 2011)
│   ├── acmtf.py               # ACMTF structure-revealing (Acar 2014)
│   └── tensor_regression.py   # régression tensorielle de rang faible
├── evaluation.py              # FMS, R², RMSE, CV
├── main_unsupervised.py       # exploration non-supervisée
├── main_supervised.py         # prédiction des concentrations
├── main_reproduce_paper.py    # reproduction Acar 2014 (NMR+LC-MS)
└── README.md
```

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib tensorly pandas
# optionnel pour RGCCA (wrapper rpy2 + package R RGCCA) :
pip install rpy2
```

## Données attendues dans `data/`

- `NMR`          — tenseur 28 × J_nmr × K_grad  (mixtures × chemical shift × gradient)
- `EEM`          — tenseur 28 × J_ex × K_em     (mixtures × excitation × emission)
- `LCMS`         — matrice  28 × P_features
- `concentrations` — matrice 28 × 5 (Val-Tyr-Val, Trp-Gly, Phe, Malto, Propanol)

Le module `data_loader.py` tolère différents noms de variables dans les `.mat`
et peut être adapté facilement.

## Exécution

```bash
# 1. Exploration non supervisée (PCA, PARAFAC, CMTF, ACMTF)
python main_unsupervised.py

# 2. Prédiction supervisée des concentrations
python main_supervised.py

# 3. Reproduction du papier Acar 2014 (couplage NMR + LC-MS)
python main_reproduce_paper.py
```

## Pistes traitées

1. **Reproduction** : ACMTF sur NMR (tenseur) couplé à LC-MS (matrice),
   identification des composantes partagées/non-partagées, détection du
   propanol invisible en LC-MS.
2. **Non-supervisé** : PCA par bloc, PARAFAC sur chaque tenseur, CMTF
   tri-blocs (NMR + EEM + LC-MS couplés dans le mode "mixtures").
3. **Supervisé** : régression tensorielle de rang faible pour prédire
   les concentrations, comparaison avec PLS / Ridge sur matrices dépliées.
4. **Évaluation** : apport de la structure tensorielle et multi-bloc
   quantifié par validation croisée leave-one-out (28 échantillons).

## Auteurs 
Hélène Lavenant & Marine Vieillard