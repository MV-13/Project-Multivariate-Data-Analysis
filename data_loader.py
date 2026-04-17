"""
data_loader.py
==============

Chargement du jeu de données JODA.

Format du fichier de données utilisé (EEM_NMR_LCMS.mat) :
    - Un seul fichier .mat contenant 3 variables X, Y, Z. Chaque DataSet
      est un struct MATLAB avec un champ `.data` contenant le tenseur/matrice
      effectif, et des champs `.label`, `.axisscale`, `.title`, etc. pour
      les métadonnées.
    - Mapping :
        Y.data (28, 13324, 8)  -> NMR   (mixtures x chemical shift x gradient)
        X.data (28, 251, 21)   -> EEM   (mixtures x emission x excitation)
        Z.data (28, 168)       -> LC-MS (mixtures x features)

    - concentrations.txt : matrice (28, 5) avec en-tête et colonne d'index.
      Colonnes : Val-Tyr-Val, Trp-Gly, Phe, Malto, Propanol.

"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.io import loadmat


# ----------------------------------------------------------------------
# Dataset container 
# ----------------------------------------------------------------------
@dataclass
class JodaData:
    NMR: Optional[np.ndarray]           # (I, J_nmr, K_grad)
    EEM: Optional[np.ndarray]           # (I, J_em,  K_ex)
    LCMS: Optional[np.ndarray]          # (I, P)
    Y: Optional[np.ndarray]             # (I, 5)  concentrations
    nmr_axes: Optional[dict] = None
    eem_axes: Optional[dict] = None
    lcms_features: Optional[np.ndarray] = None
    analyte_names: Optional[list] = None

    @property
    def n_samples(self) -> int:
        for A in (self.NMR, self.EEM, self.LCMS, self.Y):
            if A is not None:
                return A.shape[0]
        raise ValueError("Aucun bloc chargé.")


# ----------------------------------------------------------------------
# Gestion du format PLS_Toolbox DataSet
# ----------------------------------------------------------------------
def _extract_dataset_field(ds, field: str):
    """
    Extrait un champ d'un DataSet PLS_Toolbox chargé par scipy.io.loadmat.

    Un DataSet arrive typiquement comme un ndarray object de shape (1, 1)
    dont le dtype a des noms de champs (structured array). Chaque champ
    est lui-même un ndarray (1, 1) de type object contenant la vraie valeur.
    """
    # cas 1 : structured array
    if hasattr(ds, "dtype") and ds.dtype.names and field in ds.dtype.names:
        val = ds[field]
        while isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
            val = val.flat[0]
        return val
    # cas 2 : MatlabObject
    if hasattr(ds, field):
        val = getattr(ds, field)
        while isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
            val = val.flat[0]
        return val
    return None


def _unwrap_dataset_data(ds) -> np.ndarray:
    """Renvoie le champ .data d'un DataSet PLS_Toolbox comme ndarray float."""
    data = _extract_dataset_field(ds, "data")
    if data is None:
        raise ValueError("Impossible d'extraire le champ .data du DataSet")
    data = np.asarray(data, dtype=float)
    return data


def _unwrap_axisscale(ds) -> list:
    """
    Renvoie la liste des axis scales d'un DataSet (une liste de vecteurs,
    un par mode). Utile pour récupérer les chemical shifts, longueurs d'onde,
    etc. Renvoie une liste vide si indisponible.
    """
    ax = _extract_dataset_field(ds, "axisscale")
    if ax is None:
        return []
    scales = []
    try:
        ax = np.asarray(ax, dtype=object)
        if ax.ndim == 2:
            for row in range(ax.shape[0]):
                cell = ax[row, 0]
                while isinstance(cell, np.ndarray) and cell.dtype == object and cell.size == 1:
                    cell = cell.flat[0]
                if isinstance(cell, np.ndarray) and cell.size > 0:
                    scales.append(np.asarray(cell, dtype=float).ravel())
                else:
                    scales.append(None)
    except Exception:
        pass
    return scales


# ----------------------------------------------------------------------
# Concentrations en texte
# ----------------------------------------------------------------------
def load_concentrations_txt(path: str) -> tuple[np.ndarray, list[str]]:
    """
    Charge concentrations.txt. Format observé :

        (entête)  Val-Tyr-Val Trp-Gly  Phe Malto Propanol
        1         5.00    0.00 0.00  0.00     0.00
        2         0.00    5.00 0.00  0.00     0.00
        ...

    Il y a une colonne d'index au début (1..28) qu'on supprime.
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]

    header_tokens = lines[0].split()
    first_data = lines[1].split()
    has_index_col = len(first_data) == len(header_tokens) + 1

    rows = []
    for ln in lines[1:]:
        toks = ln.split()
        if has_index_col:
            toks = toks[1:]
        rows.append([float(t) for t in toks])
    Y = np.asarray(rows, dtype=float)

    names = header_tokens
    if len(names) != Y.shape[1]:
        # fallback générique
        names = [f"analyte_{i+1}" for i in range(Y.shape[1])]
    return Y, names


# ----------------------------------------------------------------------
# API principale
# ----------------------------------------------------------------------
def load_joda(data_dir: str = "data") -> JodaData:
    """
    Charge le jeu JODA depuis `data_dir`.
    """
    files = {f: os.path.join(data_dir, f) for f in os.listdir(data_dir)}
    files_lower = {f.lower(): os.path.join(data_dir, f) for f in os.listdir(data_dir)}

    NMR = EEM = LCMS = None
    nmr_axes = eem_axes = None

    condensed = None
    for fname, fpath in files.items():
        low = fname.lower()
        if low.endswith(".mat") and ("eem" in low and "nmr" in low and ("lcms" in low or "lc_ms" in low or "lc-ms" in low)):
            condensed = fpath
            break
    if condensed is None:
        mat_files = [p for f, p in files.items() if f.lower().endswith(".mat")]
        if len(mat_files) == 1:
            try:
                test = loadmat(mat_files[0])
                keys = {k for k in test.keys() if not k.startswith("__")}
                if {"X", "Y", "Z"}.issubset(keys):
                    condensed = mat_files[0]
            except Exception:
                pass

    if condensed is not None:
        print(f"[load_joda] format PLS_Toolbox détecté : {os.path.basename(condensed)}")
        mat = loadmat(condensed)
        datasets = {}
        for key in ("X", "Y", "Z"):
            if key in mat:
                arr = _unwrap_dataset_data(mat[key])
                datasets[key] = (arr, mat[key])
                print(f"  {key}.data shape = {arr.shape}")

        # identification :
        #   - le 3D avec la plus grande dim centrale = NMR (chemical shift ~10^4)
        #   - l'autre 3D                             = EEM
        #   - le 2D                                  = LC-MS
        three_way = [(k, a, ds) for k, (a, ds) in datasets.items() if a.ndim == 3]
        two_way   = [(k, a, ds) for k, (a, ds) in datasets.items() if a.ndim == 2]

        if len(three_way) == 2 and len(two_way) == 1:
            three_way.sort(key=lambda t: t[1].shape[1], reverse=True)
            (knmr, NMR, ds_nmr), (keem, EEM, ds_eem) = three_way
            klcms, LCMS, ds_lcms = two_way[0]
            print(f"  mapping : {knmr} -> NMR (shape {NMR.shape})")
            print(f"            {keem} -> EEM (shape {EEM.shape})")
            print(f"            {klcms} -> LCMS (shape {LCMS.shape})")

            nmr_scales = _unwrap_axisscale(ds_nmr)
            eem_scales = _unwrap_axisscale(ds_eem)
            nmr_axes = {
                "chemical_shift": nmr_scales[1] if len(nmr_scales) > 1 else None,
                "gradient":       nmr_scales[2] if len(nmr_scales) > 2 else None,
            }
            eem_axes = {
                "mode1": eem_scales[1] if len(eem_scales) > 1 else None,
                "mode2": eem_scales[2] if len(eem_scales) > 2 else None,
            }
        elif len(three_way) + len(two_way) > 0:
            for k, a, _ds in three_way:
                if NMR is None:
                    NMR = a
                elif EEM is None:
                    EEM = a
            for k, a, _ds in two_way:
                if LCMS is None:
                    LCMS = a
    else:
        def _find(keywords):
            for low, p in files_lower.items():
                if any(k in low for k in keywords):
                    return p
            return None

        nmr_p  = _find(["nmr"])
        eem_p  = _find(["eem", "fluo"])
        lcms_p = _find(["lcms", "lc-ms", "lc_ms"])
        if nmr_p:
            NMR = _unwrap_dataset_data(loadmat(nmr_p).get("NMR")) \
                  if "NMR" in loadmat(nmr_p) else NotImplemented

    Y_conc = None
    analyte_names = ["Val-Tyr-Val", "Trp-Gly", "Phe", "Maltoheptaose", "Propanol"]
    for fname, fpath in files.items():
        low = fname.lower()
        if low.startswith("conc") or low in ("y.txt", "y.csv", "target.txt"):
            Y_conc, names = load_concentrations_txt(fpath)
            if names and len(names) == Y_conc.shape[1]:
                analyte_names = names
            print(f"[load_joda] concentrations chargées : {fname} -> shape {Y_conc.shape}")
            break

    return JodaData(
        NMR=NMR, EEM=EEM, LCMS=LCMS, Y=Y_conc,
        nmr_axes=nmr_axes, eem_axes=eem_axes,
        analyte_names=analyte_names,
    )



if __name__ == "__main__":
    data = load_joda("data")
    print(f"\nN samples : {data.n_samples}")
    if data.NMR  is not None: print(f"  NMR  : {data.NMR.shape}")
    if data.EEM  is not None: print(f"  EEM  : {data.EEM.shape}")
    if data.LCMS is not None: print(f"  LCMS : {data.LCMS.shape}")
    if data.Y    is not None:
        print(f"  Y    : {data.Y.shape}")
        print(f"  analytes : {data.analyte_names}")
        print(f"  first 3 rows:\n{data.Y[:3]}")
