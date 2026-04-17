"""
main_supervised.py
==================

Prédiction des concentrations des 5 constituants chimiques à partir des
trois sources de mesures. On compare :

    (1) Baselines : Ridge et PLS sur chaque bloc déplié
    (2) Régression tensorielle CP de rang faible sur NMR et EEM
    (3) Multi-bloc via CMTF et ACMTF : on extrait A sur l'ensemble
        d'entraînement puis on fait une régression linéaire A -> Y

Évaluation : leave-one-out CV (28 échantillons → 28 replis).

Gestion des NaN : l'EEM contient des NaN (zones Rayleigh/Raman masquées).
On les remplace par 0 après centrage pour les méthodes qui ne gèrent pas
les NaN nativement (Ridge, PLS, CP regression). CMTF/ACMTF utilisent des
masques.

Usage :
    python main_supervised.py
"""

from __future__ import annotations

import os
import warnings
import numpy as np

from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

from data_loader import load_joda
from preprocessing import preprocess_block
from methods.tensor_regression import (
    cp_tensor_regression, cp_tensor_regression_predict
)
from methods.cmtf import cmtf_opt, CoupledDataset
from methods.acmtf import acmtf_opt
from evaluation import leave_one_out_cv, r2_rmse

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def clean_nan(X: np.ndarray) -> np.ndarray:
    """Remplace les NaN par la moyenne du mode 0 (par feature), puis 0 si reste."""
    if not np.isnan(X).any():
        return X
    mean = np.nanmean(X, axis=0, keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    X = np.where(np.isnan(X), np.broadcast_to(mean, X.shape), X)
    X = np.nan_to_num(X, nan=0.0)
    return X


def make_nan_mask(X: np.ndarray) -> np.ndarray:
    return (~np.isnan(X)).astype(float) if np.isnan(X).any() else None


# ----------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------
def fit_ridge(X_tr, Y_tr, X_te, alpha=1.0):
    model = Ridge(alpha=alpha).fit(X_tr.reshape(X_tr.shape[0], -1), Y_tr)
    return model.predict(X_te.reshape(X_te.shape[0], -1))


def fit_pls(X_tr, Y_tr, X_te, n_components=5):
    n_components = min(n_components, X_tr.shape[0] - 1)
    model = PLSRegression(n_components=n_components)
    model.fit(X_tr.reshape(X_tr.shape[0], -1), Y_tr)
    return model.predict(X_te.reshape(X_te.shape[0], -1))


def fit_cp_reg(X_tr, Y_tr, X_te, rank=3):
    """CP regression mono-sortie : un modèle par colonne de Y."""
    Y_tr = np.asarray(Y_tr, dtype=float)
    if Y_tr.ndim == 1:
        Y_tr = Y_tr[:, None]
    Y_pred = np.zeros((X_te.shape[0], Y_tr.shape[1]))
    for c in range(Y_tr.shape[1]):
        res = cp_tensor_regression(X_tr, Y_tr[:, c], rank=rank,
                                   max_iter=60, l2=1e-1)
        Y_pred[:, c] = cp_tensor_regression_predict(res, X_te)
    return Y_pred


# ----------------------------------------------------------------------
# Projection CMTF/ACMTF d'un nouvel échantillon sur les facteurs appris
# ----------------------------------------------------------------------
def project_sample(X_te_blocks, factors_per_block, use_acmtf=False):
    """
    Estime le vecteur a_te de l'échantillon test en projetant ses blocs sur
    les facteurs des autres modes (fixés lors de l'entraînement). Pour
    chaque bloc on obtient une estimation, on les moyenne à la fin.

    Parameters
    ----------
    X_te_blocks : dict {name -> array_like de l'échantillon test}
    factors_per_block : dict avec clés 'tensor_factors' / 'matrix_factors'
        et les facteurs des autres modes pour chaque bloc.
    """
    a_estimates = []
    for t_idx, facs in enumerate(factors_per_block["tensor_factors_no_A"]):
        X_te = factors_per_block["tensor_te"][t_idx]
        if np.isnan(X_te).any():
            X_te = np.nan_to_num(X_te, nan=0.0)
        from methods.cmtf import khatri_rao
        if len(facs) == 2:
            B_, C_ = facs
            kr = khatri_rao([B_, C_])  # (J*K, R)
            x_flat = X_te[0].reshape(-1)
        else:
            kr = khatri_rao(facs)
            x_flat = X_te[0].reshape(-1)
        a_i, *_ = np.linalg.lstsq(kr, x_flat, rcond=None)
        a_estimates.append(a_i)
    for m_idx, V in enumerate(factors_per_block["matrix_factors"]):
        X_te = factors_per_block["matrix_te"][m_idx]
        if np.isnan(X_te).any():
            X_te = np.nan_to_num(X_te, nan=0.0)
        x_te = X_te[0]
        a_i, *_ = np.linalg.lstsq(V, x_te, rcond=None)
        a_estimates.append(a_i)
    return np.mean(a_estimates, axis=0)


def cmtf_loo_predict(blocks_tr, masks_tr, Y_tr, blocks_te,
                     R=5, use_acmtf=False, beta=1e-2):
    """Fit CMTF/ACMTF sur l'entraînement puis prédit Y_te."""
    tensor_names = [n for n in blocks_tr if blocks_tr[n].ndim == 3]
    matrix_names = [n for n in blocks_tr if blocks_tr[n].ndim == 2]

    ds_tr = CoupledDataset(
        tensors=[blocks_tr[n] for n in tensor_names],
        matrices=[blocks_tr[n] for n in matrix_names],
        W_tensors=[masks_tr[n] for n in tensor_names]
                    if any(masks_tr[n] is not None for n in tensor_names) else None,
        W_matrices=[masks_tr[n] for n in matrix_names]
                    if any(masks_tr[n] is not None for n in matrix_names) else None,
    )

    if use_acmtf:
        res = acmtf_opt(ds_tr, R=R, beta=beta, alpha=1.0,
                        n_inits=2, max_iter=300)
        A_tr = res["A"]
        tensor_facs_no_A = [
            res["tensor_factors"][t_idx][1:] for t_idx in range(len(tensor_names))
        ]
        matrix_facs = res["matrix_factors"]
    else:
        res = cmtf_opt(ds_tr, R=R, n_inits=2, max_iter=300)
        A_tr = res["A"]
        tensor_facs_no_A = [
            res["factors_tensors"][t_idx][1:] for t_idx in range(len(tensor_names))
        ]
        matrix_facs = res["factors_matrices"]

    P = np.column_stack([np.ones(A_tr.shape[0]), A_tr])
    coefs, *_ = np.linalg.lstsq(P, Y_tr, rcond=None)

    factors_proj = {
        "tensor_factors_no_A": tensor_facs_no_A,
        "tensor_te": [blocks_te[n] for n in tensor_names],
        "matrix_factors": matrix_facs,
        "matrix_te": [blocks_te[n] for n in matrix_names],
    }
    a_te = project_sample(blocks_te, factors_proj, use_acmtf=use_acmtf)
    return (np.concatenate([[1.0], a_te]) @ coefs).reshape(1, -1)


# ----------------------------------------------------------------------
# Script principal
# ----------------------------------------------------------------------
def main():
    data = load_joda("data")
    if data.Y is None:
        raise RuntimeError("Aucune concentration trouvée.")

    # Downsampling NMR x10
    if data.NMR is not None and data.NMR.shape[1] > 2000:
        data.NMR = data.NMR[:, ::10, :]
        print(f"[downsample] NMR -> {data.NMR.shape}")

    Y = data.Y
    names = data.analyte_names or [f"Y{c+1}" for c in range(Y.shape[1])]
    print(f"[data] {data.n_samples} échantillons, Y shape = {Y.shape}")

    # Nettoyage NaN + prétraitement des blocs
    raw_blocks = {}
    if data.NMR  is not None: raw_blocks["NMR"]  = data.NMR
    if data.EEM  is not None: raw_blocks["EEM"]  = data.EEM
    if data.LCMS is not None: raw_blocks["LCMS"] = data.LCMS

    for name, X in raw_blocks.items():
        n_nan = int(np.isnan(X).sum())
        if n_nan > 0:
            print(f"  [!] {name} contient {n_nan} NaN -> imputation par mean")

    # Version "propre" pour Ridge/PLS/CP regression : NaN -> mean
    # + centrage + normalisation Frobenius
    clean_blocks = {}
    masks = {}
    for name, X in raw_blocks.items():
        X_clean = clean_nan(X)
        X_pp, _ = preprocess_block(X_clean)
        clean_blocks[name] = X_pp
        masks[name] = make_nan_mask(X)

    results = {}

    # ------------------------------------------------------------------
    # (1) Baselines Ridge / PLS par bloc
    # ------------------------------------------------------------------
    print("\n=== Baselines Ridge / PLS par bloc (LOO-CV) ===")
    for name, X in clean_blocks.items():
        res_ridge = leave_one_out_cv(X, Y, fit_ridge)
        res_pls   = leave_one_out_cv(X, Y, fit_pls)
        print(f"  {name:5s} Ridge  R²={res_ridge['r2_mean']:.3f}  "
              f"RMSE={res_ridge['rmse_mean']:.3f}")
        print(f"  {name:5s} PLS    R²={res_pls['r2_mean']:.3f}  "
              f"RMSE={res_pls['rmse_mean']:.3f}")
        results[f"ridge_{name}"] = res_ridge
        results[f"pls_{name}"]   = res_pls

    # ------------------------------------------------------------------
    # (2) Régression CP de rang faible sur les tenseurs
    # ------------------------------------------------------------------
    print("\n=== Régression CP (tensor regression, LOO-CV) ===")
    for name in ("NMR", "EEM"):
        if name in clean_blocks:
            res = leave_one_out_cv(
                clean_blocks[name], Y,
                lambda a, b, c: fit_cp_reg(a, b, c, rank=3),
            )
            print(f"  {name} CP(rank=3)  R²={res['r2_mean']:.3f}  "
                  f"RMSE={res['rmse_mean']:.3f}")
            results[f"cp_{name}"] = res

    # ------------------------------------------------------------------
    # (3) Multi-bloc via CMTF/ACMTF
    # ------------------------------------------------------------------
    print("\n=== CMTF / ACMTF + régression linéaire (LOO-CV) ===")
    print("    (attention, ~5-10 min pour chaque méthode)")

    n = data.n_samples
    Y_pred_cmtf  = np.zeros_like(Y, dtype=float)
    Y_pred_acmtf = np.zeros_like(Y, dtype=float)

    for i in range(n):
        if i % 4 == 0:
            print(f"    LOO-CV sample {i+1}/{n}...")
        idx = [j for j in range(n) if j != i]
        blocks_tr = {k: v[idx] for k, v in clean_blocks.items()}
        blocks_te = {k: v[i:i+1] for k, v in clean_blocks.items()}
        masks_tr = {k: (v[idx] if v is not None else None)
                    for k, v in masks.items()}
        try:
            Y_pred_cmtf[i] = cmtf_loo_predict(
                blocks_tr, masks_tr, Y[idx], blocks_te,
                R=5, use_acmtf=False
            )
        except Exception as e:
            print(f"      [warn] CMTF sample {i}: {e}")
            Y_pred_cmtf[i] = Y[idx].mean(axis=0)
        try:
            Y_pred_acmtf[i] = cmtf_loo_predict(
                blocks_tr, masks_tr, Y[idx], blocks_te,
                R=6, use_acmtf=True, beta=1e-2
            )
        except Exception as e:
            print(f"      [warn] ACMTF sample {i}: {e}")
            Y_pred_acmtf[i] = Y[idx].mean(axis=0)

    res_cmtf  = r2_rmse(Y, Y_pred_cmtf)
    res_acmtf = r2_rmse(Y, Y_pred_acmtf)
    print(f"\n  CMTF+lin   R²={res_cmtf['r2_mean']:.3f}  "
          f"RMSE={res_cmtf['rmse_mean']:.3f}")
    print(f"  ACMTF+lin  R²={res_acmtf['r2_mean']:.3f}  "
          f"RMSE={res_acmtf['rmse_mean']:.3f}")

    # ------------------------------------------------------------------
    # Résumé final + détail par analyte
    # ------------------------------------------------------------------
    print("\n=== Résumé LOO-CV (R² moyen sur les 5 analytes) ===")
    print(f"{'méthode':<22}  R²       RMSE")
    for k, r in results.items():
        print(f"{k:<22}  {r['r2_mean']:>6.3f}   {r['rmse_mean']:>6.3f}")
    print(f"{'CMTF+lin':<22}  {res_cmtf['r2_mean']:>6.3f}   "
          f"{res_cmtf['rmse_mean']:>6.3f}")
    print(f"{'ACMTF+lin':<22}  {res_acmtf['r2_mean']:>6.3f}   "
          f"{res_acmtf['rmse_mean']:>6.3f}")

    print("\n=== R² par analyte (meilleures méthodes) ===")
    header = f"{'méthode':<22}  " + "  ".join(f"{n[:10]:>10s}" for n in names)
    print(header)
    best_methods = ["pls_NMR", "pls_EEM", "pls_LCMS", "cp_NMR", "cp_EEM"]
    for k in best_methods:
        if k in results:
            r2s = results[k]["r2_per_col"]
            row = f"{k:<22}  " + "  ".join(f"{v:>10.3f}" for v in r2s)
            print(row)
    row = f"{'CMTF+lin':<22}  " + "  ".join(f"{v:>10.3f}" for v in res_cmtf["r2_per_col"])
    print(row)
    row = f"{'ACMTF+lin':<22}  " + "  ".join(f"{v:>10.3f}" for v in res_acmtf["r2_per_col"])
    print(row)

    # Sauvegarde
    np.savez(
        os.path.join("results", "supervised_loo_results.npz"),
        Y_true=Y,
        **{f"ypred_{k}": r["Y_pred"] for k, r in results.items()},
        ypred_cmtf=Y_pred_cmtf,
        ypred_acmtf=Y_pred_acmtf,
        analytes=np.array(names),
    )
    print(f"\n[saved] results/supervised_loo_results.npz")


if __name__ == "__main__":
    main()
