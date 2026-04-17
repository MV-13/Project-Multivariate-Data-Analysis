"""
main_unsupervised.py
====================

Exploration non-supervisée du jeu de données JODA.

Étapes :
    1. Chargement des 3 blocs : NMR (tenseur), EEM (tenseur), LC-MS (matrice)
    2. Gestion des NaN (EEM contient typiquement des NaN dans les régions
       de Rayleigh/Raman masquées)
    3. Prétraitement : centrage mode "mixtures" + normalisation Frobenius
    4. (a) PCA sur chaque bloc déplié (NaN -> 0 après centrage)
       (b) PARAFAC sur chaque tenseur avec masquage des NaN
       (c) CMTF-OPT tri-blocs avec masquage
       (d) ACMTF tri-blocs avec identification des composantes partagées
    5. Visualisation des scores, poids λ/σ, et matching avec les
       concentrations réelles.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_joda
from preprocessing import preprocess_block
from methods.pca_sparse import run_pca
from methods.parafac import best_parafac
from methods.cmtf import cmtf_opt, CoupledDataset
from methods.acmtf import acmtf_opt


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def make_nan_mask(X: np.ndarray) -> np.ndarray:
    """Retourne un masque W : 1 où X est connu, 0 où X est NaN."""
    return (~np.isnan(X)).astype(float)


def nan_to_zero_after_center(X: np.ndarray) -> np.ndarray:
    """
    Pour la PCA : on centre en ignorant les NaN puis on remplace par 0.
    """
    X = X.copy()
    mean = np.nanmean(X, axis=0, keepdims=True)
    # remplace NaN par la moyenne (imputation mean)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(mean, inds[1:][0] if len(inds) > 1 else 0)
    X = np.where(np.isnan(X), np.broadcast_to(mean, X.shape), X)
    # centrage + remplacement des NaN résiduels par 0
    X = X - mean
    X = np.nan_to_num(X, nan=0.0)
    return X


def preprocess_with_nan(X: np.ndarray, center: bool = True,
                         normalize: bool = True) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Prétraitement tolérant aux NaN :
        - centrage par la moyenne du mode 0 (nanmean)
        - remplissage des NaN par 0 (dans X centré)
        - normalisation Frobenius sur les valeurs valides uniquement
    Retourne (X_pret, W, norme).
    """
    W = make_nan_mask(X)
    mean = np.nanmean(X, axis=0, keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    if center:
        Xc = np.where(np.isnan(X), 0.0, X - mean)
    else:
        Xc = np.where(np.isnan(X), 0.0, X)
    if normalize:
        nrm = float(np.sqrt(np.sum((Xc * W) ** 2)))
        if nrm > 0:
            Xc = Xc / nrm
    else:
        nrm = 1.0
    return Xc, W, nrm


def main():
    data = load_joda("data")

    # Downsampling NMR x10
    if data.NMR is not None and data.NMR.shape[1] > 2000:
        data.NMR = data.NMR[:, ::10, :]
        print(f"[downsample] NMR -> {data.NMR.shape}")

    print(f"[data] {data.n_samples} échantillons chargés")
    for name, A in [("NMR", data.NMR), ("EEM", data.EEM), ("LCMS", data.LCMS)]:
        if A is not None:
            n_nan = int(np.isnan(A).sum())
            print(f"  {name:5s} : shape {A.shape}  NaN={n_nan}")

    blocks = {}
    masks = {}
    for name, X in [("NMR", data.NMR), ("EEM", data.EEM), ("LCMS", data.LCMS)]:
        if X is None:
            continue
        if np.isnan(X).any():
            Xp, W, _ = preprocess_with_nan(X)
            masks[name] = W
        else:
            Xp, _ = preprocess_block(X)
            masks[name] = None
        blocks[name] = Xp

    R = 5 

    # ------------------------------------------------------------------
    # 1. PCA par bloc (imputation mean pour les NaN)
    # ------------------------------------------------------------------
    print("\n=== PCA par bloc ===")
    for name, X in blocks.items():
        X_pca = np.nan_to_num(X, nan=0.0)
        scores, loadings, evr = run_pca(X_pca, n_components=R)
        print(f"  {name:5s} : variance expliquée cumulée = "
              f"{evr.cumsum()[-1]:.3f}")
        np.save(os.path.join(RESULTS_DIR, f"pca_{name}_scores.npy"), scores)

    # ------------------------------------------------------------------
    # 2. PARAFAC
    # ------------------------------------------------------------------
    print("\n=== PARAFAC par tenseur ===")
    parafac_results = {}
    for name in ("NMR", "EEM"):
        if name not in blocks:
            continue
        X = blocks[name]
        if masks[name] is not None:
            try:
                import tensorly as tl
                from tensorly.decomposition import parafac
                mask = masks[name]
                Xt = tl.tensor(np.nan_to_num(X, nan=0.0))
                cp = parafac(Xt, rank=R, n_iter_max=300, tol=1e-7,
                             init="random", normalize_factors=True,
                             mask=tl.tensor(mask), random_state=0)
                weights, factors = cp
                X_hat = tl.cp_to_tensor((weights, factors))
                rel_err = float(np.linalg.norm(
                    (X - X_hat) * mask) / (np.linalg.norm(X * mask) + 1e-12))
                print(f"  {name} (avec masque NaN) : erreur relative = {rel_err:.4f}")
                parafac_results[name] = (np.asarray(weights),
                                         [np.asarray(f) for f in factors],
                                         rel_err)
                np.save(os.path.join(RESULTS_DIR, f"parafac_{name}_A.npy"),
                        parafac_results[name][1][0])
            except Exception as e:
                print(f"  {name} PARAFAC masqué échoué ({e}), fallback NaN->0")
                w, facs, err = best_parafac(np.nan_to_num(X, nan=0.0),
                                            rank=R, n_inits=5)
                parafac_results[name] = (w, facs, err)
                print(f"    erreur relative = {err:.4f}")
        else:
            w, facs, err = best_parafac(X, rank=R, n_inits=5)
            print(f"  {name} : erreur relative = {err:.4f}")
            parafac_results[name] = (w, facs, err)
            np.save(os.path.join(RESULTS_DIR, f"parafac_{name}_A.npy"), facs[0])

    # ------------------------------------------------------------------
    # 3. CMTF-OPT tri-blocs (avec masquage des NaN)
    # ------------------------------------------------------------------
    print("\n=== CMTF-OPT tri-blocs (Acar 2011) ===")
    tensors_list = [blocks[n] for n in ("NMR", "EEM") if n in blocks]
    tensor_names = [n for n in ("NMR", "EEM") if n in blocks]
    W_tensors   = [masks[n] for n in tensor_names]
    if all(w is None for w in W_tensors):
        W_tensors = None

    matrices_list = [blocks["LCMS"]] if "LCMS" in blocks else []
    W_matrices    = [masks["LCMS"]] if "LCMS" in blocks else []
    if all(w is None for w in W_matrices):
        W_matrices = None

    ds = CoupledDataset(
        tensors=tensors_list,
        matrices=matrices_list,
        W_tensors=W_tensors,
        W_matrices=W_matrices,
    )
    cmtf_res = cmtf_opt(ds, R=R, n_inits=3, max_iter=400)
    print(f"  fval final = {cmtf_res['fval']:.6f}")
    np.save(os.path.join(RESULTS_DIR, "cmtf_A.npy"), cmtf_res["A"])

    # ------------------------------------------------------------------
    # 4. ACMTF tri-blocs
    # ------------------------------------------------------------------
    print("\n=== ACMTF tri-blocs (Acar 2014) ===")
    acmtf_res = acmtf_opt(ds, R=R + 1, beta=1e-2, alpha=1.0,
                          n_inits=5, max_iter=1000)
    print(f"  fval final = {acmtf_res['fval']:.6f}")
    for t_idx, lam in enumerate(acmtf_res["lambdas"]):
        print(f"  λ ({tensor_names[t_idx]}) = "
              f"{np.array2string(lam, precision=3)}")
    for m_idx, sig in enumerate(acmtf_res["sigmas"]):
        print(f"  σ (LCMS) = {np.array2string(sig, precision=3)}")

    # ------------------------------------------------------------------
    # 5. Figures : poids des composantes par bloc
    # ------------------------------------------------------------------
    n_blocks = len(acmtf_res["lambdas"]) + len(acmtf_res["sigmas"])
    fig, axes = plt.subplots(1, n_blocks, figsize=(4.5 * n_blocks, 4))
    if n_blocks == 1:
        axes = [axes]
    R_used = R + 1
    x = np.arange(R_used)
    b_idx = 0
    for t_idx, lam in enumerate(acmtf_res["lambdas"]):
        axes[b_idx].bar(x, lam, color="#2b5d9a")
        axes[b_idx].set_title(f"ACMTF λ — {tensor_names[t_idx]}")
        axes[b_idx].set_xticks(x)
        axes[b_idx].set_xticklabels([f"c{i+1}" for i in range(R_used)])
        axes[b_idx].grid(axis="y", alpha=0.3)
        b_idx += 1
    for m_idx, sig in enumerate(acmtf_res["sigmas"]):
        axes[b_idx].bar(x, sig, color="#c0392b")
        axes[b_idx].set_title("ACMTF σ — LCMS")
        axes[b_idx].set_xticks(x)
        axes[b_idx].set_xticklabels([f"c{i+1}" for i in range(R_used)])
        axes[b_idx].grid(axis="y", alpha=0.3)
        b_idx += 1
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsupervised_acmtf_weights.png"),
                dpi=130)
    plt.close(fig)
    print(f"\n[saved] résultats dans {RESULTS_DIR}/")

    # ------------------------------------------------------------------
    # 6. Matching avec concentrations vraies
    # ------------------------------------------------------------------
    if data.Y is not None:
        from scipy.stats import pearsonr
        A = acmtf_res["A"]
        print("\n=== Corrélations ACMTF (A) vs concentrations vraies ===")
        names = data.analyte_names or [f"col{c}" for c in range(data.Y.shape[1])]
        for r in range(A.shape[1]):
            best_rho, best_name = 0.0, ""
            for c in range(data.Y.shape[1]):
                rho = pearsonr(A[:, r], data.Y[:, c])[0]
                if abs(rho) > abs(best_rho):
                    best_rho, best_name = rho, names[c]
            print(f"  c{r+1}: {best_name:<18} ρ={best_rho:+.3f}")


if __name__ == "__main__":
    main()
