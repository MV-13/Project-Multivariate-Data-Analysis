"""
methods/acmtf.py
================

ACMTF : Advanced / Structure-revealing CMTF
(Acar, Papalexakis, Gurdeniz, Rasmussen, Lawaetz, Nilsson, Bro, 2014)

Formulation :

    min_{λ, σ, A, B, C, V}
        || X - [[λ; A, B, C]] ||^2
      + || Y - A Σ V^T     ||^2
      + β (||λ||_1 + ||σ||_1)
    s.c. ||a_r|| = ||b_r|| = ||c_r|| = ||v_r|| = 1

où λ et σ sont les poids des composantes rang-1 dans X et Y
respectivement. La pénalité L1 sur (λ, σ) fait apparaître
automatiquement les composantes partagées et non-partagées :
une composante non-partagée aura un poids ~0 dans l'un des blocs.

Résolution :
    - contraintes de norme imposées via quadratic penalty (α ≥ 0)
    - L1 approximée par sqrt(x^2 + eps) (différentiable)
    - optimisation all-at-once par L-BFGS-B

Cette implémentation gère un nombre arbitraire de tenseurs et matrices
couplés dans le mode 0 (voir également methods/cmtf.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .cmtf import CoupledDataset, khatri_rao, _unfold as unfold, cp_to_tensor


# ----------------------------------------------------------------------
# Packing / unpacking des variables
# ----------------------------------------------------------------------
def _acmtf_shapes(ds: CoupledDataset, R: int):
    """
    Variables :
      - A (I x R)
      - pour chaque tenseur T_t : un poids lambda_t (R,) et des facteurs
        des modes >0 : (J_d x R)
      - pour chaque matrice M_l : un poids sigma_l (R,) et V_l (N_l x R)
    """
    I = ds.I
    shapes = [("A", (I, R))]
    for t_idx, T in enumerate(ds.tensors):
        shapes.append((f"lambda_{t_idx}", (R,)))
        for d in range(1, T.ndim):
            shapes.append((f"T{t_idx}_mode{d}", (T.shape[d], R)))
    for m_idx, M in enumerate(ds.matrices):
        shapes.append((f"sigma_{m_idx}", (R,)))
        shapes.append((f"V_{m_idx}", (M.shape[1], R)))
    return shapes


def _pack(vars_: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([v.ravel() for v in vars_])


def _unpack(x: np.ndarray, shapes):
    out, k = [], 0
    for _, s in shapes:
        n = int(np.prod(s))
        out.append(x[k:k + n].reshape(s))
        k += n
    return out


# ----------------------------------------------------------------------
# Objectif ACMTF + gradient
# ----------------------------------------------------------------------
def _acmtf_fg(x, ds: CoupledDataset, R: int, shapes,
              beta: float, alpha: float, eps: float):
    vars_ = _unpack(x, shapes)
    A = vars_[0]
    grads = [np.zeros_like(v) for v in vars_]
    f = 0.0

    # pré-normalisation : pour calculer correctement le gradient
    # de la contrainte quadratique on a besoin de ||a_r||
    def norm_cols(M):
        return np.sqrt((M ** 2).sum(axis=0))

    # --- contrainte unit-norm sur A ---
    nA = norm_cols(A)
    f += alpha * float(np.sum((nA - 1.0) ** 2))
    # dF/dA : 2 alpha (||a_r|| - 1) * a_r / ||a_r||
    safe = np.where(nA > 1e-12, nA, 1.0)
    grads[0] += 2 * alpha * (A * ((nA - 1.0) / safe)[None, :])

    # --- Tenseurs ---
    idx = 1
    for t_idx, T in enumerate(ds.tensors):
        lam = vars_[idx]         # (R,)
        lam_idx = idx
        idx += 1

        mode_factors = [A]
        mode_indices = [0]
        for d in range(1, T.ndim):
            mode_factors.append(vars_[idx])
            mode_indices.append(idx)
            # unit-norm penalty sur chaque facteur
            nB = norm_cols(vars_[idx])
            f += alpha * float(np.sum((nB - 1.0) ** 2))
            safeB = np.where(nB > 1e-12, nB, 1.0)
            grads[idx] += 2 * alpha * (vars_[idx] * ((nB - 1.0) / safeB)[None, :])
            idx += 1

        # reconstruction pondérée : A * diag(lam) absorbé via scaling du 1er facteur
        A_scaled = A * lam[None, :]
        scaled_factors = [A_scaled] + mode_factors[1:]
        T_hat = cp_to_tensor(scaled_factors)

        if ds.W_tensors is not None and ds.W_tensors[t_idx] is not None:
            W = ds.W_tensors[t_idx]
            diff = W * (T_hat - T)
        else:
            diff = T_hat - T
        f += 0.5 * float(np.sum(diff ** 2))

        # gradients par mode
        # d/dA_scaled(n) = diff_(n) * KR(others in natural order)
        for d in range(T.ndim):
            others = [scaled_factors[j] for j in range(T.ndim) if j != d]
            kr = khatri_rao(others)
            G = unfold(diff, d) @ kr
            if d == 0:
                # dF/dA : somme sur r -> G * lam (chain rule A_scaled = A * lam)
                grads[0] += G * lam[None, :]
                # dF/dlam_r = sum_i A[i, r] * G[i, r]
                grads[lam_idx] += np.sum(A * G, axis=0)
            else:
                grads[mode_indices[d]] += G

        # pénalité L1 sur lam : beta * sum sqrt(lam^2 + eps)
        f += beta * float(np.sum(np.sqrt(lam ** 2 + eps)))
        grads[lam_idx] += beta * lam / np.sqrt(lam ** 2 + eps)

    # --- Matrices ---
    for m_idx, M in enumerate(ds.matrices):
        sig = vars_[idx]
        sig_idx = idx
        idx += 1
        V = vars_[idx]
        V_idx = idx
        idx += 1

        nV = norm_cols(V)
        f += alpha * float(np.sum((nV - 1.0) ** 2))
        safeV = np.where(nV > 1e-12, nV, 1.0)
        grads[V_idx] += 2 * alpha * (V * ((nV - 1.0) / safeV)[None, :])

        A_sc = A * sig[None, :]
        M_hat = A_sc @ V.T
        if ds.W_matrices is not None and ds.W_matrices[m_idx] is not None:
            Wm = ds.W_matrices[m_idx]
            diff = Wm * (M_hat - M)
        else:
            diff = M_hat - M
        f += 0.5 * float(np.sum(diff ** 2))

        # gradients
        grads[0] += (diff @ V) * sig[None, :]
        grads[sig_idx] += np.sum(A * (diff @ V), axis=0)
        grads[V_idx] += diff.T @ A_sc

        # L1 sur sig
        f += beta * float(np.sum(np.sqrt(sig ** 2 + eps)))
        grads[sig_idx] += beta * sig / np.sqrt(sig ** 2 + eps)

    return f, _pack(grads)


# ----------------------------------------------------------------------
# API publique
# ----------------------------------------------------------------------
def acmtf_opt(ds: CoupledDataset, R: int,
              beta: float = 1e-3,
              alpha: float = 1.0,
              eps: float = 1e-8,
              n_inits: int = 10,
              max_iter: int = 1000,
              tol: float = 1e-10,
              random_state: int = 0,
              warm_start_cmtf: bool = True,
              verbose: bool = False):
    """
    Lance ACMTF-OPT.

    Les blocs doivent être pré-normalisés (chacun divisé par sa norme de
    Frobenius) de sorte que alpha=1 soit une valeur raisonnable. beta=1e-3
    est la valeur recommandée par Acar et al. 2014.

    Retourne un dict contenant :
        A            : facteurs mode 0 (colonnes ~ unit-norm)
        lambdas      : liste de vecteurs poids pour chaque tenseur
        sigmas       : liste de vecteurs poids pour chaque matrice
        tensor_factors : liste de listes de facteurs par tenseur
        matrix_factors : liste de matrices V par matrice
        fval         : valeur finale
        history      : valeurs pour chaque random start
    """
    shapes = _acmtf_shapes(ds, R)
    n_params = sum(int(np.prod(s)) for _, s in shapes)
    rng = np.random.default_rng(random_state)

    # Initialisation "intelligente" : facteurs avec colonnes unit-norm,
    # poids λ et σ initialisés à 1/sqrt(R) pour éviter les minimums dégénérés.
    def _smart_init():
        parts = []
        for name, s in shapes:
            if name.startswith("lambda_") or name.startswith("sigma_"):
                parts.append(np.full(s, 1.0 / np.sqrt(R)))
            else:
                M_ = rng.standard_normal(s)
                M_ /= (np.linalg.norm(M_, axis=0, keepdims=True) + 1e-12)
                parts.append(M_.ravel())
        return np.concatenate(parts)

    def _cmtf_warm_init():
        """Warm-start depuis une solution CMTF-OPT non-régularisée :
        on extrait les facteurs, on normalise leurs colonnes à 1, et les poids
        absorbés (produits des normes) deviennent λ, σ."""
        from .cmtf import cmtf_opt
        cmtf_res = cmtf_opt(ds, R=R, n_inits=3, max_iter=400,
                            random_state=int(rng.integers(1 << 30)))
        # factors par bloc
        A_c = cmtf_res["A"].copy()
        # normalise A
        nA = np.linalg.norm(A_c, axis=0); nA[nA == 0] = 1.0
        A_n = A_c / nA
        # poids initiaux (seront répartis entre λ et σ selon contribution)
        parts_dict = {}
        parts_dict["A"] = A_n
        for t_idx, T in enumerate(ds.tensors):
            facs = cmtf_res["factors_tensors"][t_idx]
            # facs[0] == A_c (partagée) ; pour les modes d>0 on normalise
            mode_norms = [nA.copy()]
            normed = [A_n]
            for d in range(1, T.ndim):
                nd = np.linalg.norm(facs[d], axis=0); nd[nd == 0] = 1.0
                mode_norms.append(nd)
                normed.append(facs[d] / nd)
            lam0 = np.prod(np.stack(mode_norms, axis=0), axis=0)
            parts_dict[f"lambda_{t_idx}"] = lam0
            for d in range(1, T.ndim):
                parts_dict[f"T{t_idx}_mode{d}"] = normed[d]
        for m_idx, M in enumerate(ds.matrices):
            V_c = cmtf_res["factors_matrices"][m_idx]
            nV = np.linalg.norm(V_c, axis=0); nV[nV == 0] = 1.0
            V_n = V_c / nV
            sig0 = nA * nV
            parts_dict[f"sigma_{m_idx}"] = sig0
            parts_dict[f"V_{m_idx}"] = V_n
        # pack dans l'ordre des shapes
        parts = []
        for name, s in shapes:
            parts.append(parts_dict[name].ravel())
        return np.concatenate(parts)

    best = None
    history = []
    for init in range(n_inits):
        if warm_start_cmtf and init == 0:
            x0 = _cmtf_warm_init()
        else:
            x0 = _smart_init()
        try:
            res = minimize(
                _acmtf_fg, x0, args=(ds, R, shapes, beta, alpha, eps),
                method="L-BFGS-B", jac=True,
                options={"maxiter": max_iter, "ftol": tol, "gtol": 1e-10,
                         "disp": verbose},
            )
        except Exception as e:
            if verbose:
                print(f"[init {init}] échec : {e}")
            continue
        history.append(float(res.fun))
        if best is None or res.fun < best.fun:
            best = res

    vars_ = _unpack(best.x, shapes)
    A = vars_[0]

    lambdas = []
    tensor_factors = []
    idx = 1
    for T in ds.tensors:
        lam = vars_[idx]; idx += 1
        block = [A]
        for d in range(1, T.ndim):
            block.append(vars_[idx]); idx += 1
        lambdas.append(lam)
        tensor_factors.append(block)

    sigmas = []
    matrix_factors = []
    for M in ds.matrices:
        sig = vars_[idx]; idx += 1
        V = vars_[idx]; idx += 1
        sigmas.append(sig)
        matrix_factors.append(V)

    # normalisation finale des colonnes (pour interprétation propre)
    def normalize_cols(M, weights=None):
        n = np.linalg.norm(M, axis=0)
        n[n == 0] = 1.0
        return M / n, n

    A_n, _ = normalize_cols(A)

    return {
        "A": A_n,
        "A_raw": A,
        "lambdas": [np.abs(l) for l in lambdas],
        "sigmas":  [np.abs(s) for s in sigmas],
        "tensor_factors": tensor_factors,
        "matrix_factors": matrix_factors,
        "fval": float(best.fun),
        "history": history,
    }
