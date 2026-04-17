"""
evaluation.py
=============

Métriques et procédures d'évaluation :
    - Factor Match Score (FMS), reprend la définition de Acar et al. 2011, éq. (4).
    - R², RMSE par constituant.
    - Validation croisée leave-one-out pour les méthodes supervisées.
"""

from __future__ import annotations

import numpy as np
from itertools import permutations


def factor_match_score(true_factors: list[np.ndarray],
                       est_factors: list[np.ndarray],
                       true_weights: np.ndarray | None = None,
                       est_weights:  np.ndarray | None = None) -> float:
    """
    FMS entre deux jeux de facteurs CP. Chaque facteur est une matrice
    (d_n, R).
    """
    def _norm_cols(M):
        n = np.linalg.norm(M, axis=0)
        n[n == 0] = 1.0
        return M / n

    F_t = [_norm_cols(F) for F in true_factors]
    F_e = [_norm_cols(F) for F in est_factors]
    R = F_t[0].shape[1]

    best = -np.inf
    for perm in permutations(range(R)):
        score = 1.0
        for r_true, r_est in enumerate(perm):
            prod = 1.0
            for F_true, F_est in zip(F_t, F_e):
                prod *= abs(F_true[:, r_true] @ F_est[:, r_est])
            if true_weights is not None and est_weights is not None:
                a, b = true_weights[r_true], est_weights[r_est]
                prod *= 1.0 - abs(a - b) / (max(abs(a), abs(b)) + 1e-12)
            score = min(score, prod)
        if score > best:
            best = score
    return float(best)


def r2_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """R² et RMSE colonne par colonne + globaux."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim == 1:
        y_true = y_true[:, None]; y_pred = y_pred[:, None]

    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    return {
        "r2_per_col":   r2,
        "rmse_per_col": rmse,
        "r2_mean":      float(r2.mean()),
        "rmse_mean":    float(rmse.mean()),
    }


def leave_one_out_cv(X: np.ndarray, Y: np.ndarray, fit_predict) -> dict:
    """
    LOO-CV : pour chaque échantillon i on entraîne sur les autres et on
    prédit Y[i].
    """
    n = X.shape[0]
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Y_pred = np.zeros_like(Y)
    for i in range(n):
        idx_tr = np.array([j for j in range(n) if j != i])
        X_tr, Y_tr = X[idx_tr], Y[idx_tr]
        X_te = X[i:i+1]
        Y_pred[i] = fit_predict(X_tr, Y_tr, X_te).ravel()
    return r2_rmse(Y, Y_pred) | {"Y_pred": Y_pred}
