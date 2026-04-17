"""
preprocessing.py
================

Prétraitements classiques pour données multi-bloc et tensorielles :
    - centrage par mode (mean-centering)
    - scaling "within" (par variable ou par tranche)
    - normalisation d'un bloc par sa norme de Frobenius (équi-pondération)

Références : Smilde, Bro, Geladi, *Multi-way Analysis* (2004).
"""

from __future__ import annotations

import numpy as np


def center_mode(X: np.ndarray, mode: int = 0) -> np.ndarray:
    """Centre un tenseur le long d'un mode."""
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=mode, keepdims=True)
    return X - mean


def scale_within_mode(X: np.ndarray, mode: int = 0) -> np.ndarray:
    """
    Scaling within mode : chaque "tranche" le long de `mode` est divisée par
    son écart-type rms. Utile pour homogénéiser les variables NMR.
    """
    X = np.asarray(X, dtype=float)
    # on calcule l'écart-type sur tous les autres modes
    axes = tuple(i for i in range(X.ndim) if i != mode)
    std = np.sqrt((X ** 2).mean(axis=axes, keepdims=True))
    std[std == 0] = 1.0
    return X / std


def frobenius_normalize(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Divise X par sa norme de Frobenius. Retourne (X_normé, norme)."""
    nrm = float(np.linalg.norm(X))
    if nrm == 0:
        return X.copy(), 1.0
    return X / nrm, nrm


def preprocess_block(X: np.ndarray,
                     center: bool = True,
                     scale: bool = False,
                     normalize: bool = True) -> tuple[np.ndarray, float]:
    """Pipeline standard pour un bloc multi-way : center -> scale -> Fro-normalize."""
    if center:
        X = center_mode(X, mode=0)
    if scale:
        X = scale_within_mode(X, mode=0)
    if normalize:
        X, nrm = frobenius_normalize(X)
    else:
        nrm = 1.0
    return X, nrm


def standardize_Y(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre-réduit Y pour la régression. Retourne (Yc, mean, std)."""
    mu = Y.mean(axis=0, keepdims=True)
    sd = Y.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    return (Y - mu) / sd, mu, sd
