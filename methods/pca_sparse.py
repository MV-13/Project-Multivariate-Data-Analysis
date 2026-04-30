from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA, SparsePCA

def _flatten(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X

def run_pca(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Xf = _flatten(X)
    n_components = min(n_components, *Xf.shape)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xf)
    loadings = pca.components_.T
    evr = pca.explained_variance_ratio_
    return scores, loadings, evr


def run_sparse_pca(X: np.ndarray, n_components: int = 5, alpha: float = 1.0, max_iter: int = 200,random_state: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    Xf = _flatten(X)
    n_components = min(n_components, *Xf.shape)
    sp = SparsePCA(n_components=n_components, alpha=alpha,
                   max_iter=max_iter, random_state=random_state)
    scores = sp.fit_transform(Xf)
    loadings = sp.components_.T
    sparsity = float(np.mean(np.abs(loadings) < 1e-10))
    return scores, loadings, sparsity
