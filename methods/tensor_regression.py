from __future__ import annotations
from typing import List
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

def _khatri_rao(matrices: List[np.ndarray]) -> np.ndarray:
    R = matrices[0].shape[1]
    out = matrices[0].astype(float, copy=True)
    for m in matrices[1:]:
        Iout = out.shape[0]; J = m.shape[0]
        out = (out.reshape(Iout, 1, R) * m.reshape(1, J, R)).reshape(Iout * J, R)
    return out


def cp_tensor_regression(X: np.ndarray, y: np.ndarray, rank: int = 3, max_iter: int = 60, l2: float = 1e-1, tol: float = 1e-7, random_state: int = 0) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[0]
    dims = X.shape[1:]
    K = len(dims)

    y_mean = y.mean()
    yc = y - y_mean
    X_mean = X.mean(axis=0, keepdims=True)
    Xc = X - X_mean

    rng = np.random.default_rng(random_state)
    factors = [rng.standard_normal((d, rank)) * 0.1 for d in dims]

    prev_loss = np.inf
    for it in range(max_iter):
        for k in range(K):
            M = Xc  

            other_dims = [d for j, d in enumerate(dims) if j != k]
            other_factors = [factors[j] for j in range(K) if j != k]

            Xperm = np.moveaxis(Xc, 1 + k, 1) 
            d_k = Xperm.shape[1]
            other_size = int(np.prod(other_dims))
            X2 = Xperm.reshape(n * d_k, other_size)
            kr = _khatri_rao(other_factors)  
            M_k_flat = X2 @ kr
            M_k = M_k_flat.reshape(n, d_k, rank)

            H_k = M_k.reshape(n, d_k * rank) 
            G = H_k.T @ H_k + l2 * np.eye(d_k * rank)
            rhs = H_k.T @ yc
            U_k_vec = np.linalg.solve(G, rhs)
            factors[k] = U_k_vec.reshape(d_k, rank)

        B = _cp_to_tensor(factors)
        pred = Xc.reshape(n, -1) @ B.ravel()
        loss = float(np.sum((yc - pred) ** 2)) + l2 * float(np.sum(B ** 2))
        if abs(prev_loss - loss) < tol * max(prev_loss, 1.0):
            break
        prev_loss = loss

    return {
        "factors": factors,
        "X_mean": X_mean,
        "y_mean": float(y_mean),
        "rank": rank,
    }


def _cp_to_tensor(factors: List[np.ndarray]) -> np.ndarray:
    R = factors[0].shape[1]
    shape = tuple(f.shape[0] for f in factors)
    T = np.zeros(shape, dtype=float)
    for r in range(R):
        outer = factors[0][:, r]
        for f in factors[1:]:
            outer = np.multiply.outer(outer, f[:, r])
        T = T + outer
    return T


def cp_tensor_regression_predict(model: dict, X_te: np.ndarray) -> np.ndarray:
    X_te = np.asarray(X_te, dtype=float)
    Xc = X_te - model["X_mean"]
    B = _cp_to_tensor(model["factors"])
    n = Xc.shape[0]
    return Xc.reshape(n, -1) @ B.ravel() + model["y_mean"]

def ridge_flat(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    model = Ridge(alpha=alpha)
    model.fit(X_tr.reshape(X_tr.shape[0], -1), Y_tr)
    return model.predict(X_te.reshape(X_te.shape[0], -1))


def pls_flat(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, n_components: int = 5) -> np.ndarray:
    X_tr = np.asarray(X_tr, dtype=float)
    X_te = np.asarray(X_te, dtype=float)
    n_components = min(n_components, X_tr.shape[0] - 1)
    model = PLSRegression(n_components=max(n_components, 1))
    model.fit(X_tr.reshape(X_tr.shape[0], -1), Y_tr)
    return model.predict(X_te.reshape(X_te.shape[0], -1))

def npls_fit(X: np.ndarray, Y: np.ndarray,
             n_components: int = 5) -> dict:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    Xm, Ym = X.mean(axis=0, keepdims=True), Y.mean(axis=0, keepdims=True)
    Xres, Yres = X - Xm, Y - Ym
    n, J, K = Xres.shape
    p = Yres.shape[1]

    Wj = np.zeros((J, n_components))
    Wk = np.zeros((K, n_components))
    T  = np.zeros((n, n_components))
    Q  = np.zeros((p, n_components))

    for a in range(n_components):
        Z = np.einsum("ijk,iq->jkq", Xres, Yres)
        U, _, _ = np.linalg.svd(Z.reshape(J * K, p), full_matrices=False)
        z1 = U[:, 0].reshape(J, K)
        Uz, _, Vtz = np.linalg.svd(z1, full_matrices=False)
        wj, wk = Uz[:, 0], Vtz[0, :]
        t = np.einsum("ijk,j,k->i", Xres, wj, wk)
        denom = float(t @ t) + 1e-12
        q = (Yres.T @ t) / denom

        Wj[:, a], Wk[:, a], T[:, a], Q[:, a] = wj, wk, t, q
        Xres = Xres - np.einsum("i,j,k->ijk", t, wj, wk)
        Yres = Yres - np.outer(t, q)

    return {"Wj": Wj, "Wk": Wk, "Q": Q, "T": T, "X_mean": Xm, "Y_mean": Ym}


def npls_predict(model: dict, X_te: np.ndarray) -> np.ndarray:
    X_te = np.asarray(X_te, dtype=float)
    Xres = X_te - model["X_mean"]
    n_comp = model["Wj"].shape[1]
    T_te = np.zeros((Xres.shape[0], n_comp))
    for a in range(n_comp):
        wj, wk = model["Wj"][:, a], model["Wk"][:, a]
        t = np.einsum("ijk,j,k->i", Xres, wj, wk)
        T_te[:, a] = t
        Xres = Xres - np.einsum("i,j,k->ijk", t, wj, wk)
    return T_te @ model["Q"].T + model["Y_mean"]


def npls_flat(X_tr: np.ndarray, Y_tr: np.ndarray, X_te: np.ndarray, n_components: int = 3) -> np.ndarray:
    model = npls_fit(X_tr, Y_tr, n_components=n_components)
    return npls_predict(model, X_te)
