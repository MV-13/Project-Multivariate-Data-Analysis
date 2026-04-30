from __future__ import annotations
import numpy as np

def _shrunk_inv(X: np.ndarray, tau: float) -> np.ndarray | None:
    if tau >= 1.0 - 1e-12:
        return None
    n, p = X.shape
    M = (1.0 - tau) * (X.T @ X) / max(n - 1, 1) + tau * np.eye(p)
    return np.linalg.pinv(M)

def rgcca(blocks: list[np.ndarray],
          tau: list[float] | None = None,
          n_comp: int = 1,
          design: np.ndarray | None = None,
          scheme: str = "centroid",
          max_iter: int = 200,
          tol: float = 1e-8) -> dict:
    K = len(blocks)
    if tau is None:
        tau = [1.0] * K
    if design is None:
        design = np.ones((K, K)) - np.eye(K)
    design = np.asarray(design, dtype=float)

    X = []
    for B in blocks:
        B = np.asarray(B, dtype=float)
        X.append(B - B.mean(axis=0, keepdims=True))
    n = X[0].shape[0]

    if scheme == "horst":
        gprime = lambda v: 1.0
    elif scheme == "centroid":
        gprime = lambda v: np.sign(v) if v != 0 else 1.0
    elif scheme == "factorial":
        gprime = lambda v: 2.0 * v
    else:
        raise ValueError(f"scheme inconnu : {scheme}")

    resid = [Xk.copy() for Xk in X]
    scores = [np.zeros((n, n_comp)) for _ in range(K)]
    weights = [np.zeros((Xk.shape[1], n_comp)) for Xk in X]

    for c in range(n_comp):
        inv_M = [_shrunk_inv(resid[k], tau[k]) for k in range(K)]

        # init via SVD du résidu
        a = []
        for k in range(K):
            if min(resid[k].shape) == 0:
                a.append(np.zeros(resid[k].shape[1]))
                continue
            try:
                _, _, Vt = np.linalg.svd(resid[k], full_matrices=False)
                a.append(Vt[0])
            except np.linalg.LinAlgError:
                v = np.random.default_rng(c).standard_normal(resid[k].shape[1])
                a.append(v / (np.linalg.norm(v) + 1e-12))

        prev_crit = -np.inf
        for _ in range(max_iter):
            y = [resid[k] @ a[k] for k in range(K)]
            new_a = []
            for k in range(K):
                z = np.zeros(n)
                for l in range(K):
                    if l == k or design[k, l] == 0:
                        continue
                    w = gprime(float(y[k] @ y[l]))
                    z = z + design[k, l] * w * y[l]
                t = resid[k].T @ z
                if inv_M[k] is not None:
                    t = inv_M[k] @ t
                nrm = np.linalg.norm(t)
                if nrm < 1e-12:
                    new_a.append(a[k])
                else:
                    new_a.append(t / nrm)
            a = new_a

            # critère : sum_{k<l} g(<y_k, y_l>)
            yy = [resid[k] @ a[k] for k in range(K)]
            crit = 0.0
            for k in range(K):
                for l in range(k + 1, K):
                    if design[k, l] == 0:
                        continue
                    v = float(yy[k] @ yy[l])
                    if scheme == "horst":
                        crit += v
                    elif scheme == "centroid":
                        crit += abs(v)
                    else:  # factorial
                        crit += v * v
            if abs(crit - prev_crit) < tol:
                break
            prev_crit = crit

        for k in range(K):
            yk = resid[k] @ a[k]
            scores[k][:, c] = yk
            weights[k][:, c] = a[k]
            denom = float(yk @ yk)
            if denom > 1e-12:
                resid[k] = resid[k] - np.outer(yk, (yk @ resid[k]) / denom)

    return {"scores": scores, "weights": weights}

def _soft_threshold(v: np.ndarray, alpha: float) -> np.ndarray:
    return np.sign(v) * np.maximum(np.abs(v) - alpha, 0.0)


def _l1_proj_to_unit(v: np.ndarray, c1: float) -> np.ndarray:
    p = v.shape[0]
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        return v
    if c1 is None or c1 >= 1.0:
        return v / nrm

    target_l1 = c1 * np.sqrt(p)
    v_n = v / nrm
    if np.sum(np.abs(v_n)) <= target_l1:
        return v_n

    lo, hi = 0.0, float(np.max(np.abs(v)))
    for _ in range(60):
        mid = (lo + hi) / 2
        s = _soft_threshold(v, mid)
        ns = np.linalg.norm(s)
        if ns < 1e-12:
            hi = mid
            continue
        l1_ratio = float(np.sum(np.abs(s)) / ns)
        if l1_ratio > target_l1:
            lo = mid
        else:
            hi = mid
    s = _soft_threshold(v, (lo + hi) / 2)
    ns = np.linalg.norm(s)
    return s / ns if ns > 1e-12 else v / nrm


def sgcca(blocks: list[np.ndarray],
          tau: list[float] | None = None,
          n_comp: int = 1,
          c1: list[float] | None = None,
          design: np.ndarray | None = None,
          scheme: str = "centroid",
          max_iter: int = 200,
          tol: float = 1e-8) -> dict:

    K = len(blocks)
    if tau is None:
        tau = [1.0] * K
    if c1 is None:
        c1 = [1.0] * K
    if design is None:
        design = np.ones((K, K)) - np.eye(K)
    design = np.asarray(design, dtype=float)

    X = []
    for B in blocks:
        B = np.asarray(B, dtype=float)
        X.append(B - B.mean(axis=0, keepdims=True))
    n = X[0].shape[0]

    if scheme == "horst":
        gprime = lambda v: 1.0
    elif scheme == "centroid":
        gprime = lambda v: np.sign(v) if v != 0 else 1.0
    elif scheme == "factorial":
        gprime = lambda v: 2.0 * v
    else:
        raise ValueError(f"scheme inconnu : {scheme}")

    inv_M = [_shrunk_inv(X[k], tau[k]) for k in range(K)]
    resid = [Xk.copy() for Xk in X]
    scores = [np.zeros((n, n_comp)) for _ in range(K)]
    weights = [np.zeros((Xk.shape[1], n_comp)) for Xk in X]

    for c in range(n_comp):
        a = []
        for k in range(K):
            try:
                _, _, Vt = np.linalg.svd(resid[k], full_matrices=False)
                a.append(_l1_proj_to_unit(Vt[0], c1[k]))
            except np.linalg.LinAlgError:
                v = np.random.default_rng(c).standard_normal(resid[k].shape[1])
                a.append(_l1_proj_to_unit(v, c1[k]))

        prev_crit = -np.inf
        for _ in range(max_iter):
            y = [resid[k] @ a[k] for k in range(K)]
            new_a = []
            for k in range(K):
                z = np.zeros(n)
                for l in range(K):
                    if l == k or design[k, l] == 0:
                        continue
                    w = gprime(float(y[k] @ y[l]))
                    z = z + design[k, l] * w * y[l]
                t = resid[k].T @ z
                if inv_M[k] is not None:
                    t = inv_M[k] @ t
                # projection L1 + L2 (sparse + unit norm)
                new_a.append(_l1_proj_to_unit(t, c1[k]))
            a = new_a

            yy = [resid[k] @ a[k] for k in range(K)]
            crit = 0.0
            for k in range(K):
                for l in range(k + 1, K):
                    if design[k, l] == 0:
                        continue
                    v = float(yy[k] @ yy[l])
                    if scheme == "horst":
                        crit += v
                    elif scheme == "centroid":
                        crit += abs(v)
                    else:
                        crit += v * v
            if abs(crit - prev_crit) < tol:
                break
            prev_crit = crit

        for k in range(K):
            yk = resid[k] @ a[k]
            scores[k][:, c] = yk
            weights[k][:, c] = a[k]
            denom = float(yk @ yk)
            if denom > 1e-12:
                resid[k] = resid[k] - np.outer(yk, (yk @ resid[k]) / denom)

    # taux de parcimonie effectif par bloc
    sparsity = [float((np.abs(W) < 1e-10).mean()) for W in weights]
    return {"scores": scores, "weights": weights, "sparsity": sparsity}
