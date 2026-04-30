from __future__ import annotations
import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac

def _to_np(x):
    return np.asarray(x, dtype=float)


def run_parafac(X: np.ndarray, rank: int,
                n_iter_max: int = 300, tol: float = 1e-7,
                init: str = "random", random_state: int = 0,
                mask: np.ndarray | None = None,
                normalize_factors: bool = True,
                ) -> tuple[np.ndarray, list[np.ndarray], float]:
    X_np = _to_np(X)
    Xt = tl.tensor(X_np)
    kw = dict(rank=rank, n_iter_max=n_iter_max, tol=tol, init=init, normalize_factors=normalize_factors,random_state=random_state)
    if mask is not None:
        kw["mask"] = tl.tensor(_to_np(mask))
    cp = parafac(Xt, **kw)
    weights, factors = cp
    weights = _to_np(weights)
    factors = [_to_np(f) for f in factors]

    Xhat = tl.cp_to_tensor((tl.tensor(weights), [tl.tensor(f) for f in factors]))
    Xhat = _to_np(Xhat)
    if mask is not None:
        m = _to_np(mask)
        rel = float(np.linalg.norm((X_np - Xhat) * m) /
                    (np.linalg.norm(X_np * m) + 1e-12))
    else:
        rel = float(np.linalg.norm(X_np - Xhat) /
                    (np.linalg.norm(X_np) + 1e-12))
    return weights, factors, rel


def best_parafac(X: np.ndarray, rank: int, n_inits: int = 5, **kw) -> tuple[np.ndarray, list[np.ndarray], float]:
    best = (None, None, np.inf)
    for s in range(n_inits):
        try:
            w, fac, err = run_parafac(X, rank=rank, random_state=s, **kw)
        except Exception:
            continue
        if err < best[2]:
            best = (w, fac, err)
    if best[0] is None:
        raise RuntimeError("PARAFAC a échoué pour toutes les initialisations")
    return best
