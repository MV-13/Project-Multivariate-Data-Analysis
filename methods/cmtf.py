from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

@dataclass
class CoupledDataset:
    tensors:   List[np.ndarray]            = field(default_factory=list)
    matrices:  List[np.ndarray]            = field(default_factory=list)
    W_tensors: Optional[List[Optional[np.ndarray]]]  = None
    W_matrices: Optional[List[Optional[np.ndarray]]] = None

    def __post_init__(self):
        self.tensors  = [np.asarray(X, dtype=float) for X in self.tensors]
        self.matrices = [np.asarray(M, dtype=float) for M in self.matrices]
        if self.W_tensors is None:
            self.W_tensors = [None] * len(self.tensors)
        if self.W_matrices is None:
            self.W_matrices = [None] * len(self.matrices)
        # cohérence du mode 0
        I = None
        for X in self.tensors + self.matrices:
            if I is None:
                I = X.shape[0]
            elif X.shape[0] != I:
                raise ValueError(
                    "Tous les blocs doivent partager le mode 0 (même n)."
                )
        self.I = I if I is not None else 0


def khatri_rao(matrices: List[np.ndarray]) -> np.ndarray:
    if not matrices:
        raise ValueError("khatri_rao : liste vide")
    R = matrices[0].shape[1]
    for m in matrices:
        if m.shape[1] != R:
            raise ValueError("Toutes les matrices doivent avoir le même R")
    out = matrices[0].astype(float, copy=True)
    for m in matrices[1:]:
        Iout = out.shape[0]
        J = m.shape[0]
        out = (out.reshape(Iout, 1, R) * m.reshape(1, J, R)).reshape(Iout * J, R)
    return out


def _unfold(T: np.ndarray, mode: int) -> np.ndarray:
    return np.moveaxis(T, mode, 0).reshape(T.shape[mode], -1)


def cp_to_tensor(factors: List[np.ndarray], weights: Optional[np.ndarray] = None) -> np.ndarray:
    R = factors[0].shape[1]
    shape = tuple(f.shape[0] for f in factors)
    T = np.zeros(shape, dtype=float)
    for r in range(R):
        outer = factors[0][:, r]
        if weights is not None:
            outer = outer * weights[r]
        for f in factors[1:]:
            outer = np.multiply.outer(outer, f[:, r])
        T = T + outer
    return T


def _impute(X: np.ndarray, W: Optional[np.ndarray],
            X_hat: np.ndarray) -> np.ndarray:
    if W is None:
        return X
    return np.where(W > 0, X, X_hat)

def _gram_hadamard(factors: List[np.ndarray]) -> np.ndarray:
    R = factors[0].shape[1]
    G = np.ones((R, R))
    for f in factors:
        G = G * (f.T @ f)
    return G


def _objective(ds: CoupledDataset, A: np.ndarray, tensor_factors: List[List[np.ndarray]],matrix_factors: List[np.ndarray]) -> float:
    f = 0.0
    for t, X in enumerate(ds.tensors):
        X_hat = cp_to_tensor(tensor_factors[t])
        W = ds.W_tensors[t]
        diff = (X - X_hat) if W is None else (X - X_hat) * W
        f += float(np.sum(diff ** 2))
    for m, M in enumerate(ds.matrices):
        M_hat = A @ matrix_factors[m].T
        W = ds.W_matrices[m]
        diff = (M - M_hat) if W is None else (M - M_hat) * W
        f += float(np.sum(diff ** 2))
    return f


def _normalizer(X: np.ndarray) -> float:
    n = float(np.linalg.norm(X))
    return n if n > 1e-12 else 1.0


def cmtf_opt(ds: CoupledDataset, R: int,
             n_inits: int = 3, max_iter: int = 300,
             tol: float = 1e-8, random_state: int = 0,
             verbose: bool = False) -> dict:
    rng = np.random.default_rng(random_state)
    I = ds.I
    if I == 0:
        raise ValueError("Aucun bloc dans le dataset")

    norms_t = [_normalizer(X) for X in ds.tensors]
    norms_m = [_normalizer(M) for M in ds.matrices]

    best = None
    for init in range(n_inits):
        sub = np.random.default_rng(rng.integers(0, 2**31 - 1))
        A = sub.standard_normal((I, R))
        tensor_factors: List[List[np.ndarray]] = []
        for X in ds.tensors:
            facs = [A]
            for d in X.shape[1:]:
                facs.append(sub.standard_normal((d, R)))
            tensor_factors.append(facs)
        matrix_factors = [sub.standard_normal((M.shape[1], R))
                          for M in ds.matrices]

        prev = np.inf
        for it in range(max_iter):
            # imputation des manquants (NaN dans X / valeurs masquées)
            tens_imp = []
            for t, X in enumerate(ds.tensors):
                W = ds.W_tensors[t]
                if W is None and not np.isnan(X).any():
                    tens_imp.append(X)
                else:
                    X_hat = cp_to_tensor(tensor_factors[t])
                    if W is None:
                        W_eff = (~np.isnan(X)).astype(float)
                        X_use = np.nan_to_num(X, nan=0.0)
                    else:
                        W_eff = W
                        X_use = np.nan_to_num(X, nan=0.0)
                    tens_imp.append(_impute(X_use, W_eff, X_hat))
            mat_imp = []
            for m, M in enumerate(ds.matrices):
                W = ds.W_matrices[m]
                if W is None and not np.isnan(M).any():
                    mat_imp.append(M)
                else:
                    M_hat = A @ matrix_factors[m].T
                    if W is None:
                        W_eff = (~np.isnan(M)).astype(float)
                        M_use = np.nan_to_num(M, nan=0.0)
                    else:
                        W_eff = W
                        M_use = np.nan_to_num(M, nan=0.0)
                    mat_imp.append(_impute(M_use, W_eff, M_hat))

            # mise à jour de A jointe sur tous les blocs
            LHS = np.zeros((R, R))
            RHS = np.zeros((I, R))
            for t, X in enumerate(tens_imp):
                facs = tensor_factors[t]
                others = facs[1:]
                LHS = LHS + _gram_hadamard(others)
                kr = khatri_rao(others)
                RHS = RHS + _unfold(X, 0) @ kr
            for m, M in enumerate(mat_imp):
                V = matrix_factors[m]
                LHS = LHS + V.T @ V
                RHS = RHS + M @ V
            A = RHS @ np.linalg.pinv(LHS + 1e-10 * np.eye(R))
            for facs in tensor_factors:
                facs[0] = A

            # mise à jour des autres modes pour chaque tenseur
            for t, X in enumerate(tens_imp):
                facs = tensor_factors[t]
                for n in range(1, len(facs)):
                    others = [facs[k] for k in range(len(facs)) if k != n]
                    G = _gram_hadamard(others)
                    kr = khatri_rao(others)
                    facs[n] = _unfold(X, n) @ kr @ np.linalg.pinv(G + 1e-10 * np.eye(R))

            # mise à jour des V_m
            AtA = A.T @ A
            inv_AtA = np.linalg.pinv(AtA + 1e-10 * np.eye(R))
            for m, M in enumerate(mat_imp):
                matrix_factors[m] = M.T @ A @ inv_AtA

            # objectif sur les vraies données (pas imputées)
            fval = _objective(ds, A, tensor_factors, matrix_factors)
            # normalisation pour la comparaison entre dataset
            denom = sum(n ** 2 for n in norms_t) + sum(n ** 2 for n in norms_m)
            fval_norm = fval / max(denom, 1e-12)
            if verbose and it % 20 == 0:
                print(f"  init {init} it {it:4d}  fval={fval_norm:.6f}")
            if abs(prev - fval_norm) < tol * max(prev, 1.0):
                break
            prev = fval_norm

        if best is None or fval_norm < best["fval"]:
            best = {
                "A": A.copy(),
                "factors_tensors": [[f.copy() for f in facs]
                                    for facs in tensor_factors],
                "factors_matrices": [V.copy() for V in matrix_factors],
                "fval": float(fval_norm),
            }
    return best
