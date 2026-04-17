"""
main_reproduce_paper.py
=======================

Reproduction partielle de :
    Acar, Papalexakis, Gurdeniz, Rasmussen, Lawaetz, Nilsson, Bro (2014).
    "Structure-revealing data fusion". BMC Bioinformatics.

But : appliquer ACMTF au couplage NMR (tenseur mixtures x chemical shift
x gradient) + LC-MS (matrice mixtures x features) et vérifier que :

    1. les 5 chimiques des mélanges ressortent comme composantes partagées
       Val-Tyr-Val, Trp-Gly, Phe, Maltoheptaose — visibles dans NMR et LC-MS
       (λ et σ non nuls).
    2. le propanol ressort comme composante non-partagée : λ > 0 dans NMR,
       σ ≈ 0 dans LC-MS.
    3. la 6e composante capte le bruit résiduel.

Usage :
    python main_reproduce_paper.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

from data_loader import load_joda
from preprocessing import preprocess_block
from methods.cmtf import CoupledDataset
from methods.acmtf import acmtf_opt


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def main():
    data = load_joda("data")
    # Downsampling NMR d'un facteur 10
    if data.NMR is not None and data.NMR.shape[1] > 2000:
        data.NMR = data.NMR[:, ::10, :]
        print(f"[downsample] NMR -> {data.NMR.shape}")
    if data.NMR is None or data.LCMS is None:
        raise RuntimeError("NMR et LC-MS sont nécessaires pour cette reproduction.")

    print(f"[data] NMR  shape = {data.NMR.shape}")
    print(f"[data] LCMS shape = {data.LCMS.shape}")

    # LC-MS : on scale par l'écart-type
    lcms = data.LCMS - data.LCMS.mean(axis=0, keepdims=True)
    std = lcms.std(axis=0, keepdims=True); std[std == 0] = 1.0
    lcms = lcms / std

    # Normalisation Frobenius des deux blocs
    NMR_p,  _ = preprocess_block(data.NMR,  center=True,  normalize=True)
    LCMS_p, _ = preprocess_block(lcms,      center=False, normalize=True)

    ds = CoupledDataset(tensors=[NMR_p], matrices=[LCMS_p])

    # R = 6 (5 chimiques + 1 pour absorber le bruit LC-MS)
    R = 6
    print(f"\n[ACMTF] fit avec R={R}, β=1e-2, 10 random starts…")
    res = acmtf_opt(ds, R=R, beta=1e-2, alpha=1.0,
                    n_inits=10, max_iter=1500, random_state=42)

    lam = res["lambdas"][0]
    sig = res["sigmas"][0]
    print(f"\n  λ (NMR)   = {np.array2string(lam, precision=3)}")
    print(f"  σ (LC-MS) = {np.array2string(sig, precision=3)}")

    # Interprétation : on ordonne les composantes par leur poids global
    order = np.argsort(-(lam + sig))
    lam = lam[order]; sig = sig[order]
    print("\n  composantes triées :")
    for r in range(R):
        shared = "SHARED" if (lam[r] > 0.1 and sig[r] > 0.1) else (
            "NMR-only" if lam[r] > sig[r] else "LCMS-only"
        )
        print(f"    r={r+1}  λ={lam[r]:.3f}  σ={sig[r]:.3f}  [{shared}]")

    # Figure : poids λ et σ (reproduction Figure 12 du papier)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(R)
    w = 0.38
    ax.bar(x - w/2, lam, width=w, label=r"$\lambda$ (NMR)",   color="#2b5d9a")
    ax.bar(x + w/2, sig, width=w, label=r"$\sigma$ (LC-MS)", color="#c0392b")
    ax.set_xlabel("composante")
    ax.set_ylabel("poids")
    ax.set_title("ACMTF — poids des composantes (reproduction Acar 2014, Fig. 12)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"c{i+1}" for i in range(R)])
    ax.legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "reproduction_acmtf_weights.png")
    fig.savefig(out, dpi=130)
    print(f"\n[saved] {out}")

    # Si concentrations disponibles : on compare A (scores) à Y pour
    # identifier quelle composante correspond à quel chimique.
    if data.Y is not None:
        from scipy.stats import pearsonr
        A = res["A"][:, order]
        print("\n[match] Corrélation des composantes ACMTF avec concentrations :")
        names = data.analyte_names or [f"comp{c}" for c in range(data.Y.shape[1])]
        for r in range(R):
            best = max(((abs(pearsonr(A[:, r], data.Y[:, c])[0]), names[c])
                        for c in range(data.Y.shape[1])),
                       key=lambda t: t[0])
            print(f"    c{r+1} -> {best[1]:<18}  |ρ|={best[0]:.3f}")


if __name__ == "__main__":
    main()
