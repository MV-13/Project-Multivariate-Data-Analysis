"""
plot_results_paper.py
=====================

Génère un ensemble de figures pour illustrer la reproduction du papier
Acar et al. 2014 sur le jeu JODA.

Figures produites dans results/ :
    1. reproduction_weights.png        — poids λ (NMR) vs σ (LC-MS) par composante
    2. reproduction_scores_vs_conc.png — scatter A vs concentrations vraies
    3. reproduction_nmr_signatures.png — spectres NMR extraits (mode chemical shift)
    4. reproduction_nmr_decays.png     — courbes de décroissance (mode gradient)
    5. reproduction_lcms_loadings.png  — loadings LC-MS par composante
    6. reproduction_A_vs_design.png    — scores A comparés au design expérimental

Usage :
    python plot_results_paper.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from data_loader import load_joda
from preprocessing import preprocess_block
from methods.cmtf import CoupledDataset
from methods.acmtf import acmtf_opt


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------------------------------------------
def match_components_to_analytes(A: np.ndarray, Y_true: np.ndarray,
                                  analyte_names: list) -> dict:
    """
    Pour chaque composante r, trouve l'analyte le mieux corrélé.
    """
    R = A.shape[1]
    n_analytes = Y_true.shape[1]
    mapping = {}
    for r in range(R):
        best_rho, best_c = 0.0, -1
        for c in range(n_analytes):
            rho = pearsonr(A[:, r], Y_true[:, c])[0]
            if abs(rho) > abs(best_rho):
                best_rho, best_c = rho, c
        mapping[r] = (analyte_names[best_c] if best_c >= 0 else "?",
                      float(best_rho), best_c)
    return mapping


def align_sign(A: np.ndarray, Y_true: np.ndarray, mapping: dict,
               factors_to_flip: list) -> np.ndarray:
    """
    Aligne le signe des composantes avec celui des concentrations vraies
    (pour rendre les scatter plots visuellement corrects).
    """
    A_aligned = A.copy()
    for r, (_, rho, _) in mapping.items():
        if rho < 0:
            A_aligned[:, r] *= -1
            for F in factors_to_flip:
                F[:, r] *= -1
    return A_aligned


# ----------------------------------------------------------------------
def main():
    data = load_joda("data")
    if data.NMR is None or data.LCMS is None:
        raise RuntimeError("NMR et LC-MS sont requis.")

    if data.NMR.shape[1] > 2000:
        data.NMR = data.NMR[:, ::10, :]
        print(f"[downsample] NMR -> {data.NMR.shape}")

    nmr_ppm = None
    if data.nmr_axes and data.nmr_axes.get("chemical_shift") is not None:
        full_ppm = data.nmr_axes["chemical_shift"]
        if len(full_ppm) >= data.NMR.shape[1] * 10:
            nmr_ppm = full_ppm[::10][:data.NMR.shape[1]]
        else:
            nmr_ppm = np.arange(data.NMR.shape[1])
    else:
        nmr_ppm = np.arange(data.NMR.shape[1])

    # Prétraitement : centrage + auto-scaling LC-MS + normalisation Fro
    lcms = data.LCMS - data.LCMS.mean(axis=0, keepdims=True)
    std = lcms.std(axis=0, keepdims=True); std[std == 0] = 1.0
    lcms = lcms / std

    NMR_p,  _ = preprocess_block(data.NMR,  center=True,  normalize=True)
    LCMS_p, _ = preprocess_block(lcms,      center=False, normalize=True)

    ds = CoupledDataset(tensors=[NMR_p], matrices=[LCMS_p])

    # Fit ACMTF
    R = 6
    print(f"[ACMTF] R={R}, β=1e-2, 10 random starts...")
    res = acmtf_opt(ds, R=R, beta=1e-2, alpha=1.0,
                    n_inits=10, max_iter=1500, random_state=42)

    lam = res["lambdas"][0]
    sig = res["sigmas"][0]
    A = res["A"]
    B = res["tensor_factors"][0][1]  # chemical shift mode
    C = res["tensor_factors"][0][2]  # gradient mode
    V = res["matrix_factors"][0]     # LC-MS features mode

    order = np.argsort(-(lam + sig))
    lam = lam[order]; sig = sig[order]
    A = A[:, order]
    B = B[:, order]; C = C[:, order]; V = V[:, order]

    # Match composante -> analyte
    mapping = match_components_to_analytes(A, data.Y, data.analyte_names)
    print("\n[match] Composantes -> analytes :")
    for r, (name, rho, _) in mapping.items():
        print(f"  c{r+1}: {name:<18} ρ={rho:+.3f}  λ={lam[r]:.3f}  σ={sig[r]:.3f}")

    A = align_sign(A, data.Y, mapping, [B, C, V])

    comp_labels = []
    for r in range(R):
        name, rho, _ = mapping[r]
        if abs(rho) > 0.6:
            comp_labels.append(f"c{r+1}: {name}")
        else:
            # composante non identifiée = bruit
            if lam[r] < 0.1 or sig[r] < 0.1:
                side = "NMR-only" if lam[r] > sig[r] else "LCMS-only"
                comp_labels.append(f"c{r+1}: {side}")
            else:
                comp_labels.append(f"c{r+1}: noise")

    # ==================================================================
    # Figure 1 — poids λ et σ
    # ==================================================================
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(R); w = 0.38
    ax.bar(x - w/2, lam, width=w, label=r"$\lambda$ (NMR)",   color="#2b5d9a")
    ax.bar(x + w/2, sig, width=w, label=r"$\sigma$ (LC-MS)", color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(comp_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("poids")
    ax.set_title("ACMTF — poids des composantes (reproduction Acar 2014, Fig. 12)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reproduction_weights.png"), dpi=130)
    plt.close(fig)

    # ==================================================================
    # Figure 2 — scores A vs concentrations vraies
    # ==================================================================
    n_ana = data.Y.shape[1]
    fig, axes = plt.subplots(1, n_ana, figsize=(3.2 * n_ana, 3.2),
                             sharey=False)
    if n_ana == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        best_r, best_rho = -1, 0.0
        for r in range(R):
            rho = pearsonr(A[:, r], data.Y[:, c])[0]
            if abs(rho) > abs(best_rho):
                best_r, best_rho = r, rho
        if best_r == -1:
            continue

        a = A[:, best_r]
        y = data.Y[:, c]
        a_n = (a - a.min()) / (a.max() - a.min() + 1e-12)
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)

        ax.scatter(y_n, a_n, s=35, c="#2b5d9a", edgecolor="k", linewidth=0.4)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("concentration vraie (norm.)")
        if c == 0:
            ax.set_ylabel("score ACMTF (norm.)")
        ax.set_title(f"{data.analyte_names[c]}\nc{best_r+1}, ρ={best_rho:+.3f}",
                     fontsize=10)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    plt.suptitle("Scores ACMTF vs concentrations vraies", y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reproduction_scores_vs_conc.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 3 — signatures NMR (mode chemical shift)
    # ==================================================================
    fig, axes = plt.subplots(R, 1, figsize=(10, 1.6 * R), sharex=True)
    if R == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, R))
    for r, ax in enumerate(axes):
        ax.plot(nmr_ppm, B[:, r], color=colors[r], lw=0.8)
        ax.set_ylabel(f"b{r+1}", rotation=0, labelpad=20, fontsize=9)
        ax.set_title(comp_labels[r], fontsize=9, loc="left")
        ax.grid(alpha=0.3)
        if nmr_ppm[0] > nmr_ppm[-1] or (nmr_ppm[0] < nmr_ppm[-1] and nmr_ppm.max() > 20):
            pass
        else:
            ax.invert_xaxis() 
    axes[-1].set_xlabel("chemical shift (ppm) ou index")
    plt.suptitle("Signatures NMR extraites par ACMTF (mode chemical shift)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reproduction_nmr_signatures.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 4 — courbes de décroissance NMR (mode gradient)
    # ==================================================================
    fig, ax = plt.subplots(figsize=(7, 4))
    K = C.shape[0]
    x_grad = np.arange(1, K + 1)
    for r in range(R):
        ax.plot(x_grad, C[:, r], marker="o", color=colors[r],
                label=comp_labels[r], lw=1.5, markersize=5)
    ax.set_xlabel("gradient level")
    ax.set_ylabel("intensité")
    ax.set_title("Courbes de décroissance par gradient (mode 3 du NMR)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reproduction_nmr_decays.png"), dpi=130)
    plt.close(fig)

    # ==================================================================
    # Figure 5 — loadings LC-MS
    # ==================================================================
    fig, axes = plt.subplots(R, 1, figsize=(10, 1.5 * R), sharex=True)
    if R == 1:
        axes = [axes]
    for r, ax in enumerate(axes):
        ax.plot(np.arange(V.shape[0]), V[:, r], color=colors[r], lw=0.8)
        ax.set_ylabel(f"v{r+1}", rotation=0, labelpad=20, fontsize=9)
        ax.set_title(comp_labels[r], fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("index feature LC-MS")
    plt.suptitle("Loadings LC-MS extraits par ACMTF", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reproduction_lcms_loadings.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 6 — scores A comparés au design (Figure 14 du papier)
    # ==================================================================
    fig, axes = plt.subplots(1, n_ana, figsize=(3.2 * n_ana, 3.2))
    if n_ana == 1:
        axes = [axes]
    x_samples = np.arange(1, data.n_samples + 1)
    for c, ax in enumerate(axes):
        best_r, best_rho = -1, 0.0
        for r in range(R):
            rho = pearsonr(A[:, r], data.Y[:, c])[0]
            if abs(rho) > abs(best_rho):
                best_r, best_rho = r, rho
        if best_r == -1:
            continue

        a = A[:, best_r]; y = data.Y[:, c]
        a_n = a / (np.linalg.norm(a) + 1e-12)
        y_n = y / (np.linalg.norm(y) + 1e-12)
        if best_rho < 0:
            a_n *= -1

        ax.plot(x_samples, y_n, "o-", color="#2b5d9a", label="vraie", ms=4, lw=1)
        ax.plot(x_samples, a_n, "o-", color="#c0392b", label="ACMTF", ms=4, lw=1)
        ax.set_title(f"{data.analyte_names[c]}\nρ={best_rho:+.3f}", fontsize=10)
        ax.set_xlabel("échantillon")
        if c == 0:
            ax.set_ylabel("intensité (norm.)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Composantes ACMTF vs design expérimental (reproduction Fig. 14)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "reproduction_A_vs_design.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n[done] 6 figures sauvegardées dans {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith("reproduction_"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
