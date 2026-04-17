"""
plot_results_unsupervised.py
============================

Génère les figures pour l'analyse tri-blocs non supervisée du jeu JODA
(NMR + EEM + LC-MS avec ACMTF).

Figures produites dans results/ :
    1. unsup_weights.png           — poids λ_NMR, λ_EEM, σ_LCMS par composante
    2. unsup_scores_vs_conc.png    — scatter scores ACMTF vs concentrations
    3. unsup_A_vs_design.png       — scores A comparés au design expérimental
    4. unsup_nmr_signatures.png    — signatures NMR par composante
    5. unsup_eem_loadings.png      — loadings EEM (excitation + émission)
    6. unsup_lcms_loadings.png     — loadings LC-MS
    7. unsup_heatmap_weights.png   — heatmap des poids normalisés (identité
                                      "analytique" des composantes)
    8. unsup_cmtf_vs_acmtf.png     — comparaison fval CMTF vs ACMTF

Usage :
    python plot_results_unsupervised.py
"""

from __future__ import annotations

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from data_loader import load_joda
from methods.cmtf import cmtf_opt, CoupledDataset
from methods.acmtf import acmtf_opt

warnings.filterwarnings("ignore", category=RuntimeWarning)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------------------------------------------
# Helpers (repris de main_unsupervised.py)
# ----------------------------------------------------------------------
def make_nan_mask(X):
    return (~np.isnan(X)).astype(float)


def preprocess_with_nan(X, center=True, normalize=True):
    W = make_nan_mask(X)
    mean = np.nanmean(X, axis=0, keepdims=True)
    mean = np.nan_to_num(mean, nan=0.0)
    if center:
        Xc = np.where(np.isnan(X), 0.0, X - mean)
    else:
        Xc = np.where(np.isnan(X), 0.0, X)
    if normalize:
        nrm = float(np.sqrt(np.sum((Xc * W) ** 2)))
        if nrm > 0:
            Xc = Xc / nrm
    else:
        nrm = 1.0
    return Xc, W, nrm


def preprocess_plain(X, center=True, normalize=True):
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    if normalize:
        nrm = float(np.linalg.norm(X))
        if nrm > 0:
            X = X / nrm
    return X, None, (nrm if normalize else 1.0)


def match_components(A, Y_true, analyte_names):
    """Pour chaque composante, analyte le mieux corrélé (avec signe)."""
    R = A.shape[1]
    mapping = {}
    for r in range(R):
        best_rho, best_c = 0.0, -1
        for c in range(Y_true.shape[1]):
            rho = pearsonr(A[:, r], Y_true[:, c])[0]
            if abs(rho) > abs(best_rho):
                best_rho, best_c = rho, c
        mapping[r] = (analyte_names[best_c] if best_c >= 0 else "?",
                      float(best_rho), best_c)
    return mapping


def align_signs(A, mapping, other_factors):
    """Retourne les composantes en signe positif vs concentrations."""
    A2 = A.copy()
    for r, (_, rho, _) in mapping.items():
        if rho < 0:
            A2[:, r] *= -1
            for F in other_factors:
                if F is not None:
                    F[:, r] *= -1
    return A2


def label_component(r, mapping, lam_nmr, lam_eem, sig_lcms, threshold=0.6):
    """Étiquette lisible pour une composante."""
    name, rho, _ = mapping[r]
    if abs(rho) > threshold:
        return f"c{r+1}: {name}"
    parts = []
    if lam_nmr[r] > 0.1:
        parts.append("NMR")
    if lam_eem[r] > 0.1:
        parts.append("EEM")
    if sig_lcms[r] > 0.1:
        parts.append("LCMS")
    if len(parts) == 1:
        return f"c{r+1}: {parts[0]}-only"
    if not parts:
        return f"c{r+1}: noise"
    return f"c{r+1}: noise ({'+'.join(parts)})"


# ----------------------------------------------------------------------
def main():
    data = load_joda("data")

    if data.NMR is not None and data.NMR.shape[1] > 2000:
        data.NMR = data.NMR[:, ::10, :]
        print(f"[downsample] NMR -> {data.NMR.shape}")

    # ------------------------------------------------------------------
    # Prétraitement
    # ------------------------------------------------------------------
    print("[preprocess] ...")
    NMR_p,  W_nmr,  _ = preprocess_plain(data.NMR)
    EEM_p,  W_eem,  _ = preprocess_with_nan(data.EEM)  # EEM contient des NaN
    LCMS_p, W_lcms, _ = preprocess_plain(data.LCMS)

    ds = CoupledDataset(
        tensors=[NMR_p, EEM_p],
        matrices=[LCMS_p],
        W_tensors=[None, W_eem],
        W_matrices=None,
    )

    # ------------------------------------------------------------------
    # Fit CMTF et ACMTF
    # ------------------------------------------------------------------
    R = 6
    print(f"[CMTF-OPT] R={R}, 3 random starts...")
    cmtf_res = cmtf_opt(ds, R=R, n_inits=3, max_iter=400)
    print(f"  fval CMTF  = {cmtf_res['fval']:.4f}")

    print(f"[ACMTF] R={R}, β=1e-2, 5 random starts...")
    acmtf_res = acmtf_opt(ds, R=R, beta=1e-2, alpha=1.0,
                          n_inits=5, max_iter=1000, random_state=42)
    print(f"  fval ACMTF = {acmtf_res['fval']:.4f}")

    lam_nmr = acmtf_res["lambdas"][0]
    lam_eem = acmtf_res["lambdas"][1]
    sig_lcms = acmtf_res["sigmas"][0]
    A = acmtf_res["A"]
    B_nmr = acmtf_res["tensor_factors"][0][1]   # chemical shift
    C_nmr = acmtf_res["tensor_factors"][0][2]   # gradient
    B_eem = acmtf_res["tensor_factors"][1][1]   # emission
    C_eem = acmtf_res["tensor_factors"][1][2]   # excitation
    V_lcms = acmtf_res["matrix_factors"][0]

    # Tri des composantes par poids total décroissant
    total = lam_nmr + lam_eem + sig_lcms
    order = np.argsort(-total)
    lam_nmr = lam_nmr[order]; lam_eem = lam_eem[order]; sig_lcms = sig_lcms[order]
    A = A[:, order]
    B_nmr = B_nmr[:, order]; C_nmr = C_nmr[:, order]
    B_eem = B_eem[:, order]; C_eem = C_eem[:, order]
    V_lcms = V_lcms[:, order]

    # Matching composantes -> analytes
    mapping = match_components(A, data.Y, data.analyte_names)
    A = align_signs(A, mapping, [B_nmr, C_nmr, B_eem, C_eem, V_lcms])

    comp_labels = [label_component(r, mapping, lam_nmr, lam_eem, sig_lcms)
                   for r in range(R)]

    print("\n[match] Composantes -> analytes (après tri et alignement):")
    for r in range(R):
        name, rho, _ = mapping[r]
        print(f"  c{r+1}: {comp_labels[r]:<20}  ρ={rho:+.3f}  "
              f"λ_NMR={lam_nmr[r]:.3f}  λ_EEM={lam_eem[r]:.3f}  "
              f"σ_LC={sig_lcms[r]:.3f}")

    colors = plt.cm.tab10(np.linspace(0, 1, R))

    # ==================================================================
    # Figure 1 — poids des composantes par bloc
    # ==================================================================
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    x = np.arange(R)
    for ax, weights, title, color in zip(
        axes,
        [lam_nmr, lam_eem, sig_lcms],
        [r"$\lambda$ (NMR)", r"$\lambda$ (EEM)", r"$\sigma$ (LC-MS)"],
        ["#2b5d9a", "#27ae60", "#c0392b"],
    ):
        ax.bar(x, weights, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(comp_labels, rotation=35, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("poids")
    plt.suptitle("ACMTF tri-blocs — poids des composantes par bloc",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_weights.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 2 — scatter scores vs concentrations vraies
    # ==================================================================
    n_ana = data.Y.shape[1]
    fig, axes = plt.subplots(1, n_ana, figsize=(3.2 * n_ana, 3.2))
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
        a = A[:, best_r]; y = data.Y[:, c]
        a_n = (a - a.min()) / (a.max() - a.min() + 1e-12)
        y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
        ax.scatter(y_n, a_n, s=35, c=colors[best_r], edgecolor="k", linewidth=0.4)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
        ax.set_xlabel("concentration vraie (norm.)")
        if c == 0:
            ax.set_ylabel("score ACMTF (norm.)")
        ax.set_title(f"{data.analyte_names[c]}\nc{best_r+1}, ρ={best_rho:+.3f}",
                     fontsize=10)
        ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    plt.suptitle("Scores ACMTF vs concentrations vraies (tri-blocs)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_scores_vs_conc.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 3 — A vs design expérimental
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
        ax.plot(x_samples, y_n, "o-", color="#2b5d9a", label="vraie", ms=4, lw=1)
        ax.plot(x_samples, a_n, "o-", color="#c0392b", label="ACMTF", ms=4, lw=1)
        ax.set_title(f"{data.analyte_names[c]}\nρ={best_rho:+.3f}", fontsize=10)
        ax.set_xlabel("échantillon")
        if c == 0:
            ax.set_ylabel("intensité (norm.)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Scores ACMTF vs design expérimental",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_A_vs_design.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 4 — signatures NMR (chemical shift + gradient)
    # ==================================================================
    nmr_ppm = None
    if data.nmr_axes and data.nmr_axes.get("chemical_shift") is not None:
        full_ppm = data.nmr_axes["chemical_shift"]
        if len(full_ppm) >= B_nmr.shape[0] * 10:
            nmr_ppm = np.asarray(full_ppm[::10][:B_nmr.shape[0]], dtype=float)
    if nmr_ppm is None:
        nmr_ppm = np.arange(B_nmr.shape[0])

    fig, axes = plt.subplots(R, 2, figsize=(11, 1.6 * R),
                             gridspec_kw={"width_ratios": [4, 1]})
    if R == 1:
        axes = axes[None, :]
    for r in range(R):
        ax_spec, ax_dec = axes[r, 0], axes[r, 1]
        ax_spec.plot(nmr_ppm, B_nmr[:, r], color=colors[r], lw=0.8)
        ax_spec.set_ylabel(f"b{r+1}", rotation=0, labelpad=20, fontsize=9)
        ax_spec.set_title(comp_labels[r], fontsize=9, loc="left")
        ax_spec.grid(alpha=0.3)
        if nmr_ppm[0] < nmr_ppm[-1] and nmr_ppm.max() < 20:
            ax_spec.invert_xaxis()

        ax_dec.plot(np.arange(1, C_nmr.shape[0] + 1), C_nmr[:, r],
                    "o-", color=colors[r], ms=4)
        ax_dec.grid(alpha=0.3)
        ax_dec.set_yticklabels([])
        if r == 0:
            ax_dec.set_title("gradient", fontsize=9)
    axes[-1, 0].set_xlabel("chemical shift (ppm)")
    axes[-1, 1].set_xlabel("gradient level")
    plt.suptitle("Signatures NMR (ACMTF tri-blocs)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_nmr_signatures.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 5 — loadings EEM (émission + excitation)
    # ==================================================================
    fig, axes = plt.subplots(R, 2, figsize=(10, 1.6 * R), sharex="col")
    if R == 1:
        axes = axes[None, :]
    for r in range(R):
        axes[r, 0].plot(np.arange(B_eem.shape[0]), B_eem[:, r],
                        color=colors[r], lw=1.0)
        axes[r, 0].set_ylabel(f"c{r+1}", rotation=0, labelpad=15, fontsize=9)
        axes[r, 0].grid(alpha=0.3)
        axes[r, 0].set_title(comp_labels[r], fontsize=9, loc="left")

        axes[r, 1].plot(np.arange(C_eem.shape[0]), C_eem[:, r],
                        color=colors[r], lw=1.0)
        axes[r, 1].grid(alpha=0.3)
    axes[0, 0].set_title(axes[0, 0].get_title() + "  — émission", fontsize=9, loc="left")
    axes[0, 1].set_title("excitation", fontsize=9)
    axes[-1, 0].set_xlabel("index émission")
    axes[-1, 1].set_xlabel("index excitation")
    plt.suptitle("Signatures EEM (ACMTF tri-blocs)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_eem_loadings.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 6 — loadings LC-MS
    # ==================================================================
    fig, axes = plt.subplots(R, 1, figsize=(10, 1.4 * R), sharex=True)
    if R == 1:
        axes = [axes]
    for r, ax in enumerate(axes):
        ax.plot(np.arange(V_lcms.shape[0]), V_lcms[:, r],
                color=colors[r], lw=0.8)
        ax.set_ylabel(f"v{r+1}", rotation=0, labelpad=20, fontsize=9)
        ax.set_title(comp_labels[r], fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("index feature LC-MS")
    plt.suptitle("Loadings LC-MS (ACMTF tri-blocs)", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_lcms_loadings.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 7 — heatmap identité "analytique" des composantes
    # ==================================================================
    W = np.vstack([lam_nmr, lam_eem, sig_lcms])  # (3, R)
    # normalisation par colonne (chaque composante somme à 1) pour lire
    # la "répartition" entre blocs
    W_norm = W / (W.sum(axis=0, keepdims=True) + 1e-12)

    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    # (a) heatmap brute
    im0 = axes[0].imshow(W, aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(3))
    axes[0].set_yticklabels(["NMR (λ)", "EEM (λ)", "LC-MS (σ)"])
    axes[0].set_xticks(range(R))
    axes[0].set_xticklabels(comp_labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_title("Poids bruts")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    # annotations
    for i in range(3):
        for j in range(R):
            axes[0].text(j, i, f"{W[i, j]:.2f}",
                         ha="center", va="center",
                         color="white" if W[i, j] < W.max() * 0.5 else "black",
                         fontsize=8)

    # (b) heatmap normalisée par composante
    im1 = axes[1].imshow(W_norm, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    axes[1].set_yticks(range(3))
    axes[1].set_yticklabels(["NMR (λ)", "EEM (λ)", "LC-MS (σ)"])
    axes[1].set_xticks(range(R))
    axes[1].set_xticklabels(comp_labels, rotation=35, ha="right", fontsize=8)
    axes[1].set_title("Répartition par composante (somme = 1)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    for i in range(3):
        for j in range(R):
            axes[1].text(j, i, f"{W_norm[i, j]:.2f}",
                         ha="center", va="center",
                         color="black", fontsize=8)

    plt.suptitle("Identité analytique des composantes ACMTF",
                 y=1.02, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_heatmap_weights.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 8 — comparaison CMTF vs ACMTF
    # ==================================================================
    fig, ax = plt.subplots(figsize=(5, 4))
    fvals = [cmtf_res["fval"], acmtf_res["fval"]]
    ax.bar(["CMTF-OPT", "ACMTF"], fvals,
           color=["#95a5a6", "#2980b9"])
    for i, v in enumerate(fvals):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("valeur objectif finale")
    ax.set_title("CMTF vs ACMTF — qualité de l'ajustement")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "unsup_cmtf_vs_acmtf.png"), dpi=130)
    plt.close(fig)

    print(f"\n[done] figures sauvegardées dans {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith("unsup_"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
