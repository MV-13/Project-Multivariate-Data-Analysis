"""
plot_results_supervised.py
==========================

Figures pour l'analyse supervisée (prédiction des concentrations en LOO-CV).

Ce script charge les prédictions sauvegardées par main_supervised.py dans
results/supervised_loo_results.npz et génère :

    1. sup_r2_global.png      — R² moyen par méthode
    2. sup_rmse_global.png    — RMSE moyen par méthode
    3. sup_r2_per_analyte.png — R² par méthode × analyte
    4. sup_scatter_best.png   — scatter Y_true vs Y_pred pour les meilleures
                                 méthodes, un subplot par analyte
    5. sup_method_ranking.png — ranking des méthodes par analyte
    6. sup_pred_vs_true_all.png — courbes Y_true et Y_pred par échantillon

Usage :
    python plot_results_supervised.py
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
NPZ_PATH = os.path.join(RESULTS_DIR, "supervised_loo_results.npz")


# ----------------------------------------------------------------------
def r2_per_col(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def rmse_per_col(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))


def pretty_method_name(key: str) -> str:
    mapping = {
        "ridge_NMR":  "Ridge (NMR)",
        "ridge_EEM":  "Ridge (EEM)",
        "ridge_LCMS": "Ridge (LC-MS)",
        "pls_NMR":    "PLS (NMR)",
        "pls_EEM":    "PLS (EEM)",
        "pls_LCMS":   "PLS (LC-MS)",
        "cp_NMR":     "CP reg. (NMR)",
        "cp_EEM":     "CP reg. (EEM)",
        "cmtf":       "CMTF + lin.",
        "acmtf":      "ACMTF + lin.",
    }
    return mapping.get(key, key)


def method_color(key: str) -> str:
    if key.startswith("ridge"):
        return "#95a5a6"
    if key.startswith("pls"):
        return "#3498db"
    if key.startswith("cp"):
        return "#27ae60"
    if key == "cmtf":
        return "#e67e22"
    if key == "acmtf":
        return "#c0392b"
    return "#7f8c8d"


# ----------------------------------------------------------------------
def main():
    if not os.path.isfile(NPZ_PATH):
        raise FileNotFoundError(
            f"{NPZ_PATH} introuvable. Lance d'abord main_supervised.py."
        )

    data = np.load(NPZ_PATH, allow_pickle=True)
    Y_true = data["Y_true"]
    analytes = list(data["analytes"])
    n_ana = Y_true.shape[1]

    methods = {}
    for k in data.files:
        if k.startswith("ypred_"):
            methods[k[len("ypred_"):]] = data[k]

    methods = {k: v for k, v in methods.items() if v.shape == Y_true.shape}

    # Ordre d'affichage : ridge, pls, cp, cmtf, acmtf
    def sort_key(k):
        order = {"ridge": 0, "pls": 1, "cp": 2, "cmtf": 3, "acmtf": 4}
        for prefix, val in order.items():
            if k.startswith(prefix):
                return (val, k)
        return (99, k)
    method_keys = sorted(methods.keys(), key=sort_key)

    r2_all = {}
    rmse_all = {}
    for k in method_keys:
        r2_all[k] = r2_per_col(Y_true, methods[k])
        rmse_all[k] = rmse_per_col(Y_true, methods[k])

    print("[data] méthodes disponibles :", method_keys)
    print(f"[data] analytes : {analytes}")

    # ==================================================================
    # Figure 1 — R² moyen global par méthode
    # ==================================================================
    r2_means = [float(np.mean(r2_all[k])) for k in method_keys]
    colors = [method_color(k) for k in method_keys]
    labels = [pretty_method_name(k) for k in method_keys]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, r2_means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, r2_means):
        y_txt = v + 0.02 if v >= 0 else v - 0.05
        ax.text(bar.get_x() + bar.get_width() / 2, y_txt,
                f"{v:.3f}", ha="center",
                fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("R² moyen (LOO-CV)")
    ax.set_title("Performance prédictive par méthode — R² moyen sur les 5 analytes")
    ax.set_ylim(min(0, min(r2_means)) - 0.1, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sup_r2_global.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 2 — RMSE moyen par méthode
    # ==================================================================
    rmse_means = [float(np.mean(rmse_all[k])) for k in method_keys]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(labels, rmse_means, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, rmse_means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(rmse_means) * 0.02,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_ylabel("RMSE moyen (LOO-CV)")
    ax.set_title("RMSE moyen par méthode sur les 5 analytes")
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sup_rmse_global.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 3 — heatmap R² méthode × analyte
    # ==================================================================
    R2_matrix = np.array([r2_all[k] for k in method_keys])  # (n_methods, n_ana)
    R2_display = np.clip(R2_matrix, -0.5, 1.0)  # clip pour la lisibilité

    fig, ax = plt.subplots(figsize=(1.4 * n_ana + 2, 0.5 * len(method_keys) + 2))
    im = ax.imshow(R2_display, aspect="auto", cmap="RdYlGn",
                   vmin=-0.5, vmax=1.0)
    ax.set_yticks(range(len(method_keys)))
    ax.set_yticklabels(labels)
    ax.set_xticks(range(n_ana))
    ax.set_xticklabels(analytes, rotation=25, ha="right")
    ax.set_title("R² par méthode et par analyte (LOO-CV)")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("R²")

    for i in range(len(method_keys)):
        for j in range(n_ana):
            v = R2_matrix[i, j]
            color = "white" if v < 0.3 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sup_r2_per_analyte.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 4 — scatter Y_true vs Y_pred pour les meilleures méthodes
    # ==================================================================
    # Pour chaque analyte, on prend la meilleure méthode
    best_per_analyte = []
    for c in range(n_ana):
        best_k = max(method_keys, key=lambda k: r2_all[k][c])
        best_per_analyte.append(best_k)

    fig, axes = plt.subplots(1, n_ana, figsize=(3.2 * n_ana, 3.2))
    if n_ana == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        k = best_per_analyte[c]
        yt = Y_true[:, c]
        yp = methods[k][:, c]
        lo = float(min(yt.min(), yp.min()))
        hi = float(max(yt.max(), yp.max()))
        pad = (hi - lo) * 0.05
        ax.scatter(yt, yp, s=40, c=method_color(k),
                   edgecolor="black", linewidth=0.4, alpha=0.8)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("concentration vraie")
        if c == 0:
            ax.set_ylabel("concentration prédite")
        ax.set_title(f"{analytes[c]}\n{pretty_method_name(k)} "
                     f"(R²={r2_all[k][c]:.3f})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")
    plt.suptitle("Meilleure méthode par analyte — prédictions LOO-CV",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sup_scatter_best.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 5 — ranking des méthodes par analyte
    # ==================================================================
    fig, ax = plt.subplots(figsize=(12, 5))
    n_m = len(method_keys)
    w = 0.8 / n_m
    x = np.arange(n_ana)
    for m_idx, k in enumerate(method_keys):
        offsets = x + (m_idx - n_m / 2 + 0.5) * w
        ax.bar(offsets, r2_all[k], width=w,
               label=pretty_method_name(k),
               color=method_color(k), edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(analytes, rotation=15, ha="right")
    ax.set_ylabel("R² (LOO-CV)")
    ax.set_title("Comparaison des méthodes par analyte")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylim(min(-0.2, R2_matrix.min() - 0.05), 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sup_method_ranking.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Figure 6 — courbes Y_true vs Y_pred par échantillon
    # ==================================================================
    n_samples = Y_true.shape[0]
    x_s = np.arange(1, n_samples + 1)
    fig, axes = plt.subplots(1, n_ana, figsize=(3.4 * n_ana, 3.2))
    if n_ana == 1:
        axes = [axes]
    for c, ax in enumerate(axes):
        k = best_per_analyte[c]
        yt = Y_true[:, c]; yp = methods[k][:, c]
        ax.plot(x_s, yt, "o-", color="#2b5d9a", lw=1, ms=4, label="vraie")
        ax.plot(x_s, yp, "o-", color=method_color(k), lw=1, ms=4,
                label=pretty_method_name(k))
        ax.set_xlabel("échantillon")
        if c == 0:
            ax.set_ylabel("concentration")
        ax.set_title(f"{analytes[c]}\nR²={r2_all[k][c]:.3f}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    plt.suptitle("Prédictions LOO-CV vs concentrations vraies (meilleure méthode / analyte)",
                 y=1.03, fontsize=12)
    plt.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "sup_pred_vs_true_all.png"),
                dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ==================================================================
    # Résumé texte
    # ==================================================================
    print("\n=== Résumé des performances ===")
    print(f"{'méthode':<22}  R² moy.  RMSE moy.")
    for k in method_keys:
        r2m = float(np.mean(r2_all[k]))
        rm  = float(np.mean(rmse_all[k]))
        print(f"{pretty_method_name(k):<22}  {r2m:>6.3f}   {rm:>6.3f}")

    print("\n=== Meilleure méthode par analyte ===")
    for c in range(n_ana):
        k = best_per_analyte[c]
        print(f"  {analytes[c]:<18} -> {pretty_method_name(k):<22} "
              f"R²={r2_all[k][c]:.3f}")

    print(f"\n[done] 6 figures sauvegardées dans {RESULTS_DIR}/")
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.startswith("sup_"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
