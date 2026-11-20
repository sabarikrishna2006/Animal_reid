"""EDA figures for reports/eda.md."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(r"e:\ML_projectr\reports\figures"); OUT.mkdir(parents=True, exist_ok=True)
RAW = Path(r"e:\ML_projectr\data\reid_student_data_main")
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "legend.frameon": False})

tr = pd.read_csv(RAW / "train.csv"); X = np.load(RAW / "miewid_features.npy").astype(np.float64)
U = X / np.linalg.norm(X, axis=1, keepdims=True); Utr = U[:len(tr)]
bins = np.linspace(-0.1, 1.0, 56)

# Fig 1: same vs different identity cosine
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
for ax, sp in zip(axs, ["hyena", "leopard"]):
    m = (tr.species == sp).values; Z = Utr[m]; y = tr.identity.values[m]
    S = Z @ Z.T; iu = np.triu_indices(len(y), 1); same = (y[:, None] == y[None, :])[iu]; s = S[iu]
    ax.hist(s[~same], bins, density=True, histtype="step", lw=2, color=C1, label="different animal")
    ax.hist(s[same], bins, density=True, histtype="step", lw=2, color=C2, label="same animal")
    ax.set_title(f"{sp}: cosine between two train images", color=INK, loc="left"); ax.set_xlabel("cosine similarity (L2-normalised MiewID)")
axs[0].set_ylabel("density"); axs[0].legend()
fig.tight_layout(); fig.savefig(OUT / "fig1_same_vs_diff_cosine.png", dpi=130)

# Fig 2: top-1 cosine to gallery — known (LOO), new-proxy, test
d = np.load(r"e:\ML_projectr\eda\top1_dists.npz")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
b2 = np.linspace(0.2, 1.0, 41)
for ax, sp in zip(axs, ["hyena", "leopard"]):
    ax.hist(d[f"{sp}_known"], b2, density=True, histtype="step", lw=2, color=C1, label="train, sibling kept (known)")
    ax.hist(d[f"{sp}_newproxy"], b2, density=True, histtype="step", lw=2, color=C2, label="train, own identity removed (simulated new)")
    ax.hist(d[f"{sp}_test"], b2, density=True, histtype="step", lw=2, color=C3, label="test images")
    ax.set_title(f"{sp}: best cosine to the training gallery", color=INK, loc="left"); ax.set_xlabel("top-1 cosine")
axs[0].set_ylabel("density"); axs[1].legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "fig2_top1_known_new_test.png", dpi=130)

# Fig 3: SIFT affine inliers for same vs different animal, shortlisted pairs
P = pd.read_csv(r"e:\ML_projectr\eda\sift_pilot.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
b3 = np.arange(0, 42, 2)
for ax, sp in zip(axs, ["hyena", "leopard"]):
    g = P[P.sp == sp]
    ax.hist(np.clip(g.inl_aff[~g.same], 0, 40), b3, density=True, histtype="step", lw=2, color=C1, label="different animal")
    ax.hist(np.clip(g.inl_aff[g.same], 0, 40), b3, density=True, histtype="step", lw=2, color=C2, label="same animal")
    ax.set_title(f"{sp}: RANSAC inliers on top-12 cosine neighbours", color=INK, loc="left")
    ax.set_xlabel("partial-affine inliers (clipped at 40)"); ax.set_yscale("log")
axs[0].set_ylabel("density (log)"); axs[0].legend()
fig.tight_layout(); fig.savefig(OUT / "fig3_sift_inliers.png", dpi=130)
print("saved", sorted(p.name for p in OUT.iterdir()))
