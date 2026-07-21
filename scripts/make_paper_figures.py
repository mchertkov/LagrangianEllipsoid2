from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
import numpy as np
from scipy.stats import gaussian_kde

# Publication typography.  The previous release used matplotlib defaults, which
# became too small after four-panel figures were scaled to journal width.
plt.rcParams.update({
    "font.size": 12.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 13.0,
    "xtick.labelsize": 11.5,
    "ytick.labelsize": 11.5,
    "legend.fontsize": 10.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 6.5,
})

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lagrangian_ellipsoid.core import load_npz, derive, finite_lag_components  # noqa: E402

DATA = ROOT / "data"
RAW = DATA / "raw"
RESULTS = DATA / "results"
FIG_MAIN = ROOT / "figures" / "main"
FIG_SUPP = ROOT / "figures" / "controls"
TABLES = ROOT / "tables"
FIG_MAIN.mkdir(parents=True, exist_ok=True)
FIG_SUPP.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def sem(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def ellipse_patch(g, c, **kwargs):
    vals, vecs = np.linalg.eigh(0.5 * (g + g.T))
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    return Ellipse(c, width=2 * np.sqrt(vals[0]), height=2 * np.sqrt(vals[1]), angle=angle,
                   fill=False, **kwargs)


def load_runs(pattern):
    return [load_npz(p) for p in sorted(ROOT.glob(pattern))]


def binned_curves(runs, edges):
    centers = np.sqrt(edges[:-1] * edges[1:])
    keys = ["mee", "mass", "delta_sigma", "delta_rho"]
    arr = {k: np.full((len(runs), len(centers)), np.nan) for k in keys}
    samples = {"mee": [[] for _ in centers], "mass": [[] for _ in centers]}
    for i, d in enumerate(runs):
        me = derive(d, "mee")
        ma = derive(d, "cov")
        rr = me["r"] / float(d["r0"])
        ib = np.digitize(rr, edges) - 1
        for b in range(len(centers)):
            q = ib == b
            if q.sum() >= 2:
                arr["mee"][i, b] = np.mean(me["sigma"][q])
                arr["mass"][i, b] = np.mean(ma["sigma"][q])
                arr["delta_sigma"][i, b] = np.mean(ma["sigma"][q] - me["sigma"][q])
                arr["delta_rho"][i, b] = np.mean(np.log(me["r"][q] / ma["r"][q]))
                samples["mee"][b].extend(me["sigma"][q].tolist())
                samples["mass"][b].extend(ma["sigma"][q].tolist())
    out = {"centers": centers, "edges": edges, "arr": arr, "samples": samples}
    for k, a in arr.items():
        out[k + "_mean"] = np.nanmean(a, axis=0)
        out[k + "_sem"] = np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(a), axis=0))
        out[k + "_n"] = np.sum(np.isfinite(a), axis=0)
    return out


def load_json(name, *, fallback=None):
    """Load a frozen JSON result, optionally using an explicit fallback.

    The paper-figure path does not refit the reduced model.  Its model scores are
    already embedded in ``paper_numbers.json``.  The fallback keeps figure
    reproduction functional if the small standalone model-summary JSON was
    omitted during a partial copy or cloud-sync operation.
    """
    path = RESULTS / name
    if path.exists():
        return json.loads(path.read_text())
    if fallback is not None:
        print(f"Warning: {path.relative_to(ROOT)} is missing; using the frozen copy in paper_numbers.json.")
        return fallback
    raise FileNotFoundError(
        f"Required release file is missing: {path}. "
        "Re-extract the release into a clean directory or restore the file from the ZIP archive."
    )


analysis = load_json("phase3B_analysis_summary.json")
existing_numbers_path = RESULTS / "paper_numbers.json"
existing_numbers = json.loads(existing_numbers_path.read_text()) if existing_numbers_path.exists() else {}
model_summary = load_json(
    "intrinsic_model_b_summary.json",
    fallback=existing_numbers.get("model"),
)

b1 = load_runs("data/raw/decisive_k192/*.npz")
rough = {
    0.25: load_runs("data/raw/roughness_k128/rough025*.npz"),
    1 / 3: load_runs("data/raw/roughness_k128/rough13_k128*.npz"),
    0.5: load_runs("data/raw/roughness_k128/rough050*.npz"),
    2 / 3: load_runs("data/raw/roughness_k128/rough067*.npz"),
}

# -----------------------------------------------------------------------------
# Figure 1: methodological objects
# -----------------------------------------------------------------------------
d = b1[0]
js = 3  # snapshot at r/r0 approximately 8
pos = d["snapshot_positions"][js]
target_t = float(d["snapshot_time"][js])
it = int(np.argmin(np.abs(d["time"] - target_t)))
gm = d["mee_g"][-1, it]
gc = d["cov_g"][-1, it]
cm = d["mee_center"][-1, it]
cc = d["cov_center"][-1, it]
sids = [int(x) for x in d["support_ids"][it] if x >= 0]

fig = plt.figure(figsize=(12.4, 5.25))
ax = fig.add_axes([0.055, 0.12, 0.47, 0.80])
ax.scatter(pos[:, 0], pos[:, 1], s=4, alpha=0.42, rasterized=True, label="tracers")
ax.add_patch(ellipse_patch(gc, cc, linewidth=2.0, linestyle="--", label="mass ellipse"))
ax.add_patch(ellipse_patch(gm, cm, linewidth=2.2, label="outer MEE"))
if sids:
    ax.scatter(pos[sids, 0], pos[sids, 1], s=55, marker="o", facecolors="none", linewidths=1.8,
               label="MEE support tracers")
ax.scatter([cc[0]], [cc[1]], marker="x", s=55, linewidths=1.8, label="centroid")
ax.set_aspect("equal", adjustable="datalim")
ax.set_title(r"(a) One cloud at $r_{\rm MEE}/r_0\simeq 8$")
ax.set_xlabel("relative $x_1$")
ax.set_ylabel("relative $x_2$")
ax.legend(loc="best", fontsize=10.5)

ax2 = fig.add_axes([0.56, 0.08, 0.41, 0.84])
ax2.axis("off")
boxes = [
    (0.01, 0.68, 0.29, 0.18, "particle train\n" + r"$\{x_i(t),u_i(t)\}$"),
    (0.355, 0.68, 0.29, 0.18, "two geometries\n" + r"$G_{\rm mass},\ g_{\rm MEE}$"),
    (0.70, 0.68, 0.29, 0.18, "two gradients\n" + r"$M_E,\ M_{\rm LS}$"),
    (0.16, 0.30, 0.32, 0.18, "intrinsic train\n" + r"$(v,\sigma,q,p,\omega)$"),
    (0.56, 0.30, 0.32, 0.18, "held-out\ngenerator tests"),
]
for x, y, w, h, txt in boxes:
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", linewidth=1.4,
                           facecolor="white")
    ax2.add_patch(patch)
    ax2.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=12)
for p0, p1 in [((0.30, 0.77), (0.355, 0.77)), ((0.645, 0.77), (0.70, 0.77)),
               ((0.78, 0.68), (0.66, 0.48)), ((0.50, 0.68), (0.35, 0.48)),
               ((0.47, 0.39), (0.56, 0.39))]:
    ax2.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=12, linewidth=1.2))
ax2.text(0.50, 0.93, "(b) Transferable diagnostic-and-modeling workflow", ha="center",
         va="center", fontsize=13)
ax2.text(0.50, 0.08,
         r"$\dot\sigma_{\rm MEE}=Q(g,M_E)+R_{\rm gradient}+R_{\rm envelope}$",
         ha="center", va="center", fontsize=13)
fig.savefig(FIG_MAIN / "fig01_methodology.pdf", bbox_inches="tight")
fig.savefig(FIG_MAIN / "fig01_methodology.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 2: central saturation result
# -----------------------------------------------------------------------------
edges = np.geomspace(1, 20, 13)
curves = binned_curves(b1, edges)
fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.6))
ax = axs[0, 0]
ok = curves["mee_n"] >= 8
ax.errorbar(curves["centers"][ok], curves["mee_mean"][ok], yerr=curves["mee_sem"][ok],
            marker="o", label="outer MEE")
ok2 = curves["mass_n"] >= 8
ax.errorbar(curves["centers"][ok2], curves["mass_mean"][ok2], yerr=curves["mass_sem"][ok2],
            marker="s", label="mass ellipse")
ax.set_xscale("log")
ax.set_xlabel(r"$r_{\rm MEE}/r_0$")
ax.set_ylabel(r"$\langle\sigma\mid r\rangle$")
ax.set_title("(a) Scale-conditioned aspect ratio")
ax.legend(fontsize=10.5)

ax = axs[0, 1]
sel_bins = [4, 6, 8]
for b in sel_bins:
    vals = np.asarray(curves["samples"]["mee"][b], float)
    vals = vals[np.isfinite(vals)]
    if len(vals) > 10:
        xx = np.linspace(0, max(2.2, np.percentile(vals, 99)), 220)
        ax.plot(xx, gaussian_kde(vals)(xx), label=fr"$r/r_0\simeq{curves['centers'][b]:.1f}$")
ax.set_xlabel(r"$\sigma_{\rm MEE}$")
ax.set_ylabel("density")
ax.set_title("(b) Broad conditional laws")
ax.legend(fontsize=10.5)

ax = axs[1, 0]
intervals = [(3, 8), (4, 12), (6, 16)]
x = np.arange(len(intervals))
mee_m = [analysis["decisive_k192"][f"slopes_{a}_{b}"]["beta_mee_mean"] for a, b in intervals]
mee_e = [analysis["decisive_k192"][f"slopes_{a}_{b}"]["beta_mee_sem"] for a, b in intervals]
mas_m = [analysis["decisive_k192"][f"slopes_{a}_{b}"]["beta_mass_mean"] for a, b in intervals]
mas_e = [analysis["decisive_k192"][f"slopes_{a}_{b}"]["beta_mass_sem"] for a, b in intervals]
ax.errorbar(x - 0.08, mee_m, yerr=mee_e, fmt="o", label="outer MEE")
ax.errorbar(x + 0.08, mas_m, yerr=mas_e, fmt="s", label="mass ellipse")
ax.axhline(0, linewidth=0.8)
ax.set_xticks(x, [f"{a}-{b}" for a, b in intervals])
ax.set_xlabel(r"fit interval in $r/r_0$")
ax.set_ylabel(r"$d\langle\sigma\rangle/d\log r$")
ax.set_title("(c) No resolved post-transient growth")
ax.legend(fontsize=10.5)

ax = axs[1, 1]
pp = analysis["decisive_k192"]["paired_plateau"]
mee_diff = np.array([v["mee_b"] - v["mee_a"] for v in pp["per_seed"]], float)
mas_diff = np.array([v["mass_b"] - v["mass_a"] for v in pp["per_seed"]], float)
ax.boxplot([mee_diff[np.isfinite(mee_diff)], mas_diff[np.isfinite(mas_diff)]], tick_labels=["outer MEE", "mass ellipse"])
ax.axhline(0, linewidth=0.8)
ax.set_ylabel(r"$\langle\sigma\rangle_{8-12}-\langle\sigma\rangle_{4-8}$")
ax.set_title("(d) Paired realization test")
fig.tight_layout()
fig.savefig(FIG_MAIN / "fig02_saturation.pdf")
fig.savefig(FIG_MAIN / "fig02_saturation.png", dpi=220)
plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 3: bulk versus outer envelope and roughness
# -----------------------------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.9))
for ax, js, letter in [(axs[0, 0], 2, "a"), (axs[0, 1], 4, "b")]:
    pos = d["snapshot_positions"][js]
    target_t = float(d["snapshot_time"][js])
    it = int(np.argmin(np.abs(d["time"] - target_t)))
    gm = d["mee_g"][-1, it]; gc = d["cov_g"][-1, it]
    cm = d["mee_center"][-1, it]; cc = d["cov_center"][-1, it]
    sids = [int(x) for x in d["support_ids"][it] if x >= 0]
    ax.scatter(pos[:, 0], pos[:, 1], s=3.5, alpha=0.38, rasterized=True)
    ax.add_patch(ellipse_patch(gc, cc, linewidth=1.8, linestyle="--"))
    ax.add_patch(ellipse_patch(gm, cm, linewidth=2.0))
    if sids:
        ax.scatter(pos[sids, 0], pos[sids, 1], s=45, facecolors="none", linewidths=1.5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(fr"({letter}) $r_{{\rm MEE}}/r_0\simeq{d['snapshot_ratio'][js]:.0f}$")
    ax.set_xlabel("relative $x_1$")
    ax.set_ylabel("relative $x_2$")

ax = axs[1, 0]
ok = curves["delta_sigma_n"] >= 8
h1 = ax.errorbar(curves["centers"][ok], curves["delta_sigma_mean"][ok],
            yerr=curves["delta_sigma_sem"][ok], marker="o", label=r"$\Delta_\sigma$")
ax2 = ax.twinx()
h2 = ax2.errorbar(curves["centers"][ok], curves["delta_rho_mean"][ok],
             yerr=curves["delta_rho_sem"][ok], marker="s", label=r"$\Delta_\rho$")
ax.set_xscale("log")
ax.set_xlabel(r"$r_{\rm MEE}/r_0$")
ax.set_ylabel(r"$\Delta_\sigma=\sigma_{\rm mass}-\sigma_{\rm MEE}$")
ax2.set_ylabel(r"$\Delta_\rho=\log(r_{\rm MEE}/r_{\rm mass})$")
ax.set_title("(c) Complementary geometric gaps")
ax.legend([h1, h2], [r"$\Delta_\sigma$", r"$\Delta_\rho$"], fontsize=10.5, loc="best")

ax = axs[1, 1]
zs = sorted(rough)
drho = [analysis["roughness"][str(z)]["delta_rho_mean"] for z in zs]
drho_e = [analysis["roughness"][str(z)]["delta_rho_sem"] for z in zs]
dsig = [analysis["roughness"][str(z)]["delta_sigma_mean"] for z in zs]
dsig_e = [analysis["roughness"][str(z)]["delta_sigma_sem"] for z in zs]
ax.errorbar(zs, drho, yerr=drho_e, marker="o", label=r"$\Delta_\rho$")
ax.errorbar(zs, dsig, yerr=dsig_e, marker="s", label=r"$\Delta_\sigma$")
ax.set_xlabel(r"Hölder exponent $\zeta$")
ax.set_ylabel(r"mean gap on $2\leq r/r_0\leq6$")
ax.set_title("(d) Roughness comparison")
ax.legend(fontsize=10.5)
fig.tight_layout()
fig.savefig(FIG_MAIN / "fig03_bulk_envelope.pdf")
fig.savefig(FIG_MAIN / "fig03_bulk_envelope.png", dpi=220)
plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 4: three-term balance
# -----------------------------------------------------------------------------
edges2 = np.geomspace(2, 14, 8)
cent2 = np.sqrt(edges2[:-1] * edges2[1:])
names = ["q_E", "R_gradient", "R_envelope", "R_total", "R_mass_E", "R_mass_LS"]
per = {k: np.full((len(b1), len(cent2)), np.nan) for k in names}
central_rows = []
switch_rows = []
for i, run in enumerate(b1):
    f = finite_lag_components(run, lag=1)
    rr = f["r"] / float(run["r0"])
    ib = np.digitize(rr, edges2) - 1
    for b in range(len(cent2)):
        q = ib == b
        if q.sum() >= 2:
            for k in names:
                per[k][i, b] = np.mean(f[k][q])
    q = (rr >= 3) & (rr <= 12)
    if q.sum() >= 5:
        central_rows.append([np.mean(f[k][q]) for k in names])
    # switching proxy aligned with lag-1 residual values
    ids = run["support_ids"]
    turn = []
    for a, bset in zip(ids[:-1], ids[1:]):
        A = set(int(x) for x in a if x >= 0); B = set(int(x) for x in bset if x >= 0)
        turn.append(1 - len(A & B) / max(len(A | B), 1))
    turn = np.asarray(turn)
    env = f["R_envelope"]
    n = min(len(turn), len(env))
    if n:
        hi = turn[:n] >= np.nanmedian(turn[:n])
        switch_rows.append([np.nanmean(env[:n][hi]), np.nanmean(env[:n][~hi])])
central_rows = np.asarray(central_rows)
switch_rows = np.asarray(switch_rows)

fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.6))
ax = axs[0, 0]
for k, label in [("q_E", r"$Q(g,M_E)$"), ("R_gradient", r"$R_{\rm gradient}$"),
                 ("R_envelope", r"$R_{\rm envelope}$"), ("R_total", r"$R_{\rm total}$")]:
    m = np.nanmean(per[k], axis=0)
    e = np.nanstd(per[k], axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(per[k]), axis=0))
    ok = np.sum(np.isfinite(per[k]), axis=0) >= 8
    ax.errorbar(cent2[ok], m[ok], yerr=e[ok], marker="o", label=label)
ax.axhline(0, linewidth=0.8)
ax.set_xscale("log")
ax.set_xlabel(r"$r_{\rm MEE}/r_0$")
ax.set_ylabel("conditional rate")
ax.set_title("(a) Three-term MEE balance")
ax.legend(fontsize=10.5)

ax = axs[0, 1]
labels = [r"$Q_E$", r"$R_{\rm gradient}$", r"$R_{\rm envelope}$", r"$R_{\rm total}$"]
idx = [0, 1, 2, 3]
means = np.mean(central_rows[:, idx], axis=0)
errs = np.std(central_rows[:, idx], axis=0, ddof=1) / np.sqrt(len(central_rows))
ax.bar(np.arange(4), means, yerr=errs, capsize=3)
ax.axhline(0, linewidth=0.8)
ax.set_xticks(np.arange(4), labels)
ax.set_ylabel(r"mean rate on $3\leq r/r_0\leq12$")
ax.set_title("(b) Ensemble-mean budget")

ax = axs[1, 0]
labels = [r"$R_{\rm mass,E}$", r"$R_{\rm mass,LS}$"]
idx = [4, 5]
means = np.mean(central_rows[:, idx], axis=0)
errs = np.std(central_rows[:, idx], axis=0, ddof=1) / np.sqrt(len(central_rows))
ax.bar(np.arange(2), means, yerr=errs, capsize=3)
ax.axhline(0, linewidth=0.8)
ax.set_xticks(np.arange(2), labels)
ax.set_ylabel("mean rate")
ax.set_title("(c) Mass-ellipse control")

ax = axs[1, 1]
means = [analysis["support"]["Renv_switch_mean"], analysis["support"]["Renv_persistent_mean"]]
errs = [analysis["support"]["Renv_switch_sem"], analysis["support"]["Renv_persistent_sem"]]
ax.bar(np.arange(2), means, yerr=errs, capsize=3)
ax.axhline(0, linewidth=0.8)
ax.set_xticks(np.arange(2), ["support switching", "persistent support"])
ax.set_ylabel(r"$\langle R_{\rm envelope}\rangle$")
ax.set_title("(d) Support turnover is not decisive")
fig.tight_layout()
fig.savefig(FIG_MAIN / "fig04_balance.pdf")
fig.savefig(FIG_MAIN / "fig04_balance.png", dpi=220)
plt.close(fig)

# -----------------------------------------------------------------------------
# Figure 5: held-out reduced model; use saved Phase-3B panels
# -----------------------------------------------------------------------------
from PIL import Image
panel_files = [
    ROOT / "figures" / "source" / "B4_intrinsic_hierarchy_sigma.png",
    ROOT / "figures" / "source" / "B4_intrinsic_hierarchy_source.png",
    ROOT / "figures" / "source" / "B4_intrinsic_hierarchy_scores.png",
    ROOT / "figures" / "source" / "B4_intrinsic_hierarchy_rollout.png",
]
if not all(p.exists() for p in panel_files):
    missing = [str(p.relative_to(ROOT)) for p in panel_files if not p.exists()]
    raise FileNotFoundError(f"Missing frozen model panels: {missing}. Run scripts/fit_generator_hierarchy.py first.")
imgs = [Image.open(p).convert("RGB") for p in panel_files]
fig, axs = plt.subplots(2, 2, figsize=(11.8, 8.6))
for ax, im, lab in zip(axs.flat, imgs, ["(a)", "(b)", "(c)", "(d)"]):
    ax.imshow(im)
    ax.axis("off")
    ax.text(0.01, 0.98, lab, transform=ax.transAxes, va="top", ha="left", fontsize=15,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75))
fig.tight_layout(pad=0.3)
fig.savefig(FIG_MAIN / "fig05_model.pdf")
fig.savefig(FIG_MAIN / "fig05_model.png", dpi=220)
plt.close(fig)

# -----------------------------------------------------------------------------
# Numerical-control panels are generated separately by analyze_ensembles.py.
# -----------------------------------------------------------------------------
# N-convergence figure generated from summary
fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.6))
Ns = [100, 300, 1000]
for ax, key, ylabel, letter in [
    (axs[0], "mee_mean", r"$\langle\sigma_{\rm MEE}\rangle$", "a"),
    (axs[1], "delta_rho_mean", r"$\langle\log(r_{\rm MEE}/r_{\rm mass})\rangle$", "b")]:
    vals = [analysis["controls"]["particle_number"][str(n)][key] for n in Ns]
    errs = [analysis["controls"]["particle_number"][str(n)][key.replace("_mean", "_sem")] for n in Ns]
    ax.errorbar(Ns, vals, yerr=errs, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("number of tracers $N$")
    ax.set_ylabel(ylabel)
    ax.set_title(f"({letter})")
fig.tight_layout()
fig.savefig(FIG_SUPP / "figS_particle_number.pdf")
fig.savefig(FIG_SUPP / "figS_particle_number.png", dpi=220)
plt.close(fig)

# Write canonical numbers used by manuscript.
numbers = {
    "production": {
        "zeta": 1/3, "Kmax": 192, "N": 1000, "n_realizations": 24,
        "r0_over_uv": 1.0, "train": 16, "test": 8,
    },
    "saturation": {
        "slope_3_8_MEE": analysis["decisive_k192"]["slopes_3_8"]["beta_mee_mean"],
        "slope_3_8_MEE_sem": analysis["decisive_k192"]["slopes_3_8"]["beta_mee_sem"],
        "slope_3_8_mass": analysis["decisive_k192"]["slopes_3_8"]["beta_mass_mean"],
        "slope_3_8_mass_sem": analysis["decisive_k192"]["slopes_3_8"]["beta_mass_sem"],
        "mean_sigma_MEE": analysis["decisive_k192"]["slopes_3_8"]["mee_mean"],
        "mean_sigma_mass": analysis["decisive_k192"]["slopes_3_8"]["mass_mean"],
        "delta_sigma": analysis["decisive_k192"]["slopes_3_8"]["delta_sigma_mean"],
        "delta_rho": analysis["decisive_k192"]["slopes_3_8"]["delta_rho_mean"],
        "W1_MEE": analysis["decisive_k192"]["distribution_stability"]["mee_normalized_W1_mean"],
        "W1_mass": analysis["decisive_k192"]["distribution_stability"]["mass_normalized_W1_mean"],
    },
    "balance": analysis["residual_decomposition"]["1"],
    "support": analysis["support"],
    "model": model_summary,
    "controls": analysis["controls"],
    "roughness": analysis["roughness"],
}
(RESULTS / "paper_numbers.json").write_text(json.dumps(numbers, indent=2))

print("Generated final figures and paper_numbers.json")
