from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from lagrangian_ellipsoid.core import load_npz, derive  # noqa: E402

RAW = ROOT / "data" / "raw" / "decisive_k192"
RES = ROOT / "data" / "results"
TABLES = ROOT / "tables"
RES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

files = sorted(RAW.glob("*.npz"))
runs = [load_npz(p) for p in files]
train_runs = runs[:16]
test_runs = runs[16:24]


def state(d):
    o = derive(d, "mee")
    r0 = float(d["r0"])
    return {
        "v": np.log(o["r"] / r0),
        "sigma": o["sigma"],
        "z": np.column_stack([o["q"], o["p"], o["omega"]]),
        "time": o["time"],
    }


train = [state(d) for d in train_runs]
test = [state(d) for d in test_runs]
allv = np.concatenate([x["v"][:-1] for x in train])
edges = np.quantile(allv, np.linspace(0, 1, 6))
edges[0] -= 1e-9
edges[-1] += 1e-9
nb = len(edges) - 1


def ridge(X, Y, lam=1e-3):
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    scale = np.std(X, axis=0)
    scale[0] = 1.0
    scale[scale < 1e-10] = 1.0
    Xs = X / scale
    R = np.eye(X.shape[1])
    R[0, 0] = 0.0
    lam_eff = lam * np.trace(Xs.T @ Xs) / max(X.shape[1], 1)
    B = np.linalg.solve(Xs.T @ Xs + lam_eff * R, Xs.T @ Y)
    return B / scale[:, None]


def collect(data, lag=1):
    rows = []
    for s in data:
        dt = float(s["time"][lag] - s["time"][0])
        n = len(s["time"]) - lag
        # Trapezoidal average of q over the finite lag.
        qavg = np.zeros(n)
        weights = np.ones(lag + 1)
        weights[[0, -1]] = 0.5
        h = s["time"][1] - s["time"][0]
        for j, w in enumerate(weights):
            qavg += w * s["z"][j:j + n, 0]
        qavg *= h / dt
        rows.append({
            "v": s["v"][:-lag],
            "sigma": s["sigma"][:-lag],
            "z": s["z"][:-lag],
            "rv": (s["v"][lag:] - s["v"][:-lag]) / dt,
            "rsigma": (s["sigma"][lag:] - s["sigma"][:-lag]) / dt - qavg,
            "rz": (s["z"][lag:] - s["z"][:-lag]) / dt,
            "dt": dt,
        })
    return rows


def fit_model(data, kind="M2", lag=1):
    rows = collect(data, lag)
    dt = rows[0]["dt"]
    params = []
    for b in range(nb):
        chunks = []
        for r in rows:
            sel = (r["v"] >= edges[b]) & (r["v"] < edges[b + 1])
            if np.any(sel):
                chunks.append({k: (v[sel] if isinstance(v, np.ndarray) and len(v) == len(sel) else v)
                               for k, v in r.items()})
        v = np.concatenate([r["v"] for r in chunks])
        sig = np.concatenate([r["sigma"] for r in chunks])
        z = np.concatenate([r["z"] for r in chunks])
        rv = np.concatenate([r["rv"] for r in chunks])
        rs = np.concatenate([r["rsigma"] for r in chunks])
        rz = np.concatenate([r["rz"] for r in chunks])

        bv = float(np.mean(rv))
        Qv = float(np.var(rv - bv, ddof=1) * dt)

        if kind in ("M0", "M1"):
            ZB = np.zeros((5, 3))
            zpred = np.zeros_like(rz)
            for j in range(3):
                Xj = np.column_stack([np.ones(len(z)), z[:, j]])
                B = ridge(Xj, rz[:, j, None]).ravel()
                if kind == "M0" and j == 0:
                    B[0] = 0.0
                ZB[[0, 2 + j], j] = B
                zpred[:, j] = Xj @ B
            SB = np.array([np.mean(rs), 0.0])
            rspred = np.full_like(rs, SB[0])
        elif kind == "M2":
            Xfull = np.column_stack([np.ones(len(z)), sig, z])
            ZB = np.zeros((5, 3))
            zpred = np.zeros_like(rz)
            selections = ([0, 1, 2], [0, 3, 4], [0, 3, 4])
            for j, cols in enumerate(selections):
                B = ridge(Xfull[:, cols], rz[:, j, None]).ravel()
                ZB[list(cols), j] = B
                zpred[:, j] = Xfull[:, cols] @ B
            SX = np.column_stack([np.ones(len(sig)), sig])
            SB = ridge(SX, rs[:, None]).ravel()
            rspred = SX @ SB
        else:
            X = np.column_stack([np.ones(len(z)), sig, z])
            ZB = ridge(X, rz, lam=1e-2)
            zpred = X @ ZB
            SX = np.column_stack([np.ones(len(sig)), sig])
            SB = ridge(SX, rs[:, None], lam=1e-2).ravel()
            rspred = SX @ SB

        ez = rz - zpred
        es = rs - rspred
        Qz = np.cov(ez, rowvar=False, bias=False) * dt
        Qs = float(np.var(es, ddof=1) * dt)
        eig, V = np.linalg.eigh(0.5 * (Qz + Qz.T))
        eig = np.maximum(eig, 1e-10)
        Qz = (V * eig) @ V.T
        params.append({
            "bv": bv, "Qv": max(Qv, 1e-12), "SB": SB, "ZB": ZB,
            "Qs": max(Qs, 1e-12), "Qz": Qz, "n": len(sig),
        })
    return {"kind": kind, "lag": lag, "dt": dt, "params": params}


def ibin(v):
    return int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, nb - 1))


def drift(model, v, sigma, z):
    p = model["params"][ibin(v)]
    x = np.r_[1.0, sigma, z]
    return p["SB"][0] + p["SB"][1] * sigma, x @ p["ZB"], p


def autonomous_rollout(model, s, nsim=300, seed=123):
    rng = np.random.default_rng(seed)
    dt = float(s["time"][1] - s["time"][0])
    nt = len(s["time"])
    v = np.full(nsim, s["v"][0])
    sigma = np.full(nsim, s["sigma"][0])
    z = np.tile(s["z"][0], (nsim, 1))
    vh = np.zeros((nsim, nt)); sh = np.zeros((nsim, nt)); zh = np.zeros((nsim, nt, 3))
    vh[:, 0] = v; sh[:, 0] = sigma; zh[:, 0] = z
    for t in range(nt - 1):
        for n in range(nsim):
            rs, dz, p = drift(model, v[n], sigma[n], z[n])
            v[n] += p["bv"] * dt + np.sqrt(p["Qv"] * dt) * rng.standard_normal()
            v[n] = max(v[n], -0.1)
            sigma[n] = abs(sigma[n] + (z[n, 0] + rs) * dt + np.sqrt(p["Qs"] * dt) * rng.standard_normal())
            z[n] += dz * dt + rng.multivariate_normal(np.zeros(3), p["Qz"] * dt)
        vh[:, t + 1] = v; sh[:, t + 1] = sigma; zh[:, t + 1] = z
    return vh, sh, zh


models = {k: fit_model(train, k, 1) for k in ("M0", "M1", "M2", "M3")}
centers = 0.5 * (edges[:-1] + edges[1:])
emp_sigma = np.full(nb, np.nan)
for b in range(nb):
    vals = []
    for s in test:
        q = (s["v"] >= edges[b]) & (s["v"] < edges[b + 1])
        vals.extend(s["sigma"][q])
    if vals:
        emp_sigma[b] = np.mean(vals)

auto_scores = {}
for kind, model in models.items():
    simvals = [[] for _ in range(nb)]
    for i, s in enumerate(test):
        vh, sh, _ = autonomous_rollout(model, s, nsim=120, seed=5000 + i)
        for b in range(nb):
            q = (vh >= edges[b]) & (vh < edges[b + 1])
            simvals[b].extend(sh[q])
    sim_sigma = np.array([np.mean(x) if len(x) else np.nan for x in simvals])
    auto_scores[kind] = {
        "conditional_sigma": sim_sigma.tolist(),
        "conditional_sigma_rmse": float(np.sqrt(np.nanmean((sim_sigma - emp_sigma) ** 2))),
    }

# Bootstrap M2 coefficients by realization.
def vectorize(model):
    rows = []
    for p in model["params"]:
        Z = p["ZB"]
        rows.append([
            p["bv"], p["Qv"], p["SB"][0], p["SB"][1],
            Z[0, 0], Z[1, 0], Z[2, 0],
            Z[0, 1], Z[3, 1], Z[4, 1],
            Z[0, 2], Z[3, 2], Z[4, 2],
            p["Qs"], p["Qz"][0, 0], p["Qz"][1, 1], p["Qz"][2, 2],
            p["Qz"][0, 1], p["Qz"][0, 2], p["Qz"][1, 2], p["n"],
        ])
    return np.asarray(rows, float)

names = [
    "b_v", "Q_v", "a_0", "a_1",
    "c_q0", "c_qsigma", "c_qq",
    "c_p0", "c_pp", "c_pomega",
    "c_omega0", "c_omegap", "c_omegaomega",
    "Q_sigma", "Q_qq", "Q_pp", "Q_omegaomega", "Q_qp", "Q_qomega", "Q_pomega", "n",
]
base = vectorize(models["M2"])
rng = np.random.default_rng(20260714)
boots = []
for _ in range(36):
    idx = rng.integers(0, len(train), len(train))
    sample = [train[i] for i in idx]
    boots.append(vectorize(fit_model(sample, "M2", 1)))
boots = np.asarray(boots)
boot_sem = np.nanstd(boots, axis=0, ddof=1)

csv_path = TABLES / "m2_coefficients_with_bootstrap.csv"
with csv_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["bin", "v_low", "v_high", "parameter", "estimate", "bootstrap_sem"])
    for b in range(nb):
        for j, name in enumerate(names):
            w.writerow([b, edges[b], edges[b + 1], name, base[b, j], boot_sem[b, j]])

# Compact LaTeX table: report drift and diagonal diffusion; full CSV retains cross-diffusion.
tex = []
tex.append(r"\begin{table*}[t]")
tex.append(r"\caption{Scale-binned coefficients of the selected M2 generator. Uncertainties are standard deviations across 36 realization-level bootstrap refits of the 16-run training set. The full table, including cross-diffusion entries, is supplied as a machine-readable CSV.}")
tex.append(r"\label{tab:m2-coefficients}")
tex.append(r"\centering\scriptsize")
tex.append(r"\begin{tabular}{c c c c c c c c c c}")
tex.append(r"\hline\hline")
tex.append(r"$v$ bin & $n$ & $b_v$ & $a_0$ & $a_1$ & $c_{q\sigma}$ & $c_{qq}$ & $c_{pp}$ & $c_{\omega\omega}$ & $(Q_\sigma,Q_{qq},Q_{pp},Q_{\omega\omega})$\\")
tex.append(r"\hline")
for b in range(nb):
    def fs(j, digits=2):
        return f"{base[b,j]:.{digits}f}\\,$\\pm$\\,{boot_sem[b,j]:.{digits}f}"
    tex.append(
        f"{edges[b]:.2f}--{edges[b+1]:.2f} & {int(base[b,20])} & {fs(0)} & {fs(2)} & {fs(3)} & "
        f"{fs(5)} & {fs(6)} & {fs(8)} & {fs(12)} & "
        f"({base[b,13]:.3f},{base[b,14]:.3f},{base[b,15]:.3f},{base[b,16]:.3f})\\\\"
    )
tex.append(r"\hline\hline")
tex.append(r"\end{tabular}")
tex.append(r"\end{table*}")
(TABLES / "m2_coefficients.tex").write_text("\n".join(tex))

out = {
    "edges": edges.tolist(),
    "train_files": [p.name for p in files[:16]],
    "test_files": [p.name for p in files[16:24]],
    "empirical_conditional_sigma": emp_sigma.tolist(),
    "autonomous_scores": auto_scores,
    "selected_model": "M2",
    "bootstrap_refits": 36,
}
(RES / "final_intrinsic_model_summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
