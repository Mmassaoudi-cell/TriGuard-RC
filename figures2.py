import json, glob, re
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent; R = HERE / "results_rg"; FIG = HERE.parent / "TriGuard-RC-Paper-R1" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4})
COL = {"TriGuard-ETD": "#999999", "LightGBM": "#D55E00", "LightGBM+Aug": "#E69F00", "FT-Transformer": "#CC79A7", "TriGuard-RC": "#0072B2"}
L = lambda p: [json.load(open(f)) for f in sorted(glob.glob(str(R / p)))]


def Lx(prefix, ds, root=R):
    """Anchored match, excludes split-variant files like "{prefix}_{ds}_s43_s1.json"."""
    rx = re.compile(rf"^{re.escape(prefix)}_{re.escape(ds)}_s\d+\.json$")
    return [json.load(open(root / f)) for f in sorted(x.name for x in root.glob(f"{prefix}_{ds}_s*.json")) if rx.match(f)]


S = json.load(open(R / "robust_summary.json"))

# ---- robustness (two benchmarks)
keys = ["clean", "byte_insertion_10", "u_byte_salt_pepper", "u_byte_row_shuffle", "u_stats_noise", "missing_byte", "missing_statistics"]
lab = ["Clean", "Byte repl. 10%", "Salt-and-pepper*", "Row shuffle*", "Statistics noise*", "No byte view", "No statistics view"]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
for ax, ds, title in zip(axs, ("5gad", "ustc"), ("5GAD-2022", "USTC-TFC2016")):
    y = np.arange(len(keys)); h = 0.16
    for i, m in enumerate(COL):
        v = [S[ds][m][k] if S[ds][m][k] is not None else 0 for k in keys]
        ax.barh(y + (i - 2) * h, v, h, color=COL[m], label=m)
    ax.set_yticks(y); ax.set_yticklabels(lab); ax.invert_yaxis(); ax.set_title(title, fontsize=8); ax.set_xlabel("Macro-F1")
axs[0].legend(loc="lower center", ncol=5, fontsize=6.5, frameon=False, bbox_to_anchor=(1.05, 1.08), columnspacing=1.0, handlelength=1.2)
fig.tight_layout(); fig.savefig(FIG / "robust.pdf"); plt.close(fig)

# ---- gate weight vs leave-one-view-out drop
G = json.load(open(R / "gate_summary.json"))
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for ax, ds, title in zip(axs, ("5gad", "ustc"), ("5GAD-2022", "USTC-TFC2016")):
    if ds not in G: continue
    xx = np.arange(3); ax.bar(xx - 0.19, G[ds]["gates"], 0.38, color="#56B4E9", label="mean gate weight"); ax.bar(xx + 0.19, G[ds]["drop"], 0.38, color="#0072B2", label="F1 drop if removed")
    ax.set_xticks(xx); ax.set_xticklabels(["byte", "timing", "stats"]); ax.set_title(title, fontsize=7.5)
axs[0].legend(fontsize=6, frameon=False, loc="upper left"); axs[0].set_ylim(0, 1.0)
fig.tight_layout(); fig.savefig(FIG / "gate.pdf"); plt.close(fig)

# ---- per-class F1 on ISCXVPN2016 (independent benchmark)
def per_class(ds):
    out = {}
    for b in Lx("base", ds): out.setdefault("LightGBM", []).append(b["models"]["LightGBM"]["test_cal"]["per_class_f1"])
    for o in Lx("orig", ds): out.setdefault("TriGuard-ETD", []).append(o["test_cal"]["per_class_f1"])
    for r in L(f"rc_{ds}_full_s*.json"): out.setdefault("TriGuard-RC", []).append(r["test"]["per_class_f1"]); out.setdefault("TriGuard-RC-H", []).append(r["test_hybrid"]["per_class_f1"])
    return out
pc = per_class("iscx")
if pc.get("TriGuard-RC"):
    classes = list(pc["TriGuard-RC"][0].keys()); cols = {"TriGuard-ETD": "#999999", "TriGuard-RC": "#0072B2", "TriGuard-RC-H": "#009E73", "LightGBM": "#D55E00"}
    fig, ax = plt.subplots(figsize=(3.5, 2.2)); w = 0.2; xx = np.arange(len(classes))
    for i, m in enumerate(cols):
        if m not in pc: continue
        mean = [np.mean([x[c] for x in pc[m]]) for c in classes]; sd = [np.std([x[c] for x in pc[m]], ddof=1) if len(pc[m]) > 1 else 0 for c in classes]
        ax.bar(xx + (i - 1.5) * w, mean, w, yerr=sd, color=cols[m], label=m, error_kw=dict(lw=0.5, capsize=1))
    ax.set_xticks(xx); ax.set_xticklabels([c.replace("_", " ") for c in classes], rotation=40, ha="right", fontsize=6.3)
    ax.set_ylabel("Per-class F1"); ax.set_ylim(0.0, 1.05); ax.legend(fontsize=6, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.0)
    fig.tight_layout(); fig.savefig(FIG / "perclass_iscx.pdf"); plt.close(fig)
print("figures2 ok")
