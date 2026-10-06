import json, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
R = Path(__file__).parent / "results"; FIG = Path(__file__).parent.parent / "TriGuard-RC-Paper" / "figures"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4})
C = {"orig": "#999999", "rc": "#0072B2", "gb": "#D55E00", "hy": "#009E73", "xt": "#CC79A7"}
L = lambda p: [json.load(open(f)) for f in sorted(glob.glob(str(R / p)))]
mean = lambda v: float(np.mean(v)); std = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0

# ---- Fig 1: robustness on 5GAD
o, r = L("orig_5gad_s*.json"), L("rc_5gad_full_s*.json")
keys = ["clean", "byte_padding_20", "byte_insertion_10", "timing_jitter_50", "direction_corruption_20", "byte_insertion_25_heldout", "byte_gauss_heldout",
        "byte_block_occlusion_heldout", "missing_byte", "missing_timing", "missing_statistics"]
lab = ["Clean", "Byte pad 20%", "Byte repl. 10%", "Timing jitter 50%", "Direction 20%", "Byte repl. 25%*", "Gaussian noise*", "Block occlusion*", "No byte view", "No timing view", "No stats view"]
fig, ax = plt.subplots(figsize=(3.5, 3.3)); y = np.arange(len(keys)); h = 0.26
for i, (nm, runs, col, key) in enumerate([("TriGuard-ETD", o, C["orig"], "robust"), ("LightGBM", r, C["gb"], "robust_gbdt"), ("TriGuard-RC", r, C["rc"], "robust")]):
    m = [mean([x[key][k] for x in runs]) for k in keys]; s = [std([x[key][k] for x in runs]) for k in keys]
    ax.barh(y + (i - 1) * h, m, h, xerr=s, color=col, label=nm, error_kw=dict(lw=0.5, capsize=1))
ax.set_yticks(y); ax.set_yticklabels(lab); ax.invert_yaxis(); ax.set_xlim(0, 0.86); ax.axvline(9 / 11, color="k", lw=0.5, ls=":")
ax.set_xlabel("Macro-F1 (5GAD-2022; ceiling 0.818)"); ax.legend(loc="lower center", fontsize=6.5, frameon=False, ncol=3, bbox_to_anchor=(0.45, 1.0), columnspacing=1.0, handlelength=1.2)
fig.tight_layout(); fig.savefig(FIG / "robust.pdf"); plt.close(fig)

# ---- Fig 2: gate weight vs leave-one-view-out drop (5GAD)
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
for ax, nm, runs, col in [(axs[0], "TriGuard-ETD", o, C["orig"]), (axs[1], "TriGuard-RC", r, C["rc"])]:
    g = np.mean([x["gates"] for x in runs], 0)
    drop = [mean([x["robust"]["clean"] - x["robust"][k] for x in runs]) for k in ("missing_byte", "missing_timing", "missing_statistics")]
    xx = np.arange(3); ax.bar(xx - 0.19, g, 0.38, color="#56B4E9", label="mean gate weight"); ax.bar(xx + 0.19, drop, 0.38, color=col, label="F1 drop if removed")
    ax.set_xticks(xx); ax.set_xticklabels(["byte", "timing", "stats"]); ax.set_title(nm, fontsize=7.5)
axs[0].legend(fontsize=6, frameon=False, loc="upper right"); axs[0].set_ylim(0, 1.0)
fig.tight_layout(); fig.savefig(FIG / "gate.pdf"); plt.close(fig)

# ---- Fig 3: per-class F1 on TII
b = L("base_tii_s*.json"); o2 = L("orig_tii_s*.json"); r2 = L("rc_tii_full_s*.json")
classes = list(r2[0]["test"]["per_class_f1"].keys())
series = [("TriGuard-ETD", [x["test"]["per_class_f1"] for x in o2], C["orig"]), ("TriGuard-RC", [x["test"]["per_class_f1"] for x in r2], C["rc"]),
          ("TriGuard-RC-H", [x["test_hybrid"]["per_class_f1"] for x in r2], C["hy"]), ("LightGBM", [x["models"]["LightGBM"]["test"]["per_class_f1"] for x in b], C["gb"])]
fig, ax = plt.subplots(figsize=(3.5, 2.1)); w = 0.2; xx = np.arange(len(classes))
for i, (nm, runs, col) in enumerate(series):
    m = [mean([x[c] for x in runs]) for c in classes]; s = [std([x[c] for x in runs]) for c in classes]
    ax.bar(xx + (i - 1.5) * w, m, w, yerr=s, color=col, label=nm, error_kw=dict(lw=0.5, capsize=1))
ax.set_xticks(xx); ax.set_xticklabels([c.replace("Information Gathering", "Info. gath.") for c in classes], rotation=35, ha="right", fontsize=6.5)
ax.set_ylabel("Per-class F1"); ax.set_ylim(0.2, 1.03); ax.legend(fontsize=6, ncol=4, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), columnspacing=0.8, handlelength=1.0)
fig.tight_layout(); fig.savefig(FIG / "perclass_tii.pdf"); plt.close(fig)

# ---- Fig 4: ablation on TII
names = [("full", "Full"), ("no_kd", "No distillation"), ("insample_kd", "In-sample teacher"), ("no_viewdrop", "No view-drop/consist."), ("no_aug", "No augmentation"),
         ("no_ple", "No PLE"), ("byte_only", "Byte only"), ("timing_only", "Timing only"), ("stats_only", "Statistics only"), ("timing_stats", "Timing + stats")]
rows = []
for k, n in names:
    runs = L(f"rc_tii_{k}_s*.json")
    if k == "full": runs = [x for x in runs if x["seed"] <= 3]
    if runs: rows.append((n, [x["test_param"]["macro_f1"] for x in runs], [x["test"]["macro_f1"] for x in runs]))
fig, ax = plt.subplots(figsize=(3.5, 2.6)); y = np.arange(len(rows)); h = 0.38
ax.barh(y - h / 2, [mean(p) for _, p, _ in rows], h, xerr=[std(p) for _, p, _ in rows], color="#56B4E9", label="neural head", error_kw=dict(lw=0.5, capsize=1))
ax.barh(y + h / 2, [mean(q) for _, _, q in rows], h, xerr=[std(q) for _, _, q in rows], color=C["rc"], label="+ retrieval", error_kw=dict(lw=0.5, capsize=1))
ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows]); ax.invert_yaxis(); ax.set_xlim(0.3, 1.0); ax.set_xlabel("Macro-F1 (TII-SSRC-23)"); ax.legend(fontsize=6.5, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.4, 1.0))
fig.tight_layout(); fig.savefig(FIG / "ablation.pdf"); plt.close(fig)
print("figures ok")
