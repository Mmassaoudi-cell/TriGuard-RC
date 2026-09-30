"""Aggregate results/*.json into LaTeX tables, JSON summary and figures for the paper."""
import json, glob, re
from pathlib import Path
import numpy as np
R = Path(__file__).parent / "results"; PAPER = Path(__file__).parent.parent / "TriGuard-RC-Paper"; (PAPER / "figures").mkdir(parents=True, exist_ok=True)
DSN = {"5gad": "5GAD-2022", "ustc": "USTC-TFC2016", "tii": "TII-SSRC-23"}


def load(pat): return [json.load(open(f)) for f in sorted(glob.glob(str(R / pat)))]
def ms(v, d=3, pct=False):
    v = np.asarray(v, float); s = v.std(ddof=1) if len(v) > 1 else 0.0
    return rf"{v.mean():.{d}f}\,$\pm$\,{s:.{d}f}"
def col(runs, get): return [get(r) for r in runs]

summary = {}
rows = {}
for ds in DSN:
    entries = {}
    base = load(f"base_{ds}_s*.json")
    for m in ("LightGBM", "XGBoost", "ExtraTrees", "kNN"):
        if base: entries[m] = [b["models"][m]["test"] for b in base]
    o = load(f"orig_{ds}_s*.json")
    if o: entries["TriGuard-ETD"] = [x["test"] for x in o]
    rc = load(f"rc_{ds}_full_s*.json")
    if rc:
        entries["TriGuard-RC"] = [x["test"] for x in rc]
        entries["TriGuard-RC (neural head)"] = [x["test_param"] for x in rc]
        if "test_hybrid" in rc[0]: entries["TriGuard-RC-H"] = [x["test_hybrid"] for x in rc]
    rows[ds] = entries
    summary[ds] = {m: {k: float(np.mean([e[k] for e in v])) for k in ("macro_f1", "macro_f1_present", "accuracy", "fpr", "fnr", "pr_auc", "ece", "nll") if v[0].get(k) is not None} | {"n": len(v),
                   "macro_f1_std": float(np.std([e["macro_f1"] for e in v], ddof=1)) if len(v) > 1 else 0.0} for m, v in entries.items()}
(R / "summary.json").write_text(json.dumps(summary, indent=1))

ORDER = ["LightGBM", "XGBoost", "ExtraTrees", "kNN", "TriGuard-ETD", "TriGuard-RC", "TriGuard-RC-H"]
LAB = {"TriGuard-ETD": "TriGuard-ETD [orig.]", "TriGuard-RC": r"\textbf{TriGuard-RC (ours)}", "TriGuard-RC-H": r"\textbf{TriGuard-RC-H (ours)}"}
lines = []
for ds in DSN:
    ent = rows[ds]; best = max(np.mean([e["macro_f1"] for e in v]) for v in ent.values())
    first = True
    for m in ORDER:
        if m not in ent: continue
        v = ent[m]; f1 = np.mean([e["macro_f1"] for e in v])
        f1s = ms([e["macro_f1"] for e in v]); f1s = rf"\textbf{{{f1s}}}" if abs(f1 - best) < 1e-9 else f1s
        head = rf"\multirow{{{sum(1 for x in ORDER if x in ent)}}}{{*}}{{\rotatebox{{90}}{{{DSN[ds]}}}}}" if first else ""
        first = False
        lines.append(f"{head} & {LAB.get(m, m)} & {f1s} & {ms([e['accuracy'] for e in v])} & {ms([e['fpr']*100 for e in v], 2)} & {ms([e['fnr']*100 for e in v], 2)} & {ms([e['pr_auc'] for e in v])} & {ms([e['ece'] for e in v])} \\\\")
    lines.append(r"\midrule")
(PAPER / "table_main.tex").write_text("\n".join(lines[:-1]))

# robustness (5GAD): orig vs RC
o = load("orig_5gad_s*.json"); r = load("rc_5gad_full_s*.json")
if o and r and "robust" in r[0]:
    L = []
    names = {"clean": "Clean", "byte_padding_20": r"Byte padding 20\%", "byte_insertion_10": "Random-byte insertion 10\%$^\dagger$", "timing_jitter_20": "Timing jitter 20\%$^\dagger$",
             "timing_jitter_50": "Timing jitter 50\%$^\dagger$", "direction_corruption_20": "Direction corruption 20\%", "byte_insertion_25_heldout": "Random-byte insertion 25\%$^\ddagger$",
             "byte_block_occlusion_heldout": "Central block occlusion$^\ddagger$", "byte_gauss_heldout": "Gaussian pixel noise$^\ddagger$", "missing_byte": "Missing byte view",
             "missing_timing": "Missing timing view", "missing_statistics": "Missing statistics view"}
    for k, n in names.items():
        a = [x["robust"][k] for x in o]; b = [x["robust"][k] for x in r]
        L.append(f"{n} & {ms(a)} & {ms(b)} & {np.mean(b)-np.mean(a):+.3f} \\\\")
    (PAPER / "table_robust.tex").write_text("\n".join(L))
    summary["robust"] = {k: {"orig": float(np.mean([x["robust"][k] for x in o])), "rc": float(np.mean([x["robust"][k] for x in r]))} for k in names}

# stage table
L = []
for ds in DSN:
    rcr = load(f"rc_{ds}_full_s*.json")
    if not rcr: continue
    a = [x["test_param"]["macro_f1"] for x in rcr]; b = [x["test"]["macro_f1"] for x in rcr]; c = [x["test_hybrid"]["macro_f1"] for x in rcr]
    L.append(f"{DSN[ds]} & {ms(a)} & {ms(b)} & {ms(c)} \\\\")
(PAPER / "table_stages.tex").write_text("\n".join(L))

# efficiency
eff = json.load(open(R / "efficiency.json")); L = []
for ds in DSN:
    e = eff[ds]
    L.append(f"{DSN[ds]} & {e['orig']['params']/1e3:.0f}\,k & {e['rc']['params']/1e3:.0f}\,k & {e['orig']['macs']/1e6:.2f} & {e['rc']['macs']/1e6:.2f} & {e['orig']['lat_ms']:.2f} & {e['rc']['lat_ms']:.2f} & {e['retrieval_ms']:.2f} & {e['memory_mb']:.1f} \\\\")
(PAPER / "table_eff.tex").write_text("\n".join(L))

# ablation (TII + USTC)
VN = [("full", "Full TriGuard-RC"), ("no_kd", "No teacher distillation"), ("insample_kd", "In-sample (non-cross-fitted) teacher"), ("no_viewdrop", "No view dropout / consistency"),
      ("no_aug", "No corruption augmentation"), ("no_ple", "No PLE (single-bin statistics)"), ("byte_only", "Byte view only"), ("timing_only", "Timing view only"),
      ("stats_only", "Statistics view only"), ("timing_stats", "Timing + statistics")]
L = []; abl = {}
for v, n in VN:
    cells = []
    for ds in ("tii", "ustc"):
        runs = load(f"rc_{ds}_{v}_s*.json")
        if v == "full": runs = [x for x in runs if x["seed"] <= 3]
        if not runs: cells += ["--", "--"]; continue
        p = [x["test_param"]["macro_f1"] for x in runs]; b = [x["test"]["macro_f1"] for x in runs]
        abl[f"{ds}_{v}"] = (float(np.mean(p)), float(np.mean(b))); cells += [ms(p), ms(b)]
    L.append(f"{n} & " + " & ".join(cells) + " \\\\")
(PAPER / "table_ablation.tex").write_text("\n".join(L)); summary["ablation"] = abl

# robustness attribution on 5GAD (augmentation / subset training)
RK = [("clean", "Clean"), ("byte_insertion_10", r"Byte repl. 10\%"), ("byte_insertion_25_heldout", r"Byte repl. 25\%$^\ddagger$"), ("byte_gauss_heldout", r"Gaussian$^\ddagger$"),
      ("byte_block_occlusion_heldout", r"Block occl.$^\ddagger$"), ("missing_byte", "No byte"), ("missing_statistics", "No stats")]
L = []
for v, n in [("full", "Full TriGuard-RC"), ("no_aug", "No corruption augmentation"), ("no_viewdrop", "No view dropout / consistency")]:
    runs = [x for x in load(f"rc_5gad_{v}_s*.json") if x["seed"] <= 3 and "robust" in x]
    if runs: L.append(f"{n} & " + " & ".join(f"{np.mean([x['robust'][k] for x in runs]):.3f}" for k, _ in RK) + r" \\")
(PAPER / "table_robust_abl.tex").write_text("\n".join(L))
(PAPER / "table_robust_abl_head.tex").write_text(" & ".join(h for _, h in RK) + "\n")
(R / "summary.json").write_text(json.dumps(summary, indent=1))
