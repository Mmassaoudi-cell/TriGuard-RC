"""Aggregate results_rg/*.json into the tables, statistics and figures of the revised manuscript."""
import glob, json, itertools, re
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score, roc_curve

HERE = Path(__file__).parent
R = HERE / "results_rg"; OLD = HERE / "results"; PAPER = HERE.parent / "TriGuard-RC-Paper-R1"; (PAPER / "figures").mkdir(parents=True, exist_ok=True)
DS = ["5gad", "ustc", "iscx", "tii"]
DSN = {"5gad": "5GAD-2022", "ustc": "USTC-TFC2016", "iscx": "ISCXVPN2016", "tii": "TII-SSRC-23"}
BS = "\\"  # single backslash


def L(pat, root=R): return [json.load(open(f)) for f in sorted(glob.glob(str(root / pat)))]
def Lx(prefix, ds, root=R):
    """Like L(f"{prefix}_{ds}_s*.json") but anchored: excludes split-variant files such as
    "{prefix}_{ds}_s43_s1.json", which the plain glob would otherwise also match."""
    rx = re.compile(rf"^{re.escape(prefix)}_{re.escape(ds)}_s\d+\.json$")
    return [json.load(open(root / f)) for f in sorted(x.name for x in root.glob(f"{prefix}_{ds}_s*.json")) if rx.match(f)]
def ms(v, d=3): v = np.asarray(v, float); s = v.std(ddof=1) if len(v) > 1 else 0.0; return f"{v.mean():.{d}f}{BS},$" + f"{BS}pm${BS},{s:.{d}f}"
def wr(name, rows): (PAPER / name).write_text("\n".join(rows) + "\n", encoding="utf8")
def eol(cells): return " & ".join(cells) + f" {BS}{BS}"
F1 = "macro_f1_present"

# ------------------------------------------------------------------ collect per-dataset, per-model metric lists
MODELS = ["LightGBM", "XGBoost", "ExtraTrees", "kNN", "FT-Transformer", "LightGBM+Aug", "LightGBM+kNN", "TriGuard-ETD", "TriGuard-ETD+Ret", "TriGuard-RC", "TriGuard-RC-H"]
def collect(ds):
    out = {}
    for b in Lx("base", ds):
        for m, v in b["models"].items(): out.setdefault(m, []).append(v["test_cal"])
    for o in Lx("orig", ds):
        out.setdefault("TriGuard-ETD", []).append(o["test_cal"]); out.setdefault("TriGuard-ETD+Ret", []).append(o["test_ret"])
    for r in L(f"rc_{ds}_full_s*.json"):
        out.setdefault("TriGuard-RC", []).append(r["test"]); out.setdefault("TriGuard-RC (head)", []).append(r["test_param"])
    for h in Lx("hyb", ds): out.setdefault("TriGuard-RC-H", []).append(h["test"])
    return out
DATA = {ds: collect(ds) for ds in DS}
summary = {ds: {m: {"f1": float(np.mean([e[F1] for e in v])), "std": float(np.std([e[F1] for e in v], ddof=1)) if len(v) > 1 else 0.0, "n": len(v)} for m, v in DATA[ds].items()} for ds in DS}

# ------------------------------------------------------------------ main table
SHOW = ["LightGBM", "XGBoost", "kNN", "FT-Transformer", "LightGBM+Aug", "LightGBM+kNN", "TriGuard-ETD", "TriGuard-ETD+Ret", "TriGuard-RC", "TriGuard-RC-H"]
LAB = {"TriGuard-RC": f"{BS}textbf{{TriGuard-RC}}", "TriGuard-RC-H": f"{BS}textbf{{TriGuard-RC-H}}", "TriGuard-ETD+Ret": "TriGuard-ETD + Ret.", "LightGBM+kNN": "LightGBM + $k$-NN", "LightGBM+Aug": "LightGBM + Aug."}
rows = []
for ds in DS:
    ent = {m: v for m, v in DATA[ds].items() if m in SHOW}
    if not ent: continue
    best = max(np.mean([e[F1] for e in v]) for v in ent.values()); first = True; shown = [m for m in SHOW if m in ent]
    for m in shown:
        v = ent[m]; f1 = ms([e[F1] for e in v]); f1 = f"{BS}textbf{{{f1}}}" if abs(np.mean([e[F1] for e in v]) - best) < 1e-9 else f1
        head = f"{BS}multirow{{{len(shown)}}}{{*}}{{{BS}rotatebox{{90}}{{{DSN[ds]}}}}}" if first else ""; first = False
        rows.append(eol([head, LAB.get(m, m), f1, ms([e["accuracy"] for e in v]), ms([e["fnr"] * 100 for e in v], 2), ms([e["pr_auc"] for e in v]), ms([e["ece"] * 100 for e in v], 2)]))
    rows.append(f"{BS}midrule")
wr("table_main.tex", rows[:-1])

# additional baselines (compact)
rows = []
for ds in DS:
    d = DATA[ds]
    if "ExtraTrees" in d: rows.append(eol([DSN[ds]] + [ms([e[F1] for e in d[m]]) for m in ("ExtraTrees", "kNN")]))
wr("table_extra.tex", rows)

# stages
rows = []
for ds in DS:
    d = DATA[ds]
    if "TriGuard-RC" in d: rows.append(eol([DSN[ds]] + [ms([e[F1] for e in d[m]]) for m in ("TriGuard-RC (head)", "TriGuard-RC", "TriGuard-RC-H")]))
wr("table_stages.tex", rows)

# ------------------------------------------------------------------ statistics: Welch + Holm, bootstrap on pooled seeds
def welch(a, b):
    if np.std(a) + np.std(b) == 0: return 1.0
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)
pairs = [("TriGuard-RC", "TriGuard-ETD"), ("TriGuard-RC", "TriGuard-ETD+Ret"), ("TriGuard-RC", "LightGBM+Aug"), ("TriGuard-RC", "XGBoost"), ("TriGuard-RC-H", "XGBoost"), ("TriGuard-RC-H", "LightGBM+kNN"), ("TriGuard-RC-H", "FT-Transformer")]
tests = []
for ds in DS:
    for a, b in pairs:
        if a in DATA[ds] and b in DATA[ds] and len(DATA[ds][a]) > 1 and len(DATA[ds][b]) > 1:
            xa, xb = [e[F1] for e in DATA[ds][a]], [e[F1] for e in DATA[ds][b]]
            tests.append(dict(ds=ds, a=a, b=b, diff=float(np.mean(xa) - np.mean(xb)), p=welch(xa, xb)))
order = np.argsort([t["p"] for t in tests]); m_ = len(tests); run = 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (m_ - rank) * tests[i]["p"])); tests[i]["p_holm"] = run
json.dump(tests, open(R / "stats_tests.json", "w"), indent=1)
rows = []
for t in tests:
    rows.append(eol([DSN[t["ds"]], f"{t['a']} vs {t['b']}".replace("TriGuard-", "TG-"), f"{t['diff']:+.4f}", f"{t['p']:.3g}", f"{t['p_holm']:.3g}"]))
wr("table_stats.tex", rows)

# ------------------------------------------------------------------ operating point analysis from stored probabilities
def tpr_at_fpr(y, P, fpr):
    vals = []
    for c in range(P.shape[1]):
        pos = y == c
        if pos.sum() == 0 or (~pos).sum() == 0: continue
        thr = np.quantile(P[~pos, c], 1 - fpr); vals.append(float((P[pos, c] > thr).mean()))
    return float(np.mean(vals))
def load_probs(ds, seed):
    out = {}
    z = R / f"probs_base_{ds}_s{seed}.npz"
    if z.exists():
        a = np.load(z); out["y"] = a["y"]
        for k in ("LightGBM", "XGBoost", "LightGBM+Aug", "LightGBM+kNN", "FT-Transformer"): out[k] = a[k].astype(np.float32)
    z = R / f"probs_orig_{ds}_s{seed}.npz"
    if z.exists(): a = np.load(z); out["TriGuard-ETD"] = a["cal"].astype(np.float32); out["TriGuard-ETD+Ret"] = a["ret"].astype(np.float32)
    z = R / f"probs_rc_{ds}_full_s{seed}.npz"
    if z.exists(): a = np.load(z); out["TriGuard-RC"] = a["blend"].astype(np.float32)
    z = R / f"probs_hyb_{ds}_s{seed}.npz"
    if z.exists(): a = np.load(z); out["TriGuard-RC-H"] = a["hybrid"].astype(np.float32)
    return out
OP = {}
rows = []
for ds in DS:
    per = {}
    for s in range(1, 6):
        P = load_probs(ds, s)
        for m in ("LightGBM", "XGBoost", "LightGBM+Aug", "FT-Transformer", "TriGuard-ETD", "TriGuard-ETD+Ret", "TriGuard-RC", "TriGuard-RC-H"):
            if m in P and "y" in P: per.setdefault(m, []).append((tpr_at_fpr(P["y"], P[m], 0.01), tpr_at_fpr(P["y"], P[m], 0.001)))
    OP[ds] = {m: np.mean(v, 0).tolist() for m, v in per.items()}
for m in ("LightGBM", "XGBoost", "LightGBM+Aug", "FT-Transformer", "TriGuard-ETD", "TriGuard-ETD+Ret", "TriGuard-RC", "TriGuard-RC-H"):
    cells = [LAB.get(m, m)]
    for ds in DS:
        v = OP[ds].get(m); cells += [f"{v[0]:.3f}", f"{v[1]:.3f}"] if v else ["--", "--"]
    rows.append(eol(cells))
wr("table_op.tex", rows)

# rare-class recall with bootstrap CI (seed-1 probabilities)
rare = {}
rows = []
rng = np.random.default_rng(0)
for ds in DS:
    P = load_probs(ds, 1)
    if "y" not in P or not (R / f"rc_{ds}_full_s1.json").exists(): continue
    y = P["y"]; cls = json.load(open(R / f"rc_{ds}_full_s1.json"))["test"]["per_class_f1"]; names = list(cls.keys())
    cnt = np.bincount(y, minlength=len(names)); cnt2 = cnt.copy(); cnt2[cnt2 == 0] = 10 ** 9
    f1s = np.array([np.mean([r["test"]["per_class_f1"][n] for r in L(f"rc_{ds}_full_s*.json")]) for n in names]); f1s[cnt == 0] = 9
    for c in dict.fromkeys([int(cnt2.argmin()), int(f1s.argmin())]):
        cells = [DSN[ds], f"{names[c].replace('_', ' ')} ({int((y == c).sum())})"]
        for m in ("LightGBM", "XGBoost", "TriGuard-ETD", "TriGuard-RC", "TriGuard-RC-H"):
            if m not in P: cells.append("--"); continue
            pred = P[m].argmax(1); idx = np.flatnonzero(y == c); rec = (pred[idx] == c).astype(float)
            bs = [rec[rng.integers(0, len(rec), len(rec))].mean() for _ in range(2000)]
            cells.append(f"{rec.mean():.2f} [{np.percentile(bs, 2.5):.2f}, {np.percentile(bs, 97.5):.2f}]")
        rows.append(eol(cells))
wr("table_rare.tex", rows)

# bootstrap CI on pooled seeds (RC-H vs LightGBM; RC vs ETD)
boot = {}
for ds in DS:
    for a, b in (("TriGuard-RC-H", "XGBoost"), ("TriGuard-RC", "TriGuard-ETD"), ("TriGuard-RC", "LightGBM+Aug")):
        d = []
        for s in range(1, 6):
            P = load_probs(ds, s)
            if "y" in P and a in P and b in P:
                y = P["y"]; C = P[a].shape[1]
                for _ in range(200):
                    ix = rng.integers(0, len(y), len(y))
                    fa = f1_score(y[ix], P[a][ix].argmax(1), labels=np.unique(y), average="macro", zero_division=0)
                    fb = f1_score(y[ix], P[b][ix].argmax(1), labels=np.unique(y), average="macro", zero_division=0); d.append(fa - fb)
        if d: boot[f"{ds}|{a}|{b}"] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
json.dump(boot, open(R / "bootstrap.json", "w"), indent=1)

# ------------------------------------------------------------------ robustness (5GAD and USTC)
CONDS = [("clean", "Clean"), ("byte_insertion_10", f"Byte repl. 10{BS}%$^{BS}dagger$"), ("timing_jitter_50", f"Timing jitter 50{BS}%$^{BS}dagger$"), ("missing_byte", "No byte view"),
         ("missing_statistics", "No statistics view"), ("missing_timing", "No timing view"), ("u_byte_row_shuffle", f"Row shuffle$^{BS}ddagger$"), ("u_byte_shift", f"Row shift$^{BS}ddagger$"),
         ("u_byte_salt_pepper", f"Salt-and-pepper$^{BS}ddagger$"), ("u_timing_packet_drop", f"Packet drop$^{BS}ddagger$"), ("u_stats_noise", f"Statistics noise$^{BS}ddagger$")]
RM = ["TriGuard-ETD", "LightGBM", "LightGBM+Aug", "FT-Transformer", "TriGuard-RC"]
def factor(ds):
    rep = json.load(open(HERE / "manifests_rg" / ds / "regroup_report.json")); sup = np.array(rep["test_support"])
    return len(sup) / max(int((sup > 0).sum()), 1)   # robustness scores are stored as means over all labels; rescale to classes with support
def sc(d, f): return {k: v * f for k, v in d.items()}
def robust_collect(ds):
    f = factor(ds)
    out = {m: [] for m in RM}
    for o in Lx("orig", ds):
        if "robust" in o: out["TriGuard-ETD"].append(sc(o["robust"], f))
    for r in L(f"rc_{ds}_full_s*.json"):
        if "robust" in r: out["TriGuard-RC"].append(sc(r["robust"], f))
    for b in Lx("base", ds):
        for m, rb in b.get("robust", {}).items(): out[m].append(sc(rb, f))
    return out
RB = {ds: robust_collect(ds) for ds in ("5gad", "ustc")}
rows = []
for k, n in CONDS:
    cells = [n]
    for ds in ("5gad", "ustc"):
        for m in RM:
            v = [x[k] for x in RB[ds][m]]; cells.append(f"{np.mean(v):.3f}" if v else "--")
    rows.append(eol(cells))
wr("table_robust.tex", rows)
json.dump({ds: {m: {k: float(np.mean([x[k] for x in RB[ds][m]])) if RB[ds][m] else None for k, _ in CONDS} for m in RM} for ds in RB}, open(R / "robust_summary.json", "w"), indent=1)

# gate weights vs leave-one-view-out drop (5GAD, USTC)
gate = {}
for ds in ("5gad", "ustc"):
    rr = [r for r in L(f"rc_{ds}_full_s*.json") if "robust" in r]
    if rr:
        g = np.mean([r["gates"] for r in rr], 0).tolist()
        drop = [factor(ds) * float(np.mean([r["robust"]["clean"] - r["robust"][k] for r in rr])) for k in ("missing_byte", "missing_timing", "missing_statistics")]
        gate[ds] = dict(gates=g, drop=drop)
json.dump(gate, open(R / "gate_summary.json", "w"), indent=1)

# ------------------------------------------------------------------ split variability (three group-disjoint splits)
rows = []; SPL = {}
for ds in ("ustc", "iscx", "tii"):
    row = {}
    for tag, name in (("42", ds), ("43", f"{ds}_s43"), ("44", f"{ds}_s44")):
        cell = {}
        for r in [x for x in L(f"rc_{name}_full_s*.json") if x["seed"] == 1]: cell["TriGuard-RC"] = r["test"][F1]
        for h in [x for x in Lx("hyb", name) if x["seed"] == 1]: cell["TriGuard-RC-H"] = h["test"][F1]
        for b in [x for x in Lx("base", name) if x["seed"] == 1]: cell["XGBoost"] = b["models"]["XGBoost"]["test_cal"][F1]; cell["LightGBM"] = b["models"]["LightGBM"]["test_cal"][F1]
        for o in [x for x in Lx("orig", name) if x["seed"] == 1]: cell["TriGuard-ETD"] = o["test_cal"][F1]
        row[tag] = cell
    SPL[ds] = row
    for m in ("XGBoost", "TriGuard-ETD", "TriGuard-RC", "TriGuard-RC-H"):
        v = [row[t][m] for t in ("42", "43", "44") if m in row[t]]
        rows.append(eol([DSN[ds] if m == "XGBoost" else "", LAB.get(m, m)] + [f"{row[t][m]:.3f}" if m in row[t] else "--" for t in ("42", "43", "44")] + [ms(v) if len(v) > 1 else "--"]))
    rows.append(f"{BS}midrule")
wr("table_splits.tex", rows[:-1]); json.dump(SPL, open(R / "splits_summary.json", "w"), indent=1)

# ------------------------------------------------------------------ manifest effect (original vs duplicate-aware)
rows = []
def old_collect(ds):
    o = {}
    for r in L(f"rc_{ds}_full_s*.json", OLD): o.setdefault("TriGuard-RC", []).append(r["test"][F1]); o.setdefault("TriGuard-RC-H", []).append(r["test_hybrid"][F1])
    for r in L(f"orig_{ds}_s*.json", OLD): o.setdefault("TriGuard-ETD", []).append(r["test"][F1])
    for r in L(f"base_{ds}_s*.json", OLD): o.setdefault("LightGBM", []).append(r["models"]["LightGBM"]["test"][F1])
    return o
for ds in ("5gad", "ustc", "tii"):
    old = old_collect(ds)
    for m in ("LightGBM", "TriGuard-ETD", "TriGuard-RC", "TriGuard-RC-H"):
        if m in old and m in DATA[ds]:
            a, b = np.mean(old[m]), np.mean([e[F1] for e in DATA[ds][m]])
            rows.append(eol([DSN[ds] if m == "LightGBM" else "", LAB.get(m, m), f"{a:.3f}", f"{b:.3f}", f"{b - a:+.3f}"]))
    rows.append(f"{BS}midrule")
wr("table_manifest.tex", rows[:-1])

# ------------------------------------------------------------------ ablations (USTC + 5GAD robustness attribution)
VN = [("full", "Full TriGuard-RC"), ("no_kd", "No teacher distillation"), ("insample_kd", "In-sample teacher"), ("no_viewdrop", "No view dropout / consistency"),
      ("no_aug", "No corruption augmentation"), ("full_aug", "Augmentation on all samples"), ("no_ple", "No PLE"), ("byte_only", "Byte view only"), ("timing_only", "Timing view only"),
      ("stats_only", "Statistics view only"), ("timing_stats", "Timing + statistics")]
rows = []
for v, n in VN:
    cells = [n]; have = False
    for ds in ("ustc", "iscx"):
        runs = [x for x in L(f"rc_{ds}_{v}_s*.json") if x["seed"] == 1]
        if runs: cells += [f"{runs[0]['test_param'][F1]:.3f}", f"{runs[0]['test'][F1]:.3f}"]; have = True
        else: cells += ["--", "--"]
    if have: rows.append(eol(cells))
wr("table_ablation.tex", rows)
rk = [("clean", "Clean"), ("byte_insertion_10", f"Byte 10{BS}%"), ("u_byte_salt_pepper", f"Salt-pepper$^{BS}ddagger$"), ("u_byte_row_shuffle", f"Row shuffle$^{BS}ddagger$"), ("missing_byte", "No byte"), ("missing_statistics", "No stats")]
rows = []
for v, n in (("full", "Full TriGuard-RC"), ("full_aug", "Augmentation on all samples"), ("no_aug", "No corruption augmentation"), ("no_viewdrop", "No view dropout / consistency")):
    runs = [x for x in L(f"rc_5gad_{v}_s*.json") if x["seed"] <= 3 and "robust" in x]
    if runs: rows.append(eol([n] + [f"{factor('5gad') * np.mean([x['robust'][k] for x in runs]):.3f}" for k, _ in rk]))
wr("table_robust_abl.tex", rows)

# ------------------------------------------------------------------ efficiency
eff = HERE / "results" / "efficiency.json"
if (R / "efficiency.json").exists(): eff = R / "efficiency.json"
E = json.load(open(eff)); rows = []
for ds in DS:
    if ds not in E: continue
    e = E[ds]; rows.append(eol([DSN[ds], f"{e['orig']['params'] / 1e3:.0f}{BS},k", f"{e['rc']['params'] / 1e3:.0f}{BS},k", f"{e['orig']['macs'] / 1e6:.2f}", f"{e['rc']['macs'] / 1e6:.2f}",
                                f"{e['orig']['lat_ms']:.2f}", f"{e['rc']['lat_ms']:.2f}", f"{e['retrieval_ms']:.2f}", f"{e['memory_mb']:.1f}"]))
wr("table_eff.tex", rows)
json.dump(dict(summary=summary, op=OP), open(R / "summary_rg.json", "w"), indent=1)
print(json.dumps(summary, indent=1)[:5000])
