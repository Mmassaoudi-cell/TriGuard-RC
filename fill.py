"""Replace @@TOKEN@@ placeholders in the manuscript templates by numbers computed from results_rg (no hand-typed numbers)."""
import json, re, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent; R = HERE / "results_rg"; P = HERE.parent / "TriGuard-RC-Paper-R1"
S = json.load(open(R / "summary_rg.json"))["summary"]; OP = json.load(open(R / "summary_rg.json"))["op"]
ROB = json.load(open(R / "robust_summary.json")) if (R / "robust_summary.json").exists() else {}
TESTS = {(t["ds"], t["a"], t["b"]): t for t in json.load(open(R / "stats_tests.json"))} if (R / "stats_tests.json").exists() else {}
BOOT = json.load(open(R / "bootstrap.json")) if (R / "bootstrap.json").exists() else {}
GATE = json.load(open(R / "gate_summary.json")) if (R / "gate_summary.json").exists() else {}
SPL = json.load(open(R / "splits_summary.json")) if (R / "splits_summary.json").exists() else {}
sys.path.insert(0, str(HERE))
def Lx(prefix, ds, root=R):
    """Anchored match: {prefix}_{ds}_s<seed>.json only, excluding split-variant files such as
    "{prefix}_{ds}_s43_s1.json" that a plain glob would also match."""
    rx = re.compile(rf"^{re.escape(prefix)}_{re.escape(ds)}_s\d+\.json$")
    return [json.load(open(root / f)) for f in sorted(x.name for x in root.glob(f"{prefix}_{ds}_s*.json")) if rx.match(f)]


def extra(ds, key, model):
    """mean of a metric across seeds for FNR / ECE / accuracy of a model (read from raw run files)."""
    vals = []
    if model.startswith("TriGuard-RC-H"):
        for j in Lx("hyb", ds): vals.append(j["test"][key])
    elif model == "TriGuard-RC":
        for j in Lx("rc", f"{ds}_full"): vals.append(j["test"][key])
    elif model in ("TriGuard-ETD", "TriGuard-ETD+Ret"):
        k = "test_cal" if model == "TriGuard-ETD" else "test_ret"
        for j in Lx("orig", ds): vals.append(j[k][key])
    else:
        for j in Lx("base", ds): vals.append(j["models"][model]["test_cal"][key])
    return float(np.mean(vals))


def val(tok):
    p = tok.split(":"); k = p[0]
    if k == "F1": return f"{S[p[1]][p[2]]['f1']:.3f}"
    if k == "F1x": return f"{S[p[1]][p[2]]['f1']:.4f}"
    if k == "SD": return f"{S[p[1]][p[2]]['std']:.3f}"
    if k == "N": return f"{S[p[1]][p[2]]['n']}"
    if k == "DF": return f"{S[p[1]][p[2]]['f1'] - S[p[1]][p[3]]['f1']:+.3f}"
    if k == "DP": return f"{100 * (S[p[1]][p[2]]['f1'] - S[p[1]][p[3]]['f1']):.1f}".replace("-0.0", "0.0")
    if k == "DPA": return f"{abs(100 * (S[p[1]][p[2]]['f1'] - S[p[1]][p[3]]['f1'])):.1f}"
    if k == "FNR": return f"{100 * extra(p[1], 'fnr', p[2]):.1f}"
    if k == "ECE": return f"{100 * extra(p[1], 'ece', p[2]):.2f}"
    if k == "ACC": return f"{extra(p[1], 'accuracy', p[2]):.3f}"
    if k == "ROB": return f"{ROB[p[1]][p[2]][p[3]]:.3f}"
    if k == "ROBD": return f"{ROB[p[1]][p[2]][p[3]] - ROB[p[1]][p[4]][p[5]]:+.3f}"
    if k == "P": t = TESTS[(p[1], p[2], p[3])]; return f"{t['p']:.3g}"
    if k == "PH": t = TESTS[(p[1], p[2], p[3])]; return f"{t['p_holm']:.3g}"
    if k == "BOOT": b = BOOT[f"{p[1]}|{p[2]}|{p[3]}"]; return f"[{b[1]:+.3f}, {b[2]:+.3f}]"
    if k == "OP1": return f"{OP[p[1]][p[2]][0]:.3f}"
    if k == "OP01": return f"{OP[p[1]][p[2]][1]:.3f}"
    if k == "GATE": return f"{GATE[p[1]]['gates'][int(p[2])]:.2f}"
    if k == "DROP": return f"{GATE[p[1]]['drop'][int(p[2])]:.3f}"
    if k == "SPLM":  # mean over splits for a model
        v = [SPL[p[1]][t][p[2]] for t in ("42", "43", "44") if p[2] in SPL[p[1]][t]]; return f"{np.mean(v):.3f}"
    if k == "SPLS":
        v = [SPL[p[1]][t][p[2]] for t in ("42", "43", "44") if p[2] in SPL[p[1]][t]]; return f"{np.std(v, ddof=1):.3f}"
    if k == "SPLV":  # single split value: SPLV:ds:split_tag:model
        return f"{SPL[p[1]][p[2]][p[3]]:.3f}"
    raise KeyError(tok)


for tpl in sys.argv[1:]:
    text = (P / tpl).read_text(encoding="utf8")
    out = re.sub(r"@@([^@]+)@@", lambda m: val(m.group(1)), text)
    (P / tpl.replace(".tpl", "")).write_text(out, encoding="utf8"); print("filled", tpl)
