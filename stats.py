import json, glob
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import f1_score
R = Path(__file__).parent / "results"
L = lambda p: [json.load(open(f)) for f in sorted(glob.glob(str(R / p)))]
out = {}
for ds in ("5gad", "tii", "ustc"):
    rc, o, b = L(f"rc_{ds}_full_s*.json"), L(f"orig_{ds}_s*.json"), L(f"base_{ds}_s*.json")
    f = lambda rs, k="test": [x[k]["macro_f1"] for x in rs]
    d = {}
    d["rc_vs_orig_welch_p"] = float(stats.ttest_ind(f(rc), f(o), equal_var=False).pvalue) if np.std(f(rc)) + np.std(f(o)) > 0 else None
    d["rc_vs_orig_diff"] = float(np.mean(f(rc)) - np.mean(f(o)))
    d["neural_vs_orig_diff"] = float(np.mean(f(rc, "test_param")) - np.mean(f(o)))
    d["hyb_vs_lgbm_diff"] = float(np.mean(f(rc, "test_hybrid")) - np.mean([x["models"]["LightGBM"]["test"]["macro_f1"] for x in b]))
    # paired bootstrap on seed-1 test predictions: hybrid vs balanced-LightGBM expert, and RC vs neural head
    z = np.load(R / f"probs_{ds}_s1.npz"); y = z["y"]; C = z["hybrid"].shape[1]; rng = np.random.default_rng(0)
    pr = {k: z[k].astype(np.float32).argmax(1) for k in ("param", "blend", "gbdt", "hybrid")}
    def boot(a, bb, B=1000):
        diffs = []
        for _ in range(B):
            ix = rng.integers(0, len(y), len(y))
            diffs.append(f1_score(y[ix], pr[a][ix], labels=np.arange(C), average="macro", zero_division=0) - f1_score(y[ix], pr[bb][ix], labels=np.arange(C), average="macro", zero_division=0))
        return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    d["boot_hybrid_minus_gbdt"] = boot("hybrid", "gbdt"); d["boot_blend_minus_param"] = boot("blend", "param")
    out[ds] = d; print(ds, d, flush=True)
json.dump(out, open(R / "stats.json", "w"), indent=1)
