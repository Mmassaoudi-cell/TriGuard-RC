"""TriGuard-RC-H: validation-selected boosted-tree expert (LightGBM, XGBoost or their mean) blended with the
calibrated TriGuard-RC posterior.  Uses only stored validation/test probabilities of the trained models."""
import glob, json, sys
from pathlib import Path
import numpy as np
import evalx
from calib import macro, fit_T_prob, temper_prob, pick
from sklearn.metrics import log_loss

R = Path(__file__).parent / "results_rg"
NB = 3  # tree experts are trained with seeds 1-3 and reused cyclically for further TriGuard seeds


def run(ds):
    for f in sorted(glob.glob(str(R / f"rc_{ds}_full_s*.json"))):
        seed = int(f.split("_s")[-1].split(".")[0]); out = R / f"hyb_{ds}_s{seed}.json"
        rc = json.load(open(f)); cl = list(rc["test"]["per_class_f1"].keys()); C = len(cl)
        bseed = (seed - 1) % NB + 1; bp = R / f"probs_base_{ds}_s{bseed}.npz"; rp = R / f"probs_rc_{ds}_full_s{seed}.npz"
        if not bp.exists() or not rp.exists(): continue
        b, r = np.load(bp), np.load(rp); yv, yt = r["yv"], r["y"]
        rv, rt = r["blend_val"].astype(np.float64), r["blend"].astype(np.float64)
        cand = {"LightGBM": (b["LightGBM__val"].astype(np.float64), b["LightGBM"].astype(np.float64)),
                "XGBoost": (b["XGBoost__val"].astype(np.float64), b["XGBoost"].astype(np.float64))}
        cand["Mean"] = ((cand["LightGBM"][0] + cand["XGBoost"][0]) / 2, (cand["LightGBM"][1] + cand["XGBoost"][1]) / 2)
        scores = {k: (round(macro(yv, v[0].argmax(1), C), 4), -float(log_loss(yv, np.clip(v[0], 1e-9, 1), labels=np.arange(C)))) for k, v in cand.items()}
        e = max(scores, key=scores.get); ev, et = cand[e]
        mu = pick(yv, C, np.linspace(0, 1, 11), lambda u: (1 - u) * rv + u * ev)
        hv, ht = (1 - mu) * rv + mu * ev, (1 - mu) * rt + mu * et; T = fit_T_prob(hv, yv, C); pt = temper_prob(ht, T)
        res = dict(ds=ds, seed=seed, expert=e, mu=mu, T=T, test=evalx.full_metrics(yt, pt, cl))
        out.write_text(json.dumps(res, indent=1)); np.savez_compressed(R / f"probs_hyb_{ds}_s{seed}.npz", y=yt, hybrid=pt.astype(np.float16))
        print(ds, seed, e, round(mu, 1), round(res["test"]["macro_f1_present"], 4), flush=True)


if __name__ == "__main__":
    for ds in (sys.argv[1:] or ["5gad", "ustc", "iscx", "tii"]): run(ds)
