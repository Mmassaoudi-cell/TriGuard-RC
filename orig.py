"""Re-run the ORIGINAL TriGuard-ETD (their enhanced student + their training code) on the same frozen splits/seeds."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
import evalx, rc
sys.path.insert(0, str(evalx.ETD))
from scripts.run_enhanced_triguard import train_enhanced  # original training code
from triguard.runner import load_cache, make_loader, predict_torch
from dev import DS

PROFILE = {"5gad": "legacy_loss", "ustc": "balanced_kd", "tii": "balanced_kd"}
OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds"); ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--robust", action="store_true"); ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    data, meta = load_cache(str(DS[a.ds]))
    d = rc.load(str(DS[a.ds])); g = rc.GPUData(d)
    for seed in map(int, a.seeds.split(",")):
        f = OUT / f"orig_{a.ds}_s{seed}.json"
        if f.exists(): continue
        t0 = time.time()
        model, tr = train_enhanced(data, seed, PROFILE[a.ds], a.epochs, 256, 12, 12, "cuda")
        loader = make_loader(data, "test", 512, False)
        y, pred, prob, gates = predict_torch(model, loader, "cuda", "triguard")
        res = dict(ds=a.ds, seed=seed, model="TriGuard-ETD(orig)", params=tr["parameters"], epochs=tr["epochs_run"],
                   train_seconds=time.time() - t0, test=evalx.full_metrics(y, prob, data["classes"]),
                   gates=gates.mean(0).tolist())
        if a.robust:
            model.eval()
            res["robust"] = evalx.robustness(lambda i, t, s, m: model(i, t, s, m)["logits"], g, data["classes"])
        f.write_text(json.dumps(res, indent=1)); print(a.ds, seed, res["test"]["macro_f1"], flush=True)
