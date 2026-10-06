"""Original TriGuard-ETD (released training code) on the duplicate-aware manifests, plus matched controls:
temperature-scaled ETD and ETD + retrieval memory (same k-NN readout as TriGuard-RC, validation-selected weight)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
import evalx, rc
sys.path.insert(0, str(evalx.ETD))
from scripts.run_enhanced_triguard import train_enhanced
from triguard.runner import load_cache
from calib import macro, fit_T_logits, softmax_T, fit_T_prob, temper_prob, pick, knn_from_embeddings
from dev import DS

PROFILE = {"5gad": "legacy_loss", "ustc": "balanced_kd", "tii": "balanced_kd", "iscx": "balanced_kd"}
OUT = Path(__file__).parent / "results_rg"; OUT.mkdir(exist_ok=True)


@torch.no_grad()
def run_model(model, g, ids, bs=4096):
    model.eval(); lg, em = [], []
    for i in range(0, len(ids), bs):
        img, tim, sta, mk, _ = g.batch(ids[i:i + bs]); r = model(img, tim, sta, mk); lg.append(r["logits"]); em.append(r["embedding"])
    return torch.cat(lg), torch.cat(em)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds"); ap.add_argument("--seeds", default="1,2,3,4,5"); ap.add_argument("--robust", action="store_true")
    ap.add_argument("--epochs", type=int, default=100); a = ap.parse_args()
    data, meta = load_cache(str(DS[a.ds])); d = rc.load(str(DS[a.ds])); g = rc.GPUData(d); C = g.C; cl = d["classes"]
    yv = g.y[g.idx["val"]].cpu().numpy(); yt = g.y[g.idx["test"]].cpu().numpy()
    for seed in map(int, a.seeds.split(",")):
        f = OUT / f"orig_{a.ds}_s{seed}.json"
        if f.exists(): continue
        t0 = time.time()
        model, tr = train_enhanced(data, seed, PROFILE[a.ds.split("_")[0]], a.epochs, 256, 12, 12, "cuda")
        zv, ev = run_model(model, g, g.idx["val"]); zt, et = run_model(model, g, g.idx["test"]); _, etr = run_model(model, g, g.idx["train"])
        zv, zt = zv.cpu().numpy(), zt.cpu().numpy()
        pv, pt = softmax_T(zv, 1.0), softmax_T(zt, 1.0); T = fit_T_logits(zv, yv, C)
        kv = knn_from_embeddings(etr, g.y[g.idx["train"]], ev, C); kt = knn_from_embeddings(etr, g.y[g.idx["train"]], et, C)
        cv, ct = softmax_T(zv, T), softmax_T(zt, T)
        lam = pick(yv, C, np.linspace(0, 1, 11), lambda l: (1 - l) * cv + l * kv)
        bv, bt = (1 - lam) * cv + lam * kv, (1 - lam) * ct + lam * kt; Tb = fit_T_prob(bv, yv, C)
        res = dict(ds=a.ds, seed=seed, params=tr["parameters"], epochs=tr["epochs_run"], train_seconds=time.time() - t0, T=T, lam=lam,
                   test=evalx.full_metrics(yt, pt, cl), test_cal=evalx.full_metrics(yt, ct, cl),
                   test_ret=evalx.full_metrics(yt, temper_prob(bt, Tb), cl), val_macro_f1=macro(yv, pv.argmax(1), C))
        np.savez_compressed(OUT / f"probs_orig_{a.ds}_s{seed}.npz", y=yt, raw=pt.astype(np.float16), cal=ct.astype(np.float16), ret=temper_prob(bt, Tb).astype(np.float16))
        if a.robust:
            res["robust"] = evalx.robustness(lambda i, t, s, m: model(i, t, s, m)["logits"], g, cl)
        f.write_text(json.dumps(res, indent=1)); print(a.ds, seed, round(res["test"]["macro_f1"], 4), round(res["test_ret"]["macro_f1"], 4), flush=True)
