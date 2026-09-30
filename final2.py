"""Final protocol on the duplicate-aware manifests: train; select epoch, temperature, retrieval weight (lambda) and tree
weight (mu) on validation; open the test partition once per run.  Probabilities of every stage are stored."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
import rc, evalx
from calib import macro, fit_T_logits, softmax_T, fit_T_prob, temper_prob, pick
from dev import DS, CACHE
from baselines2 import early_all, lgbm, P

OUT = Path(__file__).parent / "results_rg"; OUT.mkdir(exist_ok=True)
VARIANTS = {
    "full": {}, "no_ret": {}, "no_kd": {"kd": 0.0}, "insample_kd": {"_teacher": "insample"},
    "no_viewdrop": {"view_drop": 0.0, "view_drop_vec": None, "cons": 0.0}, "no_aug": {"byte_sub": 0.0, "jitter": 0.0, "noise": 0.0},
    "full_aug": {"aug_frac": 1.0}, "no_ple": {"T": 1},
    "byte_only": {"_views": [1, 0, 0]}, "timing_only": {"_views": [0, 1, 0]}, "stats_only": {"_views": [0, 0, 1]}, "timing_stats": {"_views": [0, 1, 1]},
}


def teacher(name, d, seed, mode):
    f = CACHE / f"{name}_s{seed}.npy"; fi = CACHE / f"{name}_s{seed}_insample.npy"
    if not f.exists() or not fi.exists():
        oof, full, x = rc.crossfit_teacher(d, seed); np.save(f, oof)
        np.save(fi, rc._probs(full, x[d["split"]["train"]], len(d["classes"])))
    return np.load(fi if mode == "insample" else f)


def run(name, seed, variant, robust, tag=""):
    fo = OUT / f"rc_{name}_{variant}{tag}_s{seed}.json"
    if fo.exists(): return
    v = dict(VARIANTS[variant]); mode = v.pop("_teacher", "oof"); views = v.pop("_views", None)
    d = rc.load(str(DS[name])); g = rc.GPUData(d)
    if views is not None: g.mask = g.mask & torch.tensor(views, dtype=torch.bool, device=g.mask.device)[None]
    t = teacher(name, d, seed, mode) if v.get("kd", rc.DEFAULT["kd"]) > 0 else None
    m, info = rc.train(d, g, seed, v, t, log=False)
    C = g.C; cl = d["classes"]
    yv = g.y[g.idx["val"]].cpu().numpy(); yt = g.y[g.idx["test"]].cpu().numpy()
    rv = rc.infer(m, g, g.idx["val"]); rt = rc.infer(m, g, g.idx["test"])
    zv, zt = rv["logits"].cpu().numpy(), rt["logits"].cpu().numpy()
    Tp = fit_T_logits(zv, yv, C); pv, pt = softmax_T(zv, Tp), softmax_T(zt, Tp)
    kv = rc.knn_probs(m, g, g.idx["val"], info["cfg"]).cpu().numpy(); kt = rc.knn_probs(m, g, g.idx["test"], info["cfg"]).cpu().numpy()
    lam = pick(yv, C, np.linspace(0, 1, 11), lambda l: (1 - l) * pv + l * kv)
    bv, bt = (1 - lam) * pv + lam * kv, (1 - lam) * pt + lam * kt
    Tb = fit_T_prob(bv, yv, C); bv_c, bt_c = temper_prob(bv, Tb), temper_prob(bt, Tb)
    res = dict(ds=name, seed=seed, variant=variant, lam=lam, T_head=Tp, T_blend=Tb, params=info["params"], best_epoch=info["best_epoch"],
               epochs_run=info["epochs_run"], train_seconds=info["train_seconds"], val_macro_f1=info["best_val_macro_f1"],
               test_param=evalx.full_metrics(yt, pt, cl), test_knn=evalx.full_metrics(yt, kt, cl), test=evalx.full_metrics(yt, bt_c, cl),
               gates=rt["gates"].mean(0).tolist(), cfg=info["cfg"])
    probs = dict(y=yt, param=pt.astype(np.float16), blend=bt_c.astype(np.float16), yv=yv, blend_val=bv_c.astype(np.float16))
    if variant == "full":  # neural-tree hybrid
        tr_ids, va_ids, te_ids = (g.idx[k] for k in ("train", "val", "test"))
        Xtr, Xva, Xte = early_all(g, tr_ids), early_all(g, va_ids), early_all(g, te_ids); ytr = d["labels"][d["split"]["train"]]
        gm = lgbm(seed, Xtr, ytr, Xva, yv); gv, gt = P(gm, Xva, C), P(gm, Xte, C)
        mu = pick(yv, C, np.linspace(0, 1, 11), lambda u: (1 - u) * bv_c + u * gv)
        hv, ht = (1 - mu) * bv_c + mu * gv, (1 - mu) * bt_c + mu * gt; Th = fit_T_prob(hv, yv, C)
        res["mu"] = mu; res["T_hybrid"] = Th; res["test_hybrid"] = evalx.full_metrics(yt, temper_prob(ht, Th), cl)
        probs.update(hybrid=temper_prob(ht, Th).astype(np.float16), gbdt=gt.astype(np.float16))
    np.savez_compressed(OUT / f"probs_rc_{name}_{variant}{tag}_s{seed}.npz", **probs)
    if robust:
        res["robust"] = evalx.robustness(lambda i, t_, s, mk: m(i, t_, s, mk)["logits"], g, cl)
    fo.write_text(json.dumps(res, indent=1)); print(name, variant, seed, round(res["test"]["macro_f1"], 4), round(res["test_param"]["macro_f1"], 4), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds"); ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--variants", default="full"); ap.add_argument("--robust", action="store_true"); a = ap.parse_args()
    for s in map(int, a.seeds.split(",")):
        for v in a.variants.split(","): run(a.ds, s, v, a.robust)
