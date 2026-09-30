"""Final protocol: train on train; select epoch, temperature, retrieval weight (lambda) and GBDT weight (mu)
on validation; open the test partition once per run."""
import argparse, json, time
from pathlib import Path
import numpy as np, torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
import lightgbm as lgb
import rc, evalx, baselines
from dev import DS, CACHE

OUT = Path(__file__).parent / "results"; OUT.mkdir(exist_ok=True)
CFG = {"5gad": {}, "tii": {}, "ustc": {}}
VARIANTS = {
    "full": {}, "no_kd": {"kd": 0.0}, "insample_kd": {"_teacher": "insample"},
    "no_viewdrop": {"view_drop": 0.0, "cons": 0.0}, "no_aug": {"byte_sub": 0.0, "jitter": 0.0, "noise": 0.0},
    "no_ple": {"T": 1}, "byte_only": {"_views": [1, 0, 0]}, "timing_only": {"_views": [0, 1, 0]},
    "stats_only": {"_views": [0, 0, 1]}, "timing_stats": {"_views": [0, 1, 1]},
}


def teacher(name, d, seed, mode):
    f = CACHE / f"{name}_s{seed}.npy"; fi = CACHE / f"{name}_s{seed}_insample.npy"
    if not f.exists() or not fi.exists():
        oof, full, x = rc.crossfit_teacher(d, seed); np.save(f, oof)
        np.save(fi, rc._probs(full, x[d["split"]["train"]], len(d["classes"])))
    return np.load(fi if mode == "insample" else f)


def temper_logits(z, T): return torch.softmax(z / T, 1)


def fit_T_logits(z, y, C):
    f = lambda lt: log_loss(y, np.clip(torch.softmax(z / float(np.exp(lt)), 1).cpu().numpy(), 1e-9, 1), labels=np.arange(C))
    return float(np.exp(minimize_scalar(f, bounds=(np.log(0.05), np.log(5.0)), method="bounded").x))


def temper_prob(p, T):
    q = np.clip(p, 1e-9, 1) ** (1.0 / T); return q / q.sum(1, keepdims=True)


def fit_T_prob(p, y, C):
    f = lambda lt: log_loss(y, temper_prob(p, float(np.exp(lt))), labels=np.arange(C))
    return float(np.exp(minimize_scalar(f, bounds=(np.log(0.05), np.log(5.0)), method="bounded").x))


def gbdt_fit(name, d, seed):
    """Balanced LightGBM expert on early-fused features (train fit, validation early stopping)."""
    X = baselines.feats(d, "early"); y = d["labels"]; sp = d["split"]
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           class_weight="balanced", random_state=seed, verbosity=-1, n_jobs=16)
    m.fit(X[sp["train"]], y[sp["train"]], eval_set=[(X[sp["val"]], y[sp["val"]])], callbacks=[lgb.early_stopping(30, verbose=False)])
    return m, X


def pick(pv, yv, C, cands, comb):
    best = None
    for c in cands:
        p = comb(c)
        key = (round(rc.macro(yv, p.argmax(1), C), 4), -float(log_loss(yv, np.clip(p, 1e-9, 1), labels=np.arange(C))))
        if best is None or key > best[0]: best = (key, float(c))
    return best[1]


def run(name, seed, variant, robust):
    fo = OUT / f"rc_{name}_{variant}_s{seed}.json"
    if fo.exists(): return
    v = dict(VARIANTS[variant]); mode = v.pop("_teacher", "oof"); views = v.pop("_views", None)
    cfg = {**CFG[name], **v}
    d = rc.load(str(DS[name])); g = rc.GPUData(d)
    if views is not None: g.mask = g.mask & torch.tensor(views, dtype=torch.bool, device=g.mask.device)[None]
    t = teacher(name, d, seed, mode) if cfg.get("kd", rc.DEFAULT["kd"]) > 0 else None
    m, info = rc.train(d, g, seed, cfg, t, log=False)
    C = g.C; cl = d["classes"]
    yv = g.y[g.idx["val"]].cpu().numpy(); yt = g.y[g.idx["test"]].cpu().numpy()
    zv = rc.infer(m, g, g.idx["val"])["logits"]; zt = rc.infer(m, g, g.idx["test"])
    Tp = fit_T_logits(zv, yv, C)
    pv = temper_logits(zv, Tp).cpu().numpy(); pt = temper_logits(zt["logits"], Tp).cpu().numpy()
    kv = rc.knn_probs(m, g, g.idx["val"], info["cfg"]).cpu().numpy(); kt = rc.knn_probs(m, g, g.idx["test"], info["cfg"]).cpu().numpy()
    lam = pick(pv, yv, C, np.linspace(0, 1, 11), lambda l: (1 - l) * pv + l * kv)
    bv, bt = (1 - lam) * pv + lam * kv, (1 - lam) * pt + lam * kt
    Tb = fit_T_prob(bv, yv, C); bv_c, bt_c = temper_prob(bv, Tb), temper_prob(bt, Tb)
    res = dict(ds=name, seed=seed, variant=variant, lam=lam, T_head=Tp, T_blend=Tb, params=info["params"], best_epoch=info["best_epoch"],
               epochs_run=info["epochs_run"], train_seconds=info["train_seconds"], val_macro_f1=info["best_val_macro_f1"],
               test_param=evalx.full_metrics(yt, pt, cl), test_knn=evalx.full_metrics(yt, kt, cl),
               test=evalx.full_metrics(yt, bt_c, cl), gates=zt["gates"].mean(0).tolist(), cfg=info["cfg"])
    gm = None
    if variant == "full":  # neural-tree hybrid: validation-selected blend with the GBDT expert
        gm, X = gbdt_fit(name, d, seed); sp = d["split"]
        gv, gt = baselines.P(gm, X[sp["val"]], C), baselines.P(gm, X[sp["test"]], C)
        mu = pick(bv_c, yv, C, np.linspace(0, 1, 11), lambda u: (1 - u) * bv_c + u * gv)
        hv, ht = (1 - mu) * bv_c + mu * gv, (1 - mu) * bt_c + mu * gt
        Th = fit_T_prob(hv, yv, C)
        res["mu"] = mu; res["T_hybrid"] = Th
        res["test_hybrid"] = evalx.full_metrics(yt, temper_prob(ht, Th), cl)
        res["test_gbdt"] = evalx.full_metrics(yt, gt, cl)
        np.savez_compressed(OUT / f"probs_{name}_s{seed}.npz", y=yt, param=pt.astype(np.float16), blend=bt_c.astype(np.float16),
                            gbdt=gt.astype(np.float16), hybrid=temper_prob(ht, Th).astype(np.float16), gates=zt["gates"].cpu().numpy().astype(np.float16))
    if robust:
        # logit-level robustness of the neural head (temperature does not change argmax)
        res["robust"] = evalx.robustness(lambda i, t_, s, mk: m(i, t_, s, mk)["logits"], g, cl)
        if gm is not None:  # boosted-tree robustness: features recomputed from perturbed views, missing views median-imputed
            med = np.median(X[d["split"]["train"]], 0)
            slices = {"b": slice(len(d["statistics"][0]) + 25, X.shape[1]), "t": slice(len(d["statistics"][0]), len(d["statistics"][0]) + 25), "s": slice(0, len(d["statistics"][0]))}
            def gb_pred(img, tim, sta, mk):
                B = img.shape[0]
                pooled = img.reshape(B, 10, 4, 10, 4).mean((2, 4)).reshape(B, -1)
                x = torch.cat([sta, tim.reshape(B, -1), pooled], 1).cpu().numpy()
                for j, key in enumerate("bts"):
                    dead = ~mk[:, j].cpu().numpy()
                    if dead.any(): x[dead, slices[key]] = med[slices[key]]
                return torch.from_numpy(baselines.P(gm, x, C))
            res["robust_gbdt"] = evalx.robustness(gb_pred, g, cl)
    fo.write_text(json.dumps(res, indent=1)); print(name, variant, seed, round(res["test"]["macro_f1"], 4), round(res["test_param"]["macro_f1"], 4),
                                                    "ece", round(res["test"]["ece"], 4), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds"); ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--variants", default="full"); ap.add_argument("--robust", action="store_true")
    a = ap.parse_args()
    for s in map(int, a.seeds.split(",")):
        for v in a.variants.split(","): run(a.ds, s, v, a.robust)
