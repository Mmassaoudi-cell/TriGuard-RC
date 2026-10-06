"""Matched baselines on the duplicate-aware manifests.  All tuning is validation-only; test is read once.

Models: LightGBM, XGBoost, ExtraTrees, kNN, LightGBM+Aug (trained on corrupted copies, i.e. the same augmentation
budget as TriGuard-RC), LightGBM+kNN (retrieval control), FT-Transformer.  Every probabilistic model is also reported
after validation temperature scaling.  Probabilities are stored for the operating-point analysis."""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, torch
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
import rc, evalx, ftt
from calib import macro, fit_T_prob, temper_prob, pick
from dev import DS

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results_rg"; OUT.mkdir(exist_ok=True)
CORR = dict(byte_sub=0.10, jitter=0.5, noise=0.05, aug_frac=1.0)


def early_from_tensors(img, tim, sta):
    B = img.shape[0]
    return torch.cat([sta, tim.reshape(B, -1), img.reshape(B, 10, 4, 10, 4).mean((2, 4)).reshape(B, -1)], 1)


def early_all(g, ids, corrupt=False, seed=0):
    torch.manual_seed(seed); out = []
    for i in range(0, len(ids), 8192):
        img, tim, sta, _, _ = g.batch(ids[i:i + 8192])
        if corrupt: img, tim, sta = rc.augment(img, tim, sta, CORR)
        out.append(early_from_tensors(img, tim, sta).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def P(m, x, C):
    o = np.full((len(x), C), 1e-9, np.float32); o[:, m.classes_.astype(int)] = m.predict_proba(x); return o / o.sum(1, keepdims=True)


def knn_gpu(xtr, ytr, xq, C, k):
    xt = torch.tensor(xtr, device="cuda"); xqq = torch.tensor(xq, device="cuda"); yt = torch.tensor(ytr, device="cuda"); out = []
    for i in range(0, len(xqq), 1024):
        dist = torch.cdist(xqq[i:i + 1024], xt); v, ix = dist.topk(k, largest=False)
        w = 1.0 / (v + 1e-6); oh = torch.nn.functional.one_hot(yt[ix], C).float()
        p = (w.unsqueeze(-1) * oh).sum(1); out.append(p / p.sum(1, keepdim=True))
    return torch.cat(out).cpu().numpy()


def lgbm(seed, X, y, Xv, yv, w=None):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           class_weight="balanced" if w is None else None, random_state=seed, verbosity=-1, n_jobs=12)
    m.fit(X, y, sample_weight=w, eval_set=[(Xv, yv)], callbacks=[lgb.early_stopping(30, verbose=False)]); return m


def run(name, seed, robust):
    fo = OUT / f"base_{name}_s{seed}.json"
    if fo.exists(): return
    d = rc.load(str(DS[name])); g = rc.GPUData(d); C = g.C; cl = d["classes"]
    tr, va, te = (g.idx[k] for k in ("train", "val", "test"))
    y = d["labels"]; ytr, yva, yte = y[d["split"]["train"]], y[d["split"]["val"]], y[d["split"]["test"]]
    Xtr, Xva, Xte = early_all(g, tr), early_all(g, va), early_all(g, te)
    res, probs = {}, {}
    def record(mn, pv, pt, extra=None):
        Tm = fit_T_prob(pv, yva, C)
        res[mn] = dict(val_macro_f1=macro(yva, pv.argmax(1), C), T=Tm, test=evalx.full_metrics(yte, pt, cl), test_cal=evalx.full_metrics(yte, temper_prob(pt, Tm), cl), **(extra or {}))
        probs[mn] = pt.astype(np.float16); probs[mn + "__val"] = pv.astype(np.float16)
    t0 = time.time(); models = {}
    m = lgbm(seed, Xtr, ytr, Xva, yva); models["LightGBM"] = m; pl_v, pl_t = P(m, Xva, C), P(m, Xte, C); record("LightGBM", pl_v, pl_t)
    cw = (len(ytr) / (C * np.bincount(ytr, minlength=C).clip(1))) ** 0.5
    xm = xgb.XGBClassifier(n_estimators=500, learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.6, tree_method="hist", device="cuda",
                           random_state=seed, early_stopping_rounds=30, verbosity=0)
    xm.fit(Xtr, ytr, sample_weight=cw[ytr], eval_set=[(Xva, yva)], verbose=False); record("XGBoost", P(xm, Xva, C), P(xm, Xte, C))
    et = ExtraTreesClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample", n_jobs=12, random_state=seed).fit(Xtr, ytr)
    record("ExtraTrees", P(et, Xva, C), P(et, Xte, C))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Z = lambda a: ((a - mu) / sd).astype(np.float32)
    best = None
    for k in (3, 5, 10):
        pv = knn_gpu(Z(Xtr), ytr, Z(Xva), C, k); f = macro(yva, pv.argmax(1), C)
        if best is None or f > best[0]: best = (f, k)
    kk = best[1]; kv, kt = knn_gpu(Z(Xtr), ytr, Z(Xva), C, kk), knn_gpu(Z(Xtr), ytr, Z(Xte), C, kk); record("kNN", kv, kt, dict(k=kk))
    # retrieval control: LightGBM posterior blended with feature-space kNN (weight chosen on validation)
    lam = pick(yva, C, np.linspace(0, 1, 11), lambda l: (1 - l) * pl_v + l * kv)
    record("LightGBM+kNN", (1 - lam) * pl_v + lam * kv, (1 - lam) * pl_t + lam * kt, dict(lam=lam))
    # matched augmentation budget: original rows + two corrupted copies
    Xa = np.concatenate([Xtr, early_all(g, tr, True, seed * 10 + 1), early_all(g, tr, True, seed * 10 + 2)]); ya = np.concatenate([ytr, ytr, ytr])
    cw2 = (len(ya) / (C * np.bincount(ya, minlength=C).clip(1))); ma = lgbm(seed, Xa, ya, Xva, yva, w=cw2[ya]); models["LightGBM+Aug"] = ma
    record("LightGBM+Aug", P(ma, Xva, C), P(ma, Xte, C))
    lg, fbest = ftt.train_ftt(Xtr, ytr, Xva, yva, C, seed)
    from calib import softmax_T
    fv, ft = softmax_T(lg(Xva), 1.0), softmax_T(lg(Xte), 1.0); record("FT-Transformer", fv, ft)
    out = dict(ds=name, seed=seed, models=res, seconds=time.time() - t0)
    if robust:
        med = np.median(Xtr, 0); ns = d["statistics"].shape[1]
        sl = {"b": slice(ns + 25, Xtr.shape[1]), "t": slice(ns, ns + 25), "s": slice(0, ns)}
        def wrap(pred):
            def f(img, tim, sta, mk):
                x = early_from_tensors(img, tim, sta).cpu().numpy()
                for j, kx in enumerate("bts"):
                    dead = ~mk[:, j].cpu().numpy()
                    if dead.any(): x[dead, sl[kx]] = med[sl[kx]]
                return torch.from_numpy(pred(x))
            return f
        out["robust"] = {"LightGBM": evalx.robustness(wrap(lambda x: P(models["LightGBM"], x, C)), g, cl),
                         "LightGBM+Aug": evalx.robustness(wrap(lambda x: P(models["LightGBM+Aug"], x, C)), g, cl),
                         "FT-Transformer": evalx.robustness(wrap(lambda x: softmax_T(lg(x), 1.0)), g, cl)}
    np.savez_compressed(OUT / f"probs_base_{name}_s{seed}.npz", y=yte, yv=yva, **probs)
    fo.write_text(json.dumps(out, indent=1))
    print(name, seed, {k: round(v["test"]["macro_f1"], 4) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds"); ap.add_argument("--seeds", default="1,2,3"); ap.add_argument("--robust", action="store_true"); a = ap.parse_args()
    for s in map(int, a.seeds.split(",")): run(a.ds, s, a.robust)
