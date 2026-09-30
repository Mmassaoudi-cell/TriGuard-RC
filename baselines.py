"""Strong tabular / non-parametric baselines on the identical frozen splits (early-fused features).
Hyper-parameters are selected on validation macro-F1 only from a small fixed grid."""
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, torch
import lightgbm as lgb, xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
import rc, evalx
from dev import DS
warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"


def feats(d, kind):
    s, t = d["statistics"], d["timing"].reshape(len(d["timing"]), -1)
    if kind == "stats": return s
    img = d["image"].astype(np.float32).reshape(-1, 10, 4, 10, 4).mean((2, 4)).reshape(len(s), -1) / 255.0  # 10x10 pooled bytes
    return np.concatenate([s, t, img], 1).astype(np.float32)


def P(m, x, C):
    o = np.full((len(x), C), 1e-9, np.float32); o[:, m.classes_.astype(int)] = m.predict_proba(x); return o / o.sum(1, keepdims=True)


def knn_gpu(xtr, ytr, xq, C, k):
    xt = torch.tensor(xtr, device="cuda"); xqq = torch.tensor(xq, device="cuda"); yt = torch.tensor(ytr, device="cuda"); out = []
    for i in range(0, len(xqq), 1024):
        dist = torch.cdist(xqq[i:i + 1024], xt); v, ix = dist.topk(k, largest=False)
        w = 1.0 / (v + 1e-6); oh = torch.nn.functional.one_hot(yt[ix], C).float()
        p = (w.unsqueeze(-1) * oh).sum(1); out.append(p / p.sum(1, keepdim=True))
    return torch.cat(out).cpu().numpy()


def run(name, seed):
    d = rc.load(str(DS[name])); C = len(d["classes"]); y = d["labels"]; sp = d["split"]; cl = d["classes"]
    X = feats(d, "early"); S = feats(d, "stats")
    tr, va, te = sp["train"], sp["val"], sp["test"]
    res = {}
    def record(mname, pv, pt, extra=None):
        res[mname] = dict(val_macro_f1=float(f1_score(y[va], pv.argmax(1), labels=np.arange(C), average="macro", zero_division=0)),
                          test=evalx.full_metrics(y[te], pt, cl), **(extra or {}))
    t0 = time.time()
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1, colsample_bytree=0.6,
                           class_weight="balanced", random_state=seed, verbosity=-1, n_jobs=8)
    m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(30, verbose=False)])
    record("LightGBM", P(m, X[va], C), P(m, X[te], C))
    m = xgb.XGBClassifier(n_estimators=500, learning_rate=0.08, max_depth=8, subsample=0.8, colsample_bytree=0.6, tree_method="hist",
                          device="cuda", random_state=seed, early_stopping_rounds=30, verbosity=0)
    cw = (len(tr) / (C * np.bincount(y[tr], minlength=C).clip(1))) ** 0.5
    m.fit(X[tr], y[tr], sample_weight=cw[y[tr]], eval_set=[(X[va], y[va])], verbose=False)
    record("XGBoost", P(m, X[va], C), P(m, X[te], C))
    m = ExtraTreesClassifier(n_estimators=250, max_features="sqrt", class_weight="balanced_subsample", n_jobs=8, random_state=seed)
    m.fit(X[tr], y[tr]); record("ExtraTrees", P(m, X[va], C), P(m, X[te], C))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Z = (X - mu) / sd
    best = None
    for k in (3, 5, 10):
        pv = knn_gpu(Z[tr], y[tr], Z[va], C, k); f = f1_score(y[va], pv.argmax(1), labels=np.arange(C), average="macro", zero_division=0)
        if best is None or f > best[0]: best = (f, k)
    k = best[1]; record("kNN", knn_gpu(Z[tr], y[tr], Z[va], C, k), knn_gpu(Z[tr], y[tr], Z[te], C, k), dict(k=k))
    (OUT / f"base_{name}_s{seed}.json").write_text(json.dumps(dict(ds=name, seed=seed, models=res, seconds=time.time() - t0), indent=1))
    print(name, seed, {k: round(v["test"]["macro_f1"], 4) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ds"); ap.add_argument("--seeds", default="1,2,3"); a = ap.parse_args()
    for s in map(int, a.seeds.split(",")): run(a.ds, s)
