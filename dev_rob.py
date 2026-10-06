"""Validation-only robustness/clean trade-off check used to pick augmentation strengths."""
import sys, json, numpy as np, rc, evalx
from dev import DS, teacher_for
ds, seed, cfg = sys.argv[1], int(sys.argv[2]), json.loads(sys.argv[3])
d = rc.load(str(DS[ds])); g = rc.GPUData(d)
m, info = rc.train(d, g, seed, cfg, teacher_for(ds, d, seed), log=False)
r = evalx.robustness(lambda i, t, s, k: m(i, t, s, k)["logits"], g, d["classes"], part="val")
print(json.dumps(dict(cfg=cfg, val_f1=info["best_val_macro_f1"], **{k: round(v, 3) for k, v in r.items()})))
