"""Validation-only development harness (never reads test)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
import rc

ART = Path(r"c:/Users/MMASSAOUDI/Desktop/research work/Alienware/Enhanced TriGuard/LitCVit_3datasets/artifacts")
RG = Path(__file__).parent / "manifests_rg"
DS_OLD = {"5gad": ART / "5gad_full/5gad_multiview.npz", "ustc": ART / "ustc_fast/ustc_multiview.npz", "tii": ART / "tii_stratified_sample/tii_sample_multiview.npz"}
DS = {"5gad": RG / "5gad/5gad_multiview.npz", "ustc": RG / "ustc/ustc_multiview.npz", "tii": RG / "tii/tii_sample_multiview.npz", "iscx": RG / "iscx/iscx_multiview.npz",
      "ustc_s43": RG / "ustc_s43/ustc_multiview.npz", "ustc_s44": RG / "ustc_s44/ustc_multiview.npz", "tii_s43": RG / "tii_s43/tii_sample_multiview.npz", "tii_s44": RG / "tii_s44/tii_sample_multiview.npz",
      "iscx_s43": RG / "iscx_s43/iscx_multiview.npz", "iscx_s44": RG / "iscx_s44/iscx_multiview.npz"}
CACHE = Path(__file__).parent / "teacher_cache_rg"; CACHE.mkdir(exist_ok=True)


def teacher_for(name, d, seed, balanced=False):
    f = CACHE / f"{name}_s{seed}{'_bal' if balanced else ''}.npy"
    if f.exists(): return np.load(f)
    t0 = time.time(); oof, _, _ = rc.crossfit_teacher(d, seed, balanced=balanced); np.save(f, oof)
    print(f"teacher {name} s{seed} {time.time()-t0:.0f}s", flush=True); return oof


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cfg", default="{}")
    a = ap.parse_args()
    d = rc.load(str(DS[a.ds])); g = rc.GPUData(d)
    cfg = json.loads(a.cfg)
    teacher = teacher_for(a.ds, d, a.seed, cfg.pop("_bal", False)) if cfg.get("kd", rc.DEFAULT["kd"]) > 0 else None
    m, info = rc.train(d, g, a.seed, cfg, teacher)
    va = g.idx["val"]; yv = g.y[va].cpu().numpy()
    pp, pk, _ = rc.predict_all(m, g, "val", info["cfg"])
    lam = rc.fit_blend(pp, pk, yv, g.C)
    fb = rc.macro(yv, ((1 - lam) * pp + lam * pk).argmax(1).cpu().numpy(), g.C)
    print(json.dumps(dict(ds=a.ds, seed=a.seed, val_param=info["best_val_macro_f1"], lam=lam, val_blend=fb,
                          epochs=info["best_epoch"], params=info["params"], secs=info["train_seconds"])))
