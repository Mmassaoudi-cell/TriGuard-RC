"""Duplicate-aware re-split: rows sharing a provenance group OR an identical tabular signature (1e-5) are
kept together, so no exact/near-exact duplicate crosses a partition boundary.  Robust scaling is refitted on the new
training rows (an affine re-map of the cached scaled statistics, identical to refitting on raw values)."""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np
from sklearn.preprocessing import RobustScaler
sys.path.insert(0, r"c:/Users/MMASSAOUDI/Desktop/research work/Alienware/Enhanced TriGuard/TriGuard-ETD")
from triguard.data import create_split_bundle
from make_dedup import SRC, ART

OUT = Path(__file__).parent / "manifests_rg"


def union_find_groups(prov, sig):
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for p, s in zip(prov, sig):
        a, b = find("p:" + p), find("s:" + s)
        if a != b: parent[a] = b
    return np.array([find("p:" + p) for p in prov])


def regroup(name, folder, prefix, subset_from_split=True, seed=42, val_ratio=0.10, test_ratio=0.20, out_name=None):
    z = np.load(folder / f"{prefix}_multiview.npz", allow_pickle=True)
    sp = np.load(folder / f"{prefix}_split.npz", allow_pickle=True)
    d = {k: z[k] for k in z.files}
    rows = np.sort(np.concatenate([sp["train"], sp["val"], sp["test"]])) if subset_from_split else np.arange(len(d["labels"]))
    y = d["labels"][rows]; classes = [str(c) for c in d["classes"]]
    tab = np.round(np.concatenate([d["statistics"][rows], d["timing"][rows].reshape(len(rows), -1)], 1).astype(np.float64), 5)
    sig = [hashlib.blake2b(np.ascontiguousarray(r).tobytes() + int(l).to_bytes(2, "little"), digest_size=10).hexdigest() for r, l in zip(tab, y)]
    groups = union_find_groups([str(g) for g in d["groups"][rows]], sig)
    bundle = create_split_bundle(y, groups, tab, classes, seed=seed, val_ratio=val_ratio, test_ratio=test_ratio)
    tr, va, te = rows[bundle.train], rows[bundle.val], rows[bundle.test]
    # refit robust scaling on the new training rows only
    st = d["statistics"].astype(np.float32)
    sc = RobustScaler(quantile_range=(5, 95)).fit(st[tr])
    d["statistics"] = sc.transform(st).astype(np.float32)
    dest = OUT / (out_name or name); dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f"{prefix}_multiview.npz", **d)
    shutil.copy(folder / f"{prefix}_preprocessing.json", dest / f"{prefix}_preprocessing.json")
    np.savez_compressed(dest / f"{prefix}_split.npz", train=tr, val=va, test=te, classes=z["classes"])
    rep = {"dataset": name, "rows": int(len(rows)), "n_groups": int(len(set(groups))),
           "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
           "train_only_classes": bundle.train_only_classes, "duplicate_overlap_after": bundle.duplicate_overlap,
           "test_support": np.bincount(d["labels"][te], minlength=len(classes)).tolist(),
           "val_support": np.bincount(d["labels"][va], minlength=len(classes)).tolist(),
           "train_support": np.bincount(d["labels"][tr], minlength=len(classes)).tolist(), "classes": classes}
    (dest / "regroup_report.json").write_text(json.dumps(rep, indent=1))
    print(name, rep["sizes"], "groups", rep["n_groups"], "train_only", list((bundle.train_only_classes or {}).keys()), "test support", rep["test_support"], flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["main"]
    if which == ["main"]:
        for k, (folder, prefix) in SRC.items(): regroup(k, folder, prefix)
    else:  # extra split seeds:  python make_regroup.py 43 44
        for sd in map(int, which):
            for k in ("ustc", "tii"):
                folder, prefix = SRC[k]; regroup(k, folder, prefix, seed=sd, out_name=f"{k}_s{sd}")
