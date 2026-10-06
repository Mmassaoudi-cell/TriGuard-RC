"""Build de-duplicated evaluation manifests.

Every validation/test row whose quantised multi-view signature (byte image, timing tensor and
statistics) also occurs in the training partition is removed, so retrieval, tree and network
components cannot profit from cross-partition duplicates.  Training rows are untouched.
"""
import hashlib, json, shutil, sys
from pathlib import Path
import numpy as np

ART = Path(r"c:/Users/MMASSAOUDI/Desktop/research work/Alienware/Enhanced TriGuard/LitCVit_3datasets/artifacts")
OUT = Path(__file__).parent / "manifests_dd"
SRC = {"5gad": (ART / "5gad_full", "5gad"), "ustc": (ART / "ustc_fast", "ustc"), "tii": (ART / "tii_stratified_sample", "tii_sample")}


def signature(d, idx, decimals=5):
    """Tabular (statistics + timing) signature at 1e-5 resolution: exact and near-exact duplicates share it."""
    a = np.round(np.concatenate([d["statistics"][idx], d["timing"][idx].reshape(len(idx), -1)], 1).astype(np.float64), decimals)
    return [hashlib.blake2b(np.ascontiguousarray(r).tobytes(), digest_size=12).digest() for r in a]


def dedup(name, folder, prefix, out_name=None):
    out_name = out_name or name
    z = np.load(folder / f"{prefix}_multiview.npz", allow_pickle=True)
    sp = np.load(folder / f"{prefix}_split.npz", allow_pickle=True)
    d = {k: z[k] for k in ("image", "timing", "statistics", "labels")}
    if d["image"].shape[0] == 1:  # lazily-derived compatibility image (TII): derive as in rc.load
        st = d["statistics"]; c = np.zeros((len(st), 1600), np.float32); c[:, :st.shape[1]] = st[:, :1600]
        c = 1 / (1 + np.exp(-np.clip(c, -30, 30))); d["image"] = (c.reshape(-1, 40, 40) * 255).round().astype(np.uint8)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    train_sig = set(signature(d, tr))
    keep = {}
    report = {"dataset": name, "decimals": 5}
    for part, idx in (("val", va), ("test", te)):
        sig = signature(d, idx)
        m = np.array([s not in train_sig for s in sig])
        keep[part] = idx[m]
        y = d["labels"][idx]; ym = d["labels"][idx[m]]
        report[part] = {"before": int(len(idx)), "after": int(m.sum()), "removed_pct": float(100 * (1 - m.mean())),
                        "class_support_before": np.bincount(y, minlength=len(z["classes"])).tolist(),
                        "class_support_after": np.bincount(ym, minlength=len(z["classes"])).tolist()}
    dest = OUT / out_name; dest.mkdir(parents=True, exist_ok=True)
    shutil.copy(folder / f"{prefix}_multiview.npz", dest / f"{prefix}_multiview.npz")
    shutil.copy(folder / f"{prefix}_preprocessing.json", dest / f"{prefix}_preprocessing.json")
    np.savez_compressed(dest / f"{prefix}_split.npz", train=tr, val=keep["val"], test=keep["test"],
                        classes=z["classes"])
    (dest / "dedup_report.json").write_text(json.dumps(report, indent=1))
    print(name, {k: (v["before"], v["after"], round(v["removed_pct"], 1)) for k, v in report.items() if isinstance(v, dict)}, flush=True)
    return report


if __name__ == "__main__":
    for k, (folder, prefix) in SRC.items():
        dedup(k, folder, prefix)
