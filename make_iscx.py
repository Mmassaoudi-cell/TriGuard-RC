"""Independent evaluation benchmark: ISCXVPN2016 application/traffic-type flows (CS240 release, CICFlowMeter features).
Not used in any development step of TriGuard-RC.  Same view construction as TII-SSRC-23 (statistics, aggregate timing
proxy from the IAT/duration columns, compatibility image from the scaled statistics); duplicate-aware group split."""
import hashlib, json, os
from pathlib import Path
import numpy as np, pandas as pd
import sys
sys.path.insert(0, r"c:/Users/MMASSAOUDI/Desktop/research work/Alienware/Enhanced TriGuard/TriGuard-ETD")
from triguard.data import create_split_bundle, fit_train_only_features, _labels_to_ids
from make_regroup import union_find_groups

SRC = Path(r"C:/Users/MMASSAOUDI/Desktop/Data/CS240_ISCXVPN2016/data.csv")
SPLIT_SEED = int(os.environ.get("SPLIT_SEED", 42))
OUT = Path(__file__).parent / "manifests_rg" / ("iscx" if SPLIT_SEED == 42 else f"iscx_s{SPLIT_SEED}"); OUT.mkdir(parents=True, exist_ok=True)
DROP = {"Flow ID", "Src IP", "Dst IP", "Timestamp", "Label", "flow_start", "Category", "App_protocol", "Web_service"}
TIMING = ["Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Flow Duration"]
CAP = 15000; SEED = 42

df = pd.read_csv(SRC, low_memory=False)
df = df[df["Label"] != "unknown"].reset_index(drop=True)
rng = np.random.default_rng(SEED)
keep = np.concatenate([rng.permutation(np.flatnonzero(df["Label"].values == c))[:CAP] for c in sorted(df["Label"].unique())])
df = df.iloc[np.sort(keep)].reset_index(drop=True)
cols = [c for c in df.columns if c not in DROP]
raw = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32); raw[~np.isfinite(raw)] = np.nan
labels, classes = _labels_to_ids(df["Label"].astype(str).values)
prov = (df["Flow ID"].astype(str) + "|" + df["Label"].astype(str)).values  # Flow ID conflicts across labels are split by label
tab = np.round(np.nan_to_num(raw.astype(np.float64)), 5)
sig = [hashlib.blake2b(np.ascontiguousarray(r).tobytes() + int(l).to_bytes(2, "little"), digest_size=10).hexdigest() for r, l in zip(tab, labels)]
groups = union_find_groups(list(prov), sig)
bundle = create_split_bundle(labels, groups, raw, classes, seed=SPLIT_SEED)
scaled, prep = fit_train_only_features(raw, bundle)
idx = [cols.index(c) for c in TIMING]
timing = np.zeros((len(scaled), 5, 5), np.float32)
for t in range(5):
    timing[:, t, :5] = scaled[:, idx]; timing[:, t, -1] = t / 4.0
mask = np.ones((len(labels), 3), bool)
np.savez_compressed(OUT / "iscx_multiview.npz", image=np.zeros((1, 40, 40), np.uint8), timing=timing, statistics=scaled, labels=labels,
                    groups=groups, view_mask=mask, classes=np.asarray(classes, dtype=object), feature_columns=np.asarray(cols, dtype=object))
np.savez_compressed(OUT / "iscx_split.npz", train=bundle.train, val=bundle.val, test=bundle.test, classes=np.asarray(classes, dtype=object))
(OUT / "iscx_preprocessing.json").write_text(json.dumps({"dataset": "ISCXVPN2016", "image_lazy": True, "timing_view": "aggregate timing proxy from IAT/duration columns",
    "preprocessing": prep, "source": str(SRC), "classes": classes, "cap_per_class": CAP}, indent=1))
rep = {"rows": int(len(labels)), "sizes": {k: int(len(getattr(bundle, k))) for k in ("train", "val", "test")}, "classes": classes,
       "test_support": np.bincount(labels[bundle.test], minlength=len(classes)).tolist(), "train_only": list((bundle.train_only_classes or {}).keys()),
       "n_features": len(cols), "n_groups": int(len(set(groups)))}
(OUT / "regroup_report.json").write_text(json.dumps(rep, indent=1)); print(rep)
