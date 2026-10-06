# TriGuard-RC

Reliability-calibrated, retrieval-augmented tri-view detector for encrypted-traffic detection.
It builds on the released TriGuard-ETD code and reuses its frozen multi-view caches and
provenance-grouped split manifests (`LitCVit_3datasets/artifacts/{5gad_full,ustc_fast,tii_stratified_sample}`).

## Files

| File | Purpose |
|---|---|
| `rc.py` | Model (byte / order-aware timing / PLE-statistics encoders, evidence-aware gate), subset-consistent training, group-aware cross-fitted teacher, retrieval readout, EMA |
| `final.py` | Final protocol: train, select epoch / temperature / retrieval weight / GBDT weight on **validation**, open **test** once. Also produces the hybrid (RC-H), the GBDT-expert robustness and the ablation variants |
| `orig.py` | Re-runs the *original* TriGuard-ETD (their `train_enhanced`) on the same manifests and seeds, with the same perturbation study |
| `baselines.py` | LightGBM / XGBoost / ExtraTrees / kNN on early-fused features (validation-only tuning) |
| `evalx.py` | Metrics + inference-time perturbation families (three are never used for training) |
| `dev.py`, `dev_rob.py` | Validation-only development harness (never reads the test partition) |
| `stats.py`, `efficiency.py` | Welch tests / paired bootstrap; params, MACs, latency |
| `aggregate.py`, `figures.py`, `fixtables.py`, `build_paper.sh` | Tables, figures and paper build |
| `results/` | One JSON per run (`rc_*`, `orig_*`, `base_*`), `summary.json`, `stats.json`, `efficiency.json` |

## Reproduce

```bash
python baselines.py --ds 5gad --seeds 1,2,3            # and tii, ustc
python orig.py      --ds 5gad --seeds 1,2,3,4,5 --robust   # and tii, ustc (no --robust)
python final.py     --ds 5gad --seeds 1,2,3,4,5 --robust
python final.py     --ds tii  --seeds 1,2,3,4,5
python final.py     --ds ustc --seeds 1,2,3,4,5
python final.py     --ds tii  --seeds 1,2 --variants no_kd,insample_kd,no_viewdrop,no_aug,no_ple,byte_only,timing_only,stats_only,timing_stats
python stats.py && python efficiency.py && bash build_paper.sh
```

Requires PyTorch (CUDA), scikit-learn, LightGBM, XGBoost, SciPy, thop, matplotlib.
