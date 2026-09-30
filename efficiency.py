import json, sys, time
import numpy as np, torch
import rc, evalx
sys.path.insert(0, str(evalx.ETD))
from triguard.models import TriGuardETDEnhanced
from triguard.metrics import benchmark_latency, model_macs
from dev import DS
torch.set_num_threads(1)
out = {}
for ds in ("5gad", "ustc", "iscx", "tii"):
    d = rc.load(str(DS[ds])); C = len(d["classes"]); sd = d["statistics"].shape[1]
    tr = d["split"]["train"]
    edges = rc.ple_edges(d["statistics"][tr], 12)
    m_rc = rc.RC(sd, C, edges).eval()
    cuts = torch.from_numpy(np.quantile(d["statistics"][tr].astype(np.float64), np.linspace(0, 1, 14)[1:-1], axis=0).T.astype(np.float32))
    m_o = TriGuardETDEnhanced(sd, C, cuts).eval()
    x = (torch.rand(1, 1, 40, 40), torch.rand(1, 5, 5), torch.randn(1, sd), torch.ones(1, 3, dtype=torch.bool))
    r = {}
    for nm, m in (("orig", m_o), ("rc", m_rc)):
        with torch.inference_mode():
            lat = benchmark_latency(lambda: m(*x), "cpu", 30, 200); mac = model_macs(m, x)
        r[nm] = dict(params=sum(p.numel() for p in m.parameters()), lat_ms=lat["latency_ms_mean"], macs=mac["macs"])
    r["memory_mb"] = len(tr) * 64 * 4 / 2**20; r["train_n"] = int(len(tr))
    # retrieval overhead: one query vs memory
    mem = torch.nn.functional.normalize(torch.randn(len(tr), 64), dim=-1); q = torch.nn.functional.normalize(torch.randn(1, 64), dim=-1)
    lat = benchmark_latency(lambda: (q @ mem.T).topk(15), "cpu", 30, 200); r["retrieval_ms"] = lat["latency_ms_mean"]
    out[ds] = r; print(ds, r, flush=True)
json.dump(out, open("results_rg/efficiency.json", "w"), indent=1)
