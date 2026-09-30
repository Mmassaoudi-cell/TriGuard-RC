"""Shared evaluation helpers: leakage-safe metrics + inference-time perturbations."""
import sys
from pathlib import Path
import numpy as np, torch
ETD = Path(r"c:/Users/MMASSAOUDI/Desktop/research work/Alienware/Enhanced TriGuard/TriGuard-ETD")
sys.path.insert(0, str(ETD))
from triguard.metrics import classification_metrics, calibration_metrics  # noqa: E402


def full_metrics(y, prob, classes):
    pred = prob.argmax(1)
    m = classification_metrics(y, pred, prob, classes)
    present = [i for i in range(len(classes)) if (y == i).any()]
    from sklearn.metrics import f1_score
    m["macro_f1_present"] = float(f1_score(y, pred, labels=present, average="macro", zero_division=0))
    m["calibration"] = calibration_metrics(y, prob)
    keep = ("accuracy", "macro_f1", "macro_f1_present", "weighted_f1", "precision", "recall", "fpr", "fnr", "pr_auc", "roc_auc")
    out = {k: m[k] for k in keep}
    out["ece"] = m["calibration"].get("ece"); out["nll"] = m["calibration"].get("nll")
    out["per_class_f1"] = {c["class"]: c["f1"] for c in m["per_class"]}
    return out


def perturb(name, img, tim, sta, gen):
    """img in [0,1] (B,1,40,40).  Perturbations mirror TriGuard-ETD's robustness diagnostics
    (+ three held-out families that TriGuard-RC's augmentation never sees)."""
    if name == "byte_padding_20":
        img = img.clone(); img[:, :, 32:, :] = 0
    elif name == "byte_insertion_10":
        m = torch.rand(img.shape, device=img.device, generator=gen) < 0.10
        img = torch.where(m, torch.rand(img.shape, device=img.device, generator=gen), img)
    elif name == "byte_insertion_25_heldout":
        m = torch.rand(img.shape, device=img.device, generator=gen) < 0.25
        img = torch.where(m, torch.rand(img.shape, device=img.device, generator=gen), img)
    elif name == "byte_block_occlusion_heldout":
        img = img.clone(); img[:, :, 8:24, 8:24] = 0
    elif name == "byte_gauss_heldout":
        img = (img + 0.2 * torch.randn(img.shape, device=img.device, generator=gen)).clamp(0, 1)
    elif name == "u_byte_row_shuffle":
        perm = torch.randperm(img.shape[2], device=img.device, generator=gen); img = img[:, :, perm, :]
    elif name == "u_byte_shift":
        img = torch.roll(img, shifts=8, dims=2)
    elif name == "u_byte_salt_pepper":
        u = torch.rand(img.shape, device=img.device, generator=gen)
        img = torch.where(u < 0.075, torch.zeros_like(img), torch.where(u > 0.925, torch.ones_like(img), img))
    elif name == "u_timing_packet_drop":
        tim = tim.clone(); drop = torch.rand(tim.shape[:2], device=tim.device, generator=gen).argsort(1)[:, :2]
        tim.scatter_(1, drop.unsqueeze(-1).expand(-1, -1, tim.shape[2]), 0.0)
    elif name == "u_stats_noise":
        sta = sta + 0.3 * torch.randn(sta.shape, device=sta.device, generator=gen)
    elif name in ("timing_jitter_20", "timing_jitter_50"):
        f = 0.2 if name.endswith("20") else 0.5
        tim = tim + torch.randn(tim.shape, device=tim.device, generator=gen) * (tim.abs().mean() + 1e-6) * f
    elif name == "direction_corruption_20":
        m = torch.rand(tim.shape[:-1], device=tim.device, generator=gen) < 0.2
        tim = tim.clone(); tim[..., 2] = torch.where(m, -tim[..., 2], tim[..., 2])
    return img, tim, sta


UNSEEN = ["u_byte_row_shuffle", "u_byte_shift", "u_byte_salt_pepper", "u_timing_packet_drop", "u_stats_noise"]
PERTS = ["byte_padding_20", "byte_insertion_10", "timing_jitter_20", "timing_jitter_50", "direction_corruption_20",
         "byte_insertion_25_heldout", "byte_block_occlusion_heldout", "byte_gauss_heldout"]
MASKS = {"missing_byte": [0, 1, 1], "missing_timing": [1, 0, 1], "missing_statistics": [1, 1, 0]}


@torch.no_grad()
def robustness(predict_logits, g, classes, seed=0, part="test"):
    """predict_logits(img,tim,sta,mask)->logits. Returns macro-F1 per condition on TEST."""
    from sklearn.metrics import f1_score
    ids = g.idx[part]; y = g.y[ids].cpu().numpy(); C = len(classes)
    def run(fn, mask_over=None):
        gen = torch.Generator(device=g.img.device); gen.manual_seed(seed)
        outs = []
        for i in range(0, len(ids), 4096):
            img, tim, sta, mask, _ = g.batch(ids[i:i + 4096])
            if fn: img, tim, sta = perturb(fn, img, tim, sta, gen)
            if mask_over is not None: mask = mask & torch.tensor(mask_over, dtype=torch.bool, device=mask.device)[None]
            outs.append(predict_logits(img, tim, sta, mask).argmax(1))
        p = torch.cat(outs).cpu().numpy()
        return float(f1_score(y, p, labels=np.arange(C), average="macro", zero_division=0))
    res = {"clean": run(None)}
    for n in PERTS + UNSEEN: res[n] = run(n)
    for n, m in MASKS.items(): res[n] = run(None, m)
    return res
