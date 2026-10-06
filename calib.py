"""Validation-only calibration / blending helpers shared by all model families."""
import numpy as np, torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss, f1_score


def macro(y, p, C): return float(f1_score(y, p, labels=np.arange(C), average="macro", zero_division=0))


def softmax_T(z, T):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def fit_T_logits(z, y, C):
    f = lambda lt: log_loss(y, np.clip(softmax_T(z, float(np.exp(lt))), 1e-9, 1), labels=np.arange(C))
    return float(np.exp(minimize_scalar(f, bounds=(np.log(0.05), np.log(8.0)), method="bounded").x))


def temper_prob(p, T):
    q = np.clip(p, 1e-9, 1) ** (1.0 / T); return q / q.sum(1, keepdims=True)


def fit_T_prob(p, y, C):
    f = lambda lt: log_loss(y, temper_prob(p, float(np.exp(lt))), labels=np.arange(C))
    return float(np.exp(minimize_scalar(f, bounds=(np.log(0.05), np.log(8.0)), method="bounded").x))


def pick(yv, C, cands, comb):
    """Candidate maximising validation macro-F1 (NLL tie-break)."""
    best = None
    for c in cands:
        p = comb(c)
        key = (round(macro(yv, p.argmax(1), C), 4), -float(log_loss(yv, np.clip(p, 1e-9, 1), labels=np.arange(C))))
        if best is None or key > best[0]: best = (key, float(c))
    return best[1]


@torch.no_grad()
def knn_from_embeddings(mem, ym, q, C, k=15, theta=0.1, chunk=2048):
    """Class-frequency-corrected cosine k-NN posterior; mem/q are torch tensors on the same device."""
    mem = torch.nn.functional.normalize(mem, dim=-1); q = torch.nn.functional.normalize(q, dim=-1)
    cnt = torch.bincount(ym, minlength=C).float().clamp_min(1); w = (cnt.sum() / cnt) ** 0.5; w = w / w.mean()
    out = []
    for i in range(0, len(q), chunk):
        s = q[i:i + chunk] @ mem.T; v, ix = s.topk(min(k, len(mem)), dim=1)
        wt = torch.softmax(v / theta, 1); oh = torch.nn.functional.one_hot(ym[ix], C).float()
        p = (wt.unsqueeze(-1) * oh).sum(1) * w; out.append(p / p.sum(1, keepdim=True).clamp_min(1e-9))
    return torch.cat(out).cpu().numpy()
