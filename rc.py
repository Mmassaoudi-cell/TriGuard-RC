"""TriGuard-RC: reliability-calibrated, retrieval-augmented tri-view detector.

Self-contained implementation (GPU-resident tensors, no DataLoader) that reads
the frozen multi-view caches / split manifests produced by the TriGuard-ETD
preprocessing pipeline.  Nothing here touches the test partition until
``evaluate(..., partition="test")`` is called.

Novel components relative to TriGuard-ETD
  1. Subset-consistent training  : random view subsets + leave-one-view-out
                                   consistency so the gate reflects reliability.
  2. Evidence-aware gate         : gate sees expert entropies, not only embeddings.
  3. Order-aware timing encoder  : positional (flatten) read-out instead of mean.
  4. PLE statistics encoder      : piecewise-linear quantile encoding + gated MLP.
  5. Corruption-aware augmentation of each view (byte substitution, jitter, noise).
  6. Group-aware cross-fitted (out-of-fold) tree-teacher distillation.
  7. Train-only prototype retrieval head blended with the parametric posterior.
  8. EMA weights, validation-only blending / temperature.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import f1_score, log_loss

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- data
def load(cache: str):
    """Load a cache + its frozen split; TII images are rebuilt lazily from stats."""
    p = Path(cache)
    prefix = p.stem.removesuffix("_multiview")
    z = np.load(p, allow_pickle=True)
    s = np.load(p.parent / f"{prefix}_split.npz", allow_pickle=True)
    meta = json.loads((p.parent / f"{prefix}_preprocessing.json").read_text(encoding="utf-8"))
    d = {k: z[k] for k in ("image", "timing", "statistics", "labels", "view_mask", "groups")}
    d["classes"] = z["classes"].tolist()
    d["split"] = {k: s[k] for k in ("train", "val", "test")}
    if meta.get("image_lazy"):  # TII: compatibility image from scaled stats (as in TriGuard-ETD)
        st = d["statistics"]
        n = len(st)
        canvas = np.zeros((n, 1600), dtype=np.float32)
        canvas[:, : min(st.shape[1], 1600)] = st[:, :1600]
        canvas = 1 / (1 + np.exp(-np.clip(canvas, -30, 30)))
        d["image"] = (canvas.reshape(n, 40, 40) * 255).round().astype(np.uint8)
    return d


class GPUData:
    def __init__(self, d):
        self.img = torch.from_numpy(d["image"]).to(DEV)  # uint8 N,40,40
        self.tim = torch.from_numpy(d["timing"].astype(np.float32)).to(DEV)
        self.sta = torch.from_numpy(d["statistics"].astype(np.float32)).to(DEV)
        self.y = torch.from_numpy(d["labels"]).to(DEV)
        self.mask = torch.from_numpy(d["view_mask"]).to(DEV)
        self.idx = {k: torch.from_numpy(v).to(DEV) for k, v in d["split"].items()}
        self.C = len(d["classes"])

    def batch(self, ids):
        return (self.img[ids].float().div_(255).unsqueeze(1), self.tim[ids], self.sta[ids],
                self.mask[ids], self.y[ids])


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ teacher (OOF)
def _lgb(seed, balanced=False):
    import lightgbm as lgb
    return lgb.LGBMClassifier(class_weight="balanced" if balanced else None, n_estimators=240, learning_rate=0.05, num_leaves=31,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=seed, verbosity=-1, n_jobs=16)


def _probs(m, x, C):
    out = np.full((len(x), C), 1e-6, np.float32)
    out[:, m.classes_.astype(int)] = m.predict_proba(x)
    return out / out.sum(1, keepdims=True)


def crossfit_teacher(d, seed, folds=4, balanced=False):
    """Group-aware out-of-fold teacher posteriors on the training rows.

    Rows of classes with < `folds` provenance groups cannot be cross-fitted
    without emptying the class from the training folds; they receive the
    full-fit posterior.  Returns (train_teacher[len(train),C], val/test full-fit)."""
    from sklearn.model_selection import StratifiedGroupKFold
    tr = d["split"]["train"]; C = len(d["classes"])
    x = np.concatenate([d["statistics"], d["timing"].reshape(len(d["timing"]), -1)], 1)
    y, g = d["labels"], d["groups"]
    xtr, ytr, gtr = x[tr], y[tr], g[tr]
    oof = np.zeros((len(tr), C), np.float32)
    ng = {c: len(set(gtr[ytr == c])) for c in np.unique(ytr)}
    sk = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    covered = np.zeros(len(tr), bool)
    for a, b in sk.split(xtr, ytr, gtr):
        m = _lgb(seed, balanced).fit(xtr[a], ytr[a])
        oof[b] = _probs(m, xtr[b], C); covered[b] = True
    full = _lgb(seed, balanced).fit(xtr, ytr)
    weak = np.isin(ytr, [c for c, n in ng.items() if n < folds])
    if weak.any():
        oof[weak] = _probs(full, xtr[weak], C)
    return oof, full, x


# --------------------------------------------------------------------------- model
class SE(nn.Module):
    def __init__(s, c, r=4):
        super().__init__()
        h = max(c // r, 8)
        s.g = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, h, 1), nn.GELU(), nn.Conv2d(h, c, 1), nn.Sigmoid())

    def forward(s, x): return x * s.g(x)


class DSBlock(nn.Module):
    def __init__(s, c, dil=1):
        super().__init__()
        s.dw = nn.Conv2d(c, c, 3, 1, dil, groups=c, dilation=dil, bias=False); s.b1 = nn.BatchNorm2d(c)
        s.pw = nn.Conv2d(c, c, 1, bias=False); s.b2 = nn.BatchNorm2d(c); s.se = SE(c)

    def forward(s, x):
        h = F.gelu(s.b1(s.dw(x)))
        return F.gelu(x + s.se(s.b2(s.pw(h))))


class ByteEnc(nn.Module):
    def __init__(s, D, c=48, depth=3):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(1, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.GELU())
        s.blocks = nn.Sequential(*[DSBlock(c, 1 if i % 2 == 0 else 2) for i in range(depth)])
        s.out = nn.Sequential(nn.LayerNorm(2 * c), nn.Linear(2 * c, D))

    def forward(s, x):
        h = s.blocks(s.stem(x))
        return s.out(torch.cat([h.mean((2, 3)), h.amax((2, 3))], 1))


class TimeEnc(nn.Module):
    """Order-aware: temporal conv stack followed by a positional flatten read-out."""
    def __init__(s, D, L=5, cin=5, c=48, depth=2, p=0.1):
        super().__init__()
        s.proj = nn.Linear(cin, c)
        s.blocks = nn.ModuleList([nn.ModuleDict(dict(n=nn.LayerNorm(c), dw=nn.Conv1d(c, c, 3, padding=1, groups=c),
                                                    pw=nn.Conv1d(c, c, 1))) for _ in range(depth)])
        s.dp = nn.Dropout(p)
        s.out = nn.Sequential(nn.LayerNorm(L * c), nn.Linear(L * c, D))

    def forward(s, x):
        h = s.proj(x)
        for b in s.blocks:
            u = b["n"](h).transpose(1, 2)
            h = h + s.dp(b["pw"](F.gelu(b["dw"](u))).transpose(1, 2))
        return s.out(h.flatten(1))


class GMLP(nn.Module):
    def __init__(s, h, p):
        super().__init__()
        s.n = nn.LayerNorm(h); s.e = nn.Linear(h, 2 * h); s.c = nn.Linear(h, h); s.d = nn.Dropout(p)

    def forward(s, x):
        v, g = s.e(s.n(x)).chunk(2, -1)
        return x + s.d(s.c(v * F.gelu(g)))


class PLEStatEnc(nn.Module):
    """Piecewise-linear quantile encoding (train-fitted edges) + gated residual MLP."""
    def __init__(s, F_, D, edges, h=192, depth=2, p=0.1):
        super().__init__()
        s.register_buffer("lo", edges[:, :-1].contiguous()); s.register_buffer("hi", edges[:, 1:].contiguous())
        T = edges.shape[1] - 1
        s.inp = nn.Sequential(nn.LayerNorm(F_ * (1 + T)), nn.Linear(F_ * (1 + T), h), nn.GELU())
        s.blocks = nn.Sequential(*[GMLP(h, p) for _ in range(depth)])
        s.out = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, D))

    def forward(s, x):
        w = (s.hi - s.lo).clamp_min(1e-6)
        ple = ((x.unsqueeze(-1) - s.lo) / w).clamp(0, 1).flatten(1)
        return s.out(s.blocks(s.inp(torch.cat([x, ple], 1))))


def ple_edges(x, T):
    q = np.quantile(x.astype(np.float64), np.linspace(0, 1, T + 1), axis=0).T  # F,T+1
    q = np.nan_to_num(q, nan=0.0)
    for i in range(1, q.shape[1]):  # strictly increasing edges
        q[:, i] = np.maximum(q[:, i], q[:, i - 1] + 1e-4)
    return torch.tensor(q, dtype=torch.float32)


class EvidenceGate(nn.Module):
    """Softmax gate over available views; input = embeddings + expert uncertainty."""
    def __init__(s, D, h=96):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(3 * D + 6, h), nn.GELU(), nn.Linear(h, 3))

    def forward(s, zs, ev, mask):
        lg = s.net(torch.cat(zs + [ev], 1)).masked_fill(~mask, -1e4)
        return torch.softmax(lg, 1)


class RC(nn.Module):
    def __init__(s, sdim, C, edges, D=64, p=0.12, tim_len=5):
        super().__init__()
        s.D, s.C = D, C
        s.byte = ByteEnc(D); s.tim = TimeEnc(D, L=tim_len); s.sta = PLEStatEnc(sdim, D, edges, p=p)
        s.heads = nn.ModuleList([nn.Linear(D, C) for _ in range(3)])
        s.gate = EvidenceGate(D)
        s.cls = nn.Sequential(nn.LayerNorm(D), nn.Dropout(p), nn.Linear(D, C))
        s.proj = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D))
        s.mask_tok = nn.Parameter(torch.zeros(3, D))  # learned "absent view" embedding

    def encode(s, img, tim, sta):
        return [s.byte(img), s.tim(tim), s.sta(sta)]

    def fuse(s, zs, mask):
        zs = [torch.where(mask[:, i:i + 1], z, s.mask_tok[i].expand_as(z)) for i, z in enumerate(zs)]
        bl = torch.stack([h(z) for h, z in zip(s.heads, zs)], 1)  # B,3,C
        pr = torch.softmax(bl, -1)
        ent = -(pr * pr.clamp_min(1e-9).log()).sum(-1) / np.log(s.C)  # B,3
        ev = torch.cat([ent, pr.amax(-1)], 1).detach()
        a = s.gate(zs, ev, mask)
        f = (a.unsqueeze(-1) * torch.stack(zs, 1)).sum(1)
        logits = s.cls(f) + (a.unsqueeze(-1) * bl).sum(1)
        return dict(logits=logits, emb=f, proj=F.normalize(s.proj(f), dim=-1), gates=a, branch=bl)

    def forward(s, img, tim, sta, mask):
        return s.fuse(s.encode(img, tim, sta), mask)


# ------------------------------------------------------------------- augmentation
def augment(img, tim, sta, cfg):
    B = img.shape[0]
    on = (torch.rand(B, device=img.device) < cfg.get("aug_frac", 1.0)).float()  # corrupt only a fraction of the samples
    if cfg["byte_sub"] > 0:  # substitute a random fraction of pixels (byte insertion/corruption)
        frac = torch.rand(B, 1, 1, 1, device=img.device) * cfg["byte_sub"] * on.view(B, 1, 1, 1)
        m = torch.rand_like(img) < frac
        img = torch.where(m, torch.rand_like(img), img)
    if cfg["jitter"] > 0:  # additive timing noise, per-sample scale up to `jitter` x mean|x|
        sc = torch.rand(B, 1, 1, device=tim.device) * cfg["jitter"] * (tim.abs().mean() + 1e-6) * on.view(B, 1, 1)
        tim = tim + torch.randn_like(tim) * sc
    if cfg["noise"] > 0:
        sta = sta + torch.randn_like(sta) * cfg["noise"] * on.view(B, 1)
    return img, tim, sta


def sample_mask(mask, p_drop):
    """Random view subset that keeps at least one available view.  `p_drop` may be a scalar or a per-view triple."""
    p = torch.as_tensor(p_drop, dtype=torch.float32, device=mask.device)
    if float(p.max()) <= 0: return mask
    keep = mask & (torch.rand(mask.shape, device=mask.device) >= p)
    empty = ~keep.any(1)
    if empty.any():
        ch = torch.multinomial(mask[empty].float(), 1).squeeze(1)
        keep[empty, ch] = True
    return keep


# ----------------------------------------------------------------------- training
DEFAULT = dict(D=64, T=12, epochs=60, patience=14, bs=512, lr=1.5e-3, wd=2e-4, class_power=0.5,
               ls=0.05, kd=0.6, kd_T=2.0, aux=0.3, supcon=0.02, cons=0.5, view_drop=0.4,
               byte_sub=0.10, jitter=0.5, noise=0.05, aug_frac=0.5, view_drop_vec=[0.5, 0.3, 0.4], ema=0.995, knn_k=15, knn_T=0.1, dropout=0.12)


def supcon(z, y, t=0.15):
    if len(z) < 2: return z.new_zeros(())
    lg = z @ z.T / t
    eye = torch.eye(len(z), device=z.device, dtype=torch.bool)
    lg = lg.masked_fill(eye, -1e9)
    pos = y[:, None].eq(y[None]) & ~eye
    v = pos.any(1)
    if not v.any(): return z.new_zeros(())
    return -((F.log_softmax(lg, 1) * pos).sum(1) / pos.sum(1).clamp_min(1))[v].mean()


def kd_loss(logits, tprob, T):
    return T * T * F.kl_div(F.log_softmax(logits / T, 1), tprob.clamp_min(1e-8).log(), reduction="batchmean", log_target=True)


@torch.no_grad()
def infer(model, g: GPUData, ids, bs=4096, mask_override=None):
    model.eval()
    out = dict(logits=[], emb=[], gates=[])
    for i in range(0, len(ids), bs):
        b = ids[i:i + bs]
        img, tim, sta, mask, _ = g.batch(b)
        if mask_override is not None: mask = mask & mask_override.to(mask.device)[None, :]
        r = model(img, tim, sta, mask)
        for k in out: out[k].append(r[k])
    return {k: torch.cat(v) for k, v in out.items()}


def macro(y, p, C): return float(f1_score(y, p, labels=np.arange(C), average="macro", zero_division=0))


def train(d, g: GPUData, seed, cfg=None, teacher=None, log=True):
    cfg = {**DEFAULT, **(cfg or {})}
    set_seed(seed)
    C = g.C; tr, va = g.idx["train"], g.idx["val"]
    edges = ple_edges(d["statistics"][d["split"]["train"]], cfg["T"])
    model = RC(d["statistics"].shape[1], C, edges.to(DEV), cfg["D"], cfg["dropout"]).to(DEV)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters(): p.requires_grad_(False)
    cnt = torch.bincount(g.y[tr], minlength=C).float()
    w = (cnt.sum() / cnt.clamp_min(1)) ** cfg["class_power"]; w = (w / w.mean()).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps_per = int(np.ceil(len(tr) / cfg["bs"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["epochs"] * steps_per, pct_start=0.1)
    T_tr = None
    if teacher is not None:
        T_tr = torch.from_numpy(teacher).to(DEV)  # aligned with train order
    yv = g.y[va].cpu().numpy()
    best, best_state, best_ep, bad, hist = -1, None, 0, 0, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train()
        if cfg.get("sampler", 0) > 0:  # class-balanced resampling (exponent on inverse class frequency)
            pw = (1.0 / cnt[g.y[tr]]) ** cfg["sampler"]
            perm = torch.multinomial(pw, len(tr), replacement=True)
        else:
            perm = torch.randperm(len(tr), device=DEV)
        tot = 0.0
        for i in range(steps_per):
            pos = perm[i * cfg["bs"]:(i + 1) * cfg["bs"]]
            ids = tr[pos]
            img, tim, sta, mask, y = g.batch(ids)
            img, tim, sta = augment(img, tim, sta, cfg)
            zs = model.encode(img, tim, sta)
            full = model.fuse(zs, mask)
            sub_mask = sample_mask(mask, cfg.get("view_drop_vec") or cfg["view_drop"])
            sub = model.fuse(zs, sub_mask)
            lg = full["logits"]
            ce = F.cross_entropy(lg, y, weight=w, label_smoothing=cfg["ls"])
            ce_sub = F.cross_entropy(sub["logits"], y, weight=w, label_smoothing=cfg["ls"])
            aux = torch.stack([F.cross_entropy(full["branch"][:, v][mask[:, v]], y[mask[:, v]], weight=w)
                               for v in range(3)]).mean()
            # counterfactual consistency: subset posterior should match the (detached) full posterior
            cons = kd_loss(sub["logits"], F.softmax(lg.detach(), 1), 1.0)
            loss = ce + ce_sub + cfg["aux"] * aux + cfg["cons"] * cons + cfg["supcon"] * supcon(full["proj"], y)
            if T_tr is not None and cfg["kd"] > 0:
                loss = loss + cfg["kd"] * kd_loss(lg, T_tr[pos], cfg["kd_T"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            with torch.no_grad():
                dcy = min(cfg["ema"], (1 + ep * steps_per + i) / (10 + ep * steps_per + i))
                for pe, pm in zip(ema.parameters(), model.parameters()): pe.mul_(dcy).add_(pm.detach(), alpha=1 - dcy)
                for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)
            tot += float(loss.detach())
        r = infer(ema, g, va)
        pv = r["logits"].argmax(1).cpu().numpy()
        sc = macro(yv, pv, C)
        nll = float(F.cross_entropy(r["logits"], g.y[va]).cpu())
        hist.append(dict(epoch=ep + 1, loss=tot / steps_per, val_macro_f1=sc, val_nll=nll))
        if sc > best + 1e-9: best, best_state, best_ep, bad = sc, copy.deepcopy(ema.state_dict()), ep + 1, 0
        else: bad += 1
        if log: print(f"  ep{ep+1:3d} loss={tot/steps_per:.4f} val_f1={sc:.4f} nll={nll:.4f} best={best:.4f}@{best_ep} {time.time()-t0:.0f}s", flush=True)
        if bad >= cfg["patience"]: break
    ema.load_state_dict(best_state)
    return ema, dict(best_val_macro_f1=best, best_epoch=best_ep, epochs_run=len(hist), history=hist, cfg=cfg,
                     params=sum(p.numel() for p in ema.parameters()), train_seconds=time.time() - t0)


# ----------------------------------------------------------- retrieval + blending
@torch.no_grad()
def knn_probs(model, g, query_ids, cfg, exclude_self=False):
    """Cosine k-NN over train-only fused-embedding memory (class-balanced vote)."""
    mem = F.normalize(infer(model, g, g.idx["train"])["emb"], dim=-1)
    ym = g.y[g.idx["train"]]
    q = F.normalize(infer(model, g, query_ids)["emb"], dim=-1)
    C = g.C
    cnt = torch.bincount(ym, minlength=C).float().clamp_min(1)
    prior_w = (cnt.sum() / cnt) ** 0.5; prior_w = prior_w / prior_w.mean()
    out = []
    for i in range(0, len(q), 2048):
        s = q[i:i + 2048] @ mem.T
        v, ix = s.topk(cfg["knn_k"], dim=1)
        wgt = torch.softmax(v / cfg["knn_T"], 1)
        oh = F.one_hot(ym[ix], C).float()
        p = (wgt.unsqueeze(-1) * oh).sum(1) * prior_w
        out.append(p / p.sum(1, keepdim=True).clamp_min(1e-9))
    return torch.cat(out)


def fit_blend(p_par, p_knn, y, C, grid=np.linspace(0, 1, 11)):
    """Choose blend weight on validation only (macro-F1, NLL tie-break)."""
    best = None
    for lam in grid:
        p = ((1 - lam) * p_par + lam * p_knn).cpu().numpy()
        f = macro(y, p.argmax(1), C)
        n = float(log_loss(y, np.clip(p, 1e-7, 1), labels=np.arange(C)))
        key = (round(f, 4), -n)
        if best is None or key > best[0]: best = (key, lam)
    return float(best[1])


def predict_all(model, g, part, cfg, lam=None, mask_override=None):
    ids = g.idx[part]
    r = infer(model, g, ids, mask_override=mask_override)
    p_par = F.softmax(r["logits"], 1)
    p_knn = knn_probs(model, g, ids, cfg) if lam is None or lam > 0 else None
    return p_par, p_knn, r["gates"]
