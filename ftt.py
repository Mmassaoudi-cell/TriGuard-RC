"""Compact FT-Transformer baseline (Gorishniy et al., 2021) for early-fused tabular features."""
import copy
import numpy as np, torch
from torch import nn
from torch.nn import functional as F


class FTT(nn.Module):
    def __init__(self, n_feat, C, d=32, heads=4, layers=2, ff=64, p=0.1):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_feat, d) * 0.1); self.b = nn.Parameter(torch.zeros(n_feat, d))
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        layer = nn.TransformerEncoderLayer(d, heads, ff, p, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C))

    def forward(self, x):
        t = x.unsqueeze(-1) * self.w + self.b
        h = self.enc(torch.cat([self.cls.expand(len(x), -1, -1), t], 1))
        return self.head(h[:, 0])


def train_ftt(Xtr, ytr, Xva, yva, C, seed, dev="cuda", epochs=40, patience=8, bs=512, lr=1e-3):
    from sklearn.metrics import f1_score
    torch.manual_seed(seed); np.random.seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    tf = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32, device=dev).clamp(-8, 8)
    xtr, xva = tf(Xtr), tf(Xva); ytr_t = torch.tensor(ytr, device=dev)
    cnt = np.bincount(ytr, minlength=C).clip(1); w = (cnt.sum() / cnt) ** 0.5; w = torch.tensor(w / w.mean(), dtype=torch.float32, device=dev)
    m = FTT(Xtr.shape[1], C).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    best, state, bad = -1, None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(xtr), device=dev)
        for i in range(0, len(perm), bs):
            ix = perm[i:i + bs]; opt.zero_grad(); F.cross_entropy(m(xtr[ix]), ytr_t[ix], weight=w).backward(); opt.step()
        m.eval()
        with torch.no_grad(): pv = torch.cat([m(xva[i:i + 4096]) for i in range(0, len(xva), 4096)]).argmax(1).cpu().numpy()
        f = f1_score(yva, pv, labels=np.arange(C), average="macro", zero_division=0)
        if f > best + 1e-9: best, state, bad = f, copy.deepcopy(m.state_dict()), 0
        else: bad += 1
        if bad >= patience: break
    m.load_state_dict(state); m.eval()

    @torch.no_grad()
    def logits(X):
        x = tf(X); return torch.cat([m(x[i:i + 4096]) for i in range(0, len(x), 4096)]).cpu().numpy()
    return logits, best
