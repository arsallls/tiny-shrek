"""Llama-style decoder-only transformer, from scratch.

RoPE + RMSNorm + SwiGLU + no biases. ~14M params at the default config.
Run `python model.py` for a self-check.
"""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 100
    n_layer: int = 8
    n_head: int = 6
    n_embd: int = 384


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # norm in fp32: bf16 rsqrt loses too much precision here
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def precompute_rope(head_dim, max_seq, base=10000.0):
    """Returns cos, sin of shape (max_seq, head_dim // 2)."""
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, nh, T, hd).  cos/sin: (T, hd//2), already sliced to this segment."""
    cos = cos.view(1, 1, cos.size(0), -1)
    sin = sin.view(1, 1, sin.size(0), -1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        assert self.head_dim % 2 == 0, "RoPE needs an even head_dim"
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)

    def forward(self, x, cos, sin, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # with a cache we are decoding at position `past`, so rotate from there
        past = 0 if cache is None else cache[0].size(2)
        q = apply_rope(q, cos[past:past + T], sin[past:past + T])
        k = apply_rope(k, cos[past:past + T], sin[past:past + T])

        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)

        # prefill (no cache) needs the causal mask; single-token decode attends
        # to the whole cache, so no mask. ponytail: assumes T==1 when cache is set.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None and T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = 64 * ((hidden + 63) // 64)  # keep the matmuls tensor-core friendly
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1 = RMSNorm(cfg.n_embd)
        self.attn = Attention(cfg)
        self.n2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        h, new_cache = self.attn(self.n1(x), cos, sin, cache)
        x = x + h
        x = x + self.mlp(self.n2(x))
        return x, new_cache


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # tied

        cos, sin = precompute_rope(cfg.n_embd // cfg.n_head, cfg.block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        # scale residual projections so the residual stream does not blow up with depth
        for name, p in self.named_parameters():
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx, targets=None, caches=None):
        x = self.tok(idx)
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, self.cos, self.sin, None if caches is None else caches[i])
            new_caches.append(c)
        x = self.norm(x)

        if targets is None:
            # only the last position matters when generating
            logits = self.head(x[:, [-1], :])
            return logits, new_caches

        logits = self.head(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def configure_optimizers(self, weight_decay, lr, betas, device_type):
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        nodecay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        fused = device_type == "cuda"
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=200, use_cache=True):
        caches = None
        for _ in range(max_new_tokens):
            if use_cache and caches is not None:
                step_in = idx[:, -1:]
            else:
                step_in = idx[:, -self.cfg.block_size:]
                caches = None

            logits, caches = self(step_in, caches=caches)
            if not use_cache:
                caches = None
            # drop the oldest cached position if we are at the context limit
            elif caches[0][0].size(2) >= self.cfg.block_size:
                caches = [(k[:, :, 1:], v[:, :, 1:]) for k, v in caches]

            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


if __name__ == "__main__":
    cfg = GPTConfig()
    m = GPT(cfg)
    print(f"params: {m.num_params() / 1e6:.2f}M")

    x = torch.randint(0, cfg.vocab_size, (2, 64))
    # targets must be independent of x: tied embeddings + the residual stream make
    # "predict the current token" trivial at init, which hides a broken init.
    y = torch.randint(0, cfg.vocab_size, (2, 64))
    logits, loss = m(x, targets=y)
    assert logits.shape == (2, 64, cfg.vocab_size), logits.shape

    # an untrained model should be uniform over the vocab
    expected = math.log(cfg.vocab_size)
    assert abs(loss.item() - expected) < 0.35, f"init loss {loss.item():.3f} != ~{expected:.3f}"
    print(f"init loss {loss.item():.3f} (expected ~{expected:.3f}) OK")

    # cached and uncached generation must agree given a fixed seed
    torch.manual_seed(0); a = m.generate(x[:, :8], 16, temperature=1.0, use_cache=True)
    torch.manual_seed(0); b = m.generate(x[:, :8], 16, temperature=1.0, use_cache=False)
    assert torch.equal(a, b), "KV cache changed the output"
    print("kv-cache parity OK")
