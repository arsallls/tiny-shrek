"""Generate dialogue, or benchmark the KV cache.

  python sample.py --ckpt out/shrek.pt --prompt "SHREK: "
  python sample.py --ckpt out/shrek.pt --bench
"""
import argparse
import os
import pickle
import time

import torch

from model import GPT, GPTConfig

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ck["cfg"])).to(device).eval()
    model.load_state_dict(ck["model"])
    with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
        vocab = pickle.load(f)["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    return model, stoi, dict(enumerate(vocab))


def bench(model, stoi, device, n=512):
    """KV-cache throughput.

    Batch matters more than length here. At batch 1 a 14M model is launch-bound -
    ~100 tiny CUDA kernels per token in eager mode - so the attention FLOPs the
    cache removes are a few percent of wall time and the speedup vanishes. Batching
    amortizes that overhead until attention is actually the bottleneck.
    """
    print(f"{'batch':>6} {'no cache':>13} {'cache':>13} {'speedup':>9}")
    for b in (1, 8, 32, 64):
        idx = torch.full((b, 1), stoi.get("\n", 0), dtype=torch.long, device=device)
        row = []
        for use_cache in (False, True):
            model.generate(idx, 8, use_cache=use_cache)           # warmup
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            model.generate(idx, n, use_cache=use_cache)
            if device == "cuda":
                torch.cuda.synchronize()                           # or you time nothing
            row.append(n * b / (time.perf_counter() - t))
        print(f"{b:>6} {row[0]:>10.0f} t/s {row[1]:>10.0f} t/s {row[1] / row[0]:>8.2f}x")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="out/shrek.pt")
    p.add_argument("--prompt", default="SHREK: ")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--bench", action="store_true")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load(a.ckpt, device)

    if a.bench:
        return bench(model, stoi, device)

    prompt = a.prompt.encode().decode("unicode_escape")   # so --prompt can carry \n
    ids = [stoi[c] for c in prompt if c in stoi] or [0]
    idx = torch.tensor([ids], device=device)
    for i in range(a.num):
        out = model.generate(idx, a.tokens, a.temperature, a.top_k)
        print(f"\n===== sample {i + 1} =====")
        print("".join(itos[int(t)] for t in out[0]))


if __name__ == "__main__":
    main()
