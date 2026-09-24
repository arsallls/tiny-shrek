"""Evals: bits-per-char, character-voice perplexity, memorization audit.

  python eval.py --ckpt out/pretrain.pt --ckpt out/shrek.pt --split shrek_val
  python eval.py --ckpt out/shrek.pt --memorize
"""
import argparse
import math
import os
import pickle

import numpy as np
import torch

from sample import load

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@torch.no_grad()
def bpc(model, split, device, block=None, batches=200, bs=16):
    """Bits per character on a held-out split - the right metric for char-level."""
    block = block or model.cfg.block_size   # must match the checkpoint, not a constant
    data = np.memmap(os.path.join(DATA, f"{split}.bin"), dtype=np.uint16, mode="r")
    rng = np.random.default_rng(0)     # fixed so checkpoints are compared on the same windows
    total = 0.0
    for _ in range(batches):
        ix = rng.integers(0, len(data) - block - 1, bs)
        x = torch.stack([torch.from_numpy(data[i:i + block].astype(np.int64)) for i in ix]).to(device)
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block].astype(np.int64)) for i in ix]).to(device)
        _, loss = model(x, targets=y)
        total += loss.item()
    nll = total / batches
    return nll, nll / math.log(2)


def memorize(model, device, k_values=(10, 20, 40), n_samples=20, tokens=400):
    """What fraction of generated k-grams appear verbatim in the training corpora?

    Checks every corpus the model saw. The finetuned model's recitation risk is
    highest for the Shrek scripts - 105k tokens seen several times over - so
    auditing only the large pretraining corpus would miss the copyrighted one.
    """
    with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
        vocab = pickle.load(f)["vocab"]
    itos, stoi = dict(enumerate(vocab)), {c: i for i, c in enumerate(vocab)}

    idx = torch.tensor([[stoi.get("\n", 0)]], device=device)
    samples = []
    for _ in range(n_samples):
        out = model.generate(idx, tokens, temperature=0.8)
        samples.append("".join(itos[int(t)] for t in out[0]))

    for name in ("cornell_train", "shrek_train"):
        path = os.path.join(DATA, f"{name}.bin")
        if not os.path.exists(path):
            continue
        arr = np.memmap(path, dtype=np.uint16, mode="r")
        corpus = "".join(itos[int(t)] for t in arr)
        print(f"  vs {name} ({len(corpus):,} chars)")
        for k in k_values:
            # hash-set of every corpus k-gram; collisions are negligible at this scale
            seen = {hash(corpus[i:i + k]) for i in range(len(corpus) - k)}
            hits = tot = 0
            for s in samples:
                for i in range(len(s) - k):
                    tot += 1
                    hits += hash(s[i:i + k]) in seen
            print(f"    {k:>3}-gram verbatim: {100 * hits / max(tot, 1):5.2f}%  ({hits}/{tot})")
            del seen
        del corpus


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", action="append", required=True)
    p.add_argument("--split", default="shrek_val")
    p.add_argument("--memorize", action="store_true")
    a = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    for path in a.ckpt:
        model, _, _ = load(path, device)
        if a.memorize:
            print(f"\n== memorization: {path}")
            memorize(model, device)
        else:
            nll, b = bpc(model, a.split, device)
            print(f"{path:<34} {a.split:<14} nll {nll:.4f}  {b:.4f} bpc  ppl {math.exp(nll):.2f}")


if __name__ == "__main__":
    main()
