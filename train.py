"""Train stage 1 (pretrain) or stage 2 (finetune).

  python train.py --stage pretrain --out_dir /content/drive/MyDrive/tiny-shrek
  python train.py --stage finetune --out_dir /content/drive/MyDrive/tiny-shrek
  python train.py --stage scratch  --out_dir out/ablation   # no-pretrain baseline

Resumes automatically from the newest checkpoint in --out_dir.
"""
import argparse
import math
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["pretrain", "finetune", "scratch"], default="pretrain")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--max_iters", type=int, default=0, help="0 = stage default")
    p.add_argument("--lr", type=float, default=0.0, help="0 = stage default")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--ckpt_interval", type=int, default=500)
    p.add_argument("--patience", type=int, default=6, help="evals without val improvement")
    p.add_argument("--replay", type=float, default=0.15, help="finetune: fraction of cornell")
    p.add_argument("--compile", action="store_true")
    a = p.parse_args()
    if not a.max_iters:
        a.max_iters = {"pretrain": 12000, "finetune": 600, "scratch": 2000}[a.stage]
    if not a.lr:
        a.lr = {"pretrain": 6e-4, "finetune": 3e-5, "scratch": 6e-4}[a.stage]
    return a


def load_bin(name):
    path = os.path.join(DATA, f"{name}.bin")
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} - run prepare.py first")
    return np.memmap(path, dtype=np.uint16, mode="r")


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(1337)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # T4 (sm_75) has no bf16 -> fp16 + GradScaler; L4/A100 -> bf16, no scaler
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        print(f"{torch.cuda.get_device_name(0)} | {'bf16' if use_bf16 else 'fp16'}")
    else:
        dtype = torch.float32
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))
    autocast = (torch.amp.autocast(device_type="cuda", dtype=dtype)
                if device == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False))

    prefix = "cornell" if args.stage == "pretrain" else "shrek"
    data = {s: load_bin(f"{prefix}_{s}") for s in ("train", "val")}
    replay = load_bin("cornell_train") if args.stage == "finetune" and args.replay > 0 else None
    print(f"{args.stage}: train {len(data['train']):,} | val {len(data['val']):,} tokens")

    def get_batch(split):
        # replay mixing keeps the finetune from forgetting general English
        src = data[split]
        if split == "train" and replay is not None and np.random.rand() < args.replay:
            src = replay
        ix = torch.randint(len(src) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([torch.from_numpy(src[i:i + args.block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(src[i + 1:i + 1 + args.block_size].astype(np.int64)) for i in ix])
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)

    with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
        vocab_size = len(pickle.load(f)["vocab"])

    cfg = GPTConfig(block_size=args.block_size, vocab_size=vocab_size,
                    n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd)
    model = GPT(cfg).to(device)
    opt = model.configure_optimizers(0.1, args.lr, (0.9, 0.95), device)

    start_iter, best_val, stale = 0, float("inf"), 0
    ckpt_path = os.path.join(args.out_dir, f"{args.stage}.pt")

    if args.stage == "finetune" and not os.path.exists(ckpt_path):
        src = os.path.join(args.out_dir, "pretrain.pt")
        if not os.path.exists(src):
            raise SystemExit(f"finetune needs {src}")
        model.load_state_dict(torch.load(src, map_location=device)["model"])
        print(f"initialized from {src}")
    elif os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_iter, best_val = ck["iter"] + 1, ck["best_val"]
        print(f"resumed from iter {start_iter} (best val {best_val:.4f})")

    print(f"params: {model.num_params() / 1e6:.2f}M")
    if args.compile:
        model = torch.compile(model)
    raw = getattr(model, "_orig_mod", model)

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        r = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        return 0.1 * args.lr + 0.45 * args.lr * (1 + math.cos(math.pi * min(r, 1.0)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for split in ("train", "val"):
            losses = torch.zeros(args.eval_iters)
            for k in range(args.eval_iters):
                x, y = get_batch(split)
                with autocast:
                    _, loss = model(x, targets=y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    t0 = time.time()
    for it in range(start_iter, args.max_iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        for micro in range(args.grad_accum):
            x, y = get_batch("train")
            with autocast:
                _, loss = model(x, targets=y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        if it % 50 == 0:
            print(f"iter {it:6d} | loss {loss.item() * args.grad_accum:.4f} "
                  f"| lr {lr_at(it):.2e} | {time.time() - t0:.0f}s")

        if it % args.eval_interval == 0 and it > start_iter:
            m = evaluate()
            bpc = m["val"] / math.log(2)
            print(f"  eval {it}: train {m['train']:.4f} | val {m['val']:.4f} | {bpc:.4f} bpc")
            if m["val"] < best_val:
                best_val, stale = m["val"], 0
                torch.save({"model": raw.state_dict(), "optim": opt.state_dict(),
                            "cfg": cfg.__dict__, "iter": it, "best_val": best_val},
                           ckpt_path)
                print(f"  saved {ckpt_path}")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop: val flat for {stale} evals")
                    break

        # periodic save so a Colab disconnect doesn't cost the whole run
        if it % args.ckpt_interval == 0 and it > start_iter:
            torch.save({"model": raw.state_dict(), "optim": opt.state_dict(),
                        "cfg": cfg.__dict__, "iter": it, "best_val": best_val},
                       os.path.join(args.out_dir, f"{args.stage}_last.pt"))

    print(f"done. best val {best_val:.4f} ({best_val / math.log(2):.4f} bpc)")


if __name__ == "__main__":
    main()
