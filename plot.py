"""Render the training curves from the .log files produced by train.py's tee.

  python plot.py --logs /content/drive/MyDrive/tiny-shrek --out figures
"""
import argparse
import math
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVAL = re.compile(r"eval (\d+): train ([\d.]+) \| val ([\d.]+)")

SURFACE = "#fcfcfb"
INK, INK_2 = "#0b0b0b", "#52514e"
BLUE, ORANGE = "#2a78d6", "#eb6834"      # categorical slots 1 and 2, CVD-validated
GRID = "#e5e4e0"
LN2 = math.log(2)


def read(path):
    """-> iters, train_bpc, val_bpc. Losses are nats; bits/char is the char-level metric."""
    it, tr, va = [], [], []
    with open(path) as f:
        for m in EVAL.finditer(f.read()):
            it.append(int(m.group(1)))
            tr.append(float(m.group(2)) / LN2)
            va.append(float(m.group(3)) / LN2)
    return it, tr, va


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=13, weight="bold", loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=10)
    ax.grid(axis="y", color=GRID, lw=0.8)          # recessive: y only, behind marks
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def best(it, va):
    i = min(range(len(va)), key=lambda k: va[k])
    return it[i], va[i]


def fig(out):
    f, ax = plt.subplots(figsize=(8, 4.5))
    f.patch.set_facecolor(SURFACE)
    return f, ax, out


def save(f, path):
    f.tight_layout()
    f.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(f)
    print("wrote", path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=".")
    p.add_argument("--out", default="figures")
    p.add_argument("--baseline", type=float, default=1.9588,
                   help="pretrain bpc on shrek_val, from eval.py")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = lambda n: os.path.join(a.logs, n)

    # --- stage 1 -----------------------------------------------------------
    if os.path.exists(L("pretrain.log")):
        it, tr, va = read(L("pretrain.log"))
        f, ax, _ = fig(a.out)
        ax.plot(it, tr, color=ORANGE, lw=2, marker="o", ms=3, label="train")
        ax.plot(it, va, color=BLUE, lw=2, marker="o", ms=3, label="validation")
        bi, bv = best(it, va)
        ax.plot([bi], [bv], marker="o", ms=8, color=BLUE, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"best {bv:.3f} bpc", (bi, bv), textcoords="offset points",
                    xytext=(-10, 14), ha="right", color=INK, fontsize=9, weight="bold")
        style(ax, "Stage 1 — pretraining on 617 films",
              "iteration", "bits per character")
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper right")
        save(f, os.path.join(a.out, "pretrain.png"))

    # --- stage 2: the ablation --------------------------------------------
    have = [n for n in ("finetune.log", "scratch.log") if os.path.exists(L(n))]
    if have:
        f, ax, _ = fig(a.out)
        lo, hi = float("inf"), float("-inf")
        series = [("finetune.log", BLUE, "pretrained → finetuned"),
                  ("scratch.log", ORANGE, "from scratch, Shrek only")]
        for name, color, label in series:
            if name not in have:
                continue
            it, _, va = read(L(name))
            ax.plot(it, va, color=color, lw=2, marker="o", ms=3, label=label)
            bi, bv = best(it, va)
            ax.plot([bi], [bv], marker="o", ms=8, color=color, mec=SURFACE, mew=2, zorder=5)
            dy = 14 if name == "scratch.log" else -20     # opposite sides, no collision
            ax.annotate(f"{bv:.3f}", (bi, bv), textcoords="offset points",
                        xytext=(0, dy), ha="center", color=INK, fontsize=9, weight="bold")
            lo, hi = min(lo, min(va)), max(hi, max(va))

        ax.axhline(a.baseline, color=INK_2, lw=1.2, ls=(0, (5, 4)), zorder=1)
        ax.annotate(f"base model, never saw Shrek — {a.baseline:.3f}",
                    (ax.get_xlim()[0], a.baseline), textcoords="offset points",
                    xytext=(8, 7), ha="left", color=INK_2, fontsize=9)
        ax.set_ylim(lo - 0.28 * (hi - lo), hi + 0.10 * (hi - lo))
        style(ax, "Stage 2 — what pretraining is worth",
              "iteration", "bits per character on held-out Shrek")
        ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="upper right")
        save(f, os.path.join(a.out, "ablation.png"))


if __name__ == "__main__":
    main()
