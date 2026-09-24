"""Build training bins.

  python prepare.py cornell    # download + parse Cornell Movie-Dialogs -> stage 1
  python prepare.py shrek      # parse data/shrek_raw/*.txt -> stage 2
  python prepare.py shrek --inspect   # print parsed exchanges without writing

Emits speaker-tagged plaintext, encoded to uint16 .bin files.
"""
import argparse
import glob
import os
import pickle
import re
import sys
import unicodedata
import zipfile
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CORNELL_URL = "http://www.cs.cornell.edu/~cristian/data/cornell_movie_dialogs_corpus.zip"
N_VAL_MOVIES = 10

_SUBS = {"‘": "'", "’": "'", "“": '"', "”": '"',
         "–": "-", "—": "-", "…": "...", "\xa0": " "}


def clean(text):
    """Normalize to a small stable charset so stage 2 can't introduce unseen chars."""
    for a, b in _SUBS.items():
        text = text.replace(a, b)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if c.isprintable() or c == "\n")
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[ \t]+", " ", text).strip()


def write_split(name, train_text, val_text, vocab=None):
    os.makedirs(DATA, exist_ok=True)
    if vocab is None:
        vocab = sorted(set(train_text + val_text))
    stoi = {c: i for i, c in enumerate(vocab)}

    dropped = 0
    def encode(s):
        nonlocal dropped
        out = []
        for c in s:
            if c in stoi:
                out.append(stoi[c])
            else:
                dropped += 1
        return np.array(out, dtype=np.uint16)

    for split, text in (("train", train_text), ("val", val_text)):
        arr = encode(text)
        arr.tofile(os.path.join(DATA, f"{name}_{split}.bin"))
        print(f"  {name}_{split}.bin  {len(arr):,} tokens")
    if dropped:
        print(f"  dropped {dropped:,} out-of-vocab chars")
    return vocab


# ---------------------------------------------------------------- cornell

def cornell():
    zip_path = os.path.join(DATA, "cornell.zip")
    root = os.path.join(DATA, "cornell movie-dialogs corpus")
    os.makedirs(DATA, exist_ok=True)

    if not os.path.isdir(root):
        if not os.path.exists(zip_path):
            import urllib.request
            print("downloading Cornell corpus...")
            try:
                urllib.request.urlretrieve(CORNELL_URL, zip_path)
            except Exception as e:
                sys.exit(f"download failed ({e}).\n"
                         f"Grab it manually from {CORNELL_URL}\n"
                         f"and save it to {zip_path}, then re-run.")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA)

    enc = dict(encoding="iso-8859-1")  # the corpus is NOT utf-8
    lines, speakers, movie_of = {}, {}, {}
    with open(os.path.join(root, "movie_lines.txt"), **enc) as f:
        for row in f:
            p = row.split(" +++$+++ ")
            if len(p) == 5:
                lines[p[0]] = clean(p[4])
                speakers[p[0]] = clean(p[3]).upper()
                movie_of[p[0]] = p[2]

    convs = []
    with open(os.path.join(root, "movie_conversations.txt"), **enc) as f:
        for row in f:
            p = row.split(" +++$+++ ")
            if len(p) == 4:
                ids = re.findall(r"L\d+", p[3])
                if len(ids) > 1:
                    convs.append((p[2], ids))

    # hold out WHOLE MOVIES: a random-line split leaks context across the boundary
    movies = sorted({m for m, _ in convs})
    val_movies = set(movies[-N_VAL_MOVIES:])
    print(f"{len(convs):,} conversations, {len(movies)} movies "
          f"({len(val_movies)} held out for val)")

    train_parts, val_parts = [], []
    for movie, ids in convs:
        block = "\n".join(f"{speakers[i]}: {lines[i]}"
                          for i in ids if i in lines and lines[i])
        if block.count("\n") < 1:
            continue
        (val_parts if movie in val_movies else train_parts).append(block)

    train_text = "\n\n".join(train_parts)
    val_text = "\n\n".join(val_parts)
    print(f"train {len(train_text):,} chars | val {len(val_text):,} chars")

    vocab = write_split("cornell", train_text, val_text)
    with open(os.path.join(DATA, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab": vocab, "stoi": {c: i for i, c in enumerate(vocab)},
                     "itos": dict(enumerate(vocab))}, f)
    print(f"vocab: {len(vocab)} chars -> data/meta.pkl")


# ---------------------------------------------------------------- shrek

SPEAKER_RE = re.compile(r"^\s{0,40}([A-Z][A-Z0-9 .'\-]{1,28})\s*(?:\(.*\))?\s*$")
SKIP = {"FADE IN", "FADE OUT", "CUT TO", "INT", "EXT", "THE END",
        "DISSOLVE TO", "CONTINUED", "SMASH CUT TO"}
PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\.?\s*$")


def drop_running_headers(lines, min_repeats=5, min_len=20):
    """Screenplay page headers repeat on every page with only the number changing.

    Normalize digits away, count, and drop the frequent long ones. Length and the
    digit requirement keep short speaker tags (SHREK, PIG #1) out of the count.
    """
    norm = [re.sub(r"\d+", "#", l.strip()) for l in lines]
    counts = Counter(n for n, l in zip(norm, lines)
                     if len(n) >= min_len and any(c.isdigit() for c in l))
    return [l for l, n in zip(lines, norm) if counts.get(n, 0) < min_repeats]


def parse_screenplay(text):
    """Heuristic screenplay parser: ALL-CAPS line = speaker, indented lines below = dialogue."""
    out, speaker, buf = [], None, []

    def flush():
        if speaker and buf:
            # strip parentheticals AFTER joining - they often span several lines
            said = re.sub(r"\([^)]*\)", " ", " ".join(buf))
            said = re.sub(r"[()]", " ", said)      # orphans left by a dropped opener
            said = clean(re.sub(r"\s+", " ", said))
            if said:
                out.append(f"{speaker}: {said}")

    for raw in drop_running_headers(text.splitlines()):
        line = raw.rstrip()
        if PAGE_NUM_RE.match(line):
            continue                      # page number mid-scene: skip, don't split the speech
        if not line.strip():
            flush(); speaker, buf = None, []
            continue
        m = SPEAKER_RE.match(line)
        name = m.group(1).strip() if m else None
        if name and not any(name.startswith(s) for s in SKIP) and len(name.split()) <= 3:
            flush(); speaker, buf = name, []
        elif speaker:
            buf.append(line.strip())
    flush()
    return out


def shrek(inspect=False):
    raw_dir = os.path.join(DATA, "shrek_raw")
    os.makedirs(raw_dir, exist_ok=True)   # gitignored, so absent on a fresh clone
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.txt")))
    if not paths:
        sys.exit(f"put the script .txt files in {raw_dir}/ first")

    exchanges = []
    for p in paths:
        got = parse_screenplay(open(p, encoding="utf-8", errors="ignore").read())
        print(f"{os.path.basename(p)}: {len(got)} lines")
        exchanges += got

    if inspect:
        from collections import Counter
        print("\n--- first 40 ---")
        print("\n".join(exchanges[:40]))
        print("\n--- top speakers ---")
        for n, c in Counter(e.split(":")[0] for e in exchanges).most_common(15):
            print(f"  {c:5d}  {n}")
        return

    with open(os.path.join(DATA, "meta.pkl"), "rb") as f:
        vocab = pickle.load(f)["vocab"]   # reuse stage-1 vocab

    cut = int(len(exchanges) * 0.85)      # last 15% of scenes held out for the voice eval
    write_split("shrek", "\n".join(exchanges[:cut]), "\n".join(exchanges[cut:]), vocab)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["cornell", "shrek"])
    ap.add_argument("--inspect", action="store_true")
    a = ap.parse_args()
    cornell() if a.source == "cornell" else shrek(a.inspect)
