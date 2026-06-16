"""
Generate candidate action strings for LeWM MPC planning.

Two modes:
  --mode systematic  (default) — build a comprehensive grid from the vocab:
                                  ~200 combos covering mouse × keys
  --mode dataset     — extract top-N most frequent action sequences
                        from gameagent.h5 (run this on Lightning AI)

Usage:
  # Locally — generates systematic candidates
  python scripts/generate_candidates.py --mode systematic \
      --vocab GameAgent/data_processing/outputs/action_vocab.json \
      --out   le-wm/candidates/gameagent_actions.txt

  # On Lightning AI — extracts from real dataset (more representative)
  python GameAgent/scripts/generate_candidates.py --mode dataset \
      --hdf5  /teamspace/studios/this_studio/stable-wm/datasets/gameagent.h5 \
      --vocab GameAgent/data_processing/outputs/action_vocab.json \
      --out   le-wm/candidates/gameagent_actions.txt \
      --top-n 300
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["systematic", "dataset"], default="systematic")
    p.add_argument("--vocab", default="GameAgent/data_processing/outputs/action_vocab.json")
    p.add_argument("--out",   default="le-wm/candidates/gameagent_actions.txt")
    p.add_argument("--hdf5",  default="/teamspace/studios/this_studio/stable-wm/datasets/gameagent.h5",
                   help="(dataset mode only) path to gameagent.h5")
    p.add_argument("--top-n", type=int, default=300,
                   help="(dataset mode only) keep top-N most frequent actions")
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────────────
# Vocab helpers
# ────────────────────────────────────────────────────────────────────────────

def load_vocab(vocab_path: str) -> dict[str, int]:
    d = json.loads(Path(vocab_path).read_text())
    return d["token_to_id"]


def id_to_tok(vocab: dict[str, int]) -> dict[int, str]:
    return {v: k for k, v in vocab.items()}


def motion_value(token: str) -> int | None:
    """Extract bin value from dx_bin_N / dy_bin_N / dz_bin_N tokens."""
    for prefix in ("dx_bin_", "dy_bin_", "dz_bin_"):
        if token.startswith(prefix):
            try:
                return int(token[len(prefix):])
            except ValueError:
                return None
    return None


# ────────────────────────────────────────────────────────────────────────────
# Token-sequence → action string
# ────────────────────────────────────────────────────────────────────────────

def tokens_to_action_str(token_ids: list[int], i2t: dict[int, str]) -> str:
    dx = dy = dz = "0"
    groups: list[list[str]] = [[] for _ in range(6)]
    active: int | None = None

    for tid in token_ids:
        tok = i2t.get(tid, "<unk>")
        if tok in ("<pad>", "<action_start>", "<action_end>", "<unk>"):
            continue
        val = motion_value(tok)
        if val is not None:
            if tok.startswith("dx_"):
                dx = str(val)
            elif tok.startswith("dy_"):
                dy = str(val)
            elif tok.startswith("dz_"):
                dz = str(val)
            continue
        if tok.startswith("<group_") and tok.endswith(">"):
            inner = tok[len("<group_"):-1]
            if inner.isdigit():
                active = int(inner) - 1
            continue
        if tok == "<empty_group>":
            continue
        if tok.startswith("key_") and active is not None:
            groups[active].append(tok.split("_", 1)[1])

    return f"{dx} {dy} {dz} ; " + " ; ".join(",".join(g) for g in groups)


def action_str_to_key(s: str) -> str:
    """Normalise action string for deduplication."""
    return " ".join(s.split())


# ────────────────────────────────────────────────────────────────────────────
# MODE 1: Systematic generation
# ────────────────────────────────────────────────────────────────────────────

def generate_systematic(vocab: dict[str, int]) -> list[str]:
    """
    Build a comprehensive grid of candidate actions from the vocab.
    Covers:
      - No-op
      - All individual keys (one at a time, no mouse)
      - Common mouse movements (no keys)
      - Mouse + single key combos for movement keys
      - Sprint / jump / crouch combos
      - Mouse look + movement combos
    """
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(s: str):
        key = action_str_to_key(s)
        if key not in seen:
            seen.add(key)
            candidates.append(s)

    def _fmt(dx: int, dy: int, dz: int, *group_keys: str) -> str:
        # group_keys: list of key names for group 1 only (can extend later)
        groups = [",".join(group_keys)] + [""] * 5
        return f"{dx} {dy} {dz} ; " + " ; ".join(groups)

    # ── 1. No-op (always index 0) ──────────────────────────────────────────
    _add("0 0 0 ;  ;  ;  ;  ;  ;")

    # ── 2. All individual keys (no mouse, no combos) ───────────────────────
    key_tokens = [k for k in vocab if k.startswith("key_")]
    key_names = sorted(set(k.split("_", 1)[1] for k in key_tokens))
    for kn in key_names:
        _add(_fmt(0, 0, 0, kn))

    # ── 3. Mouse only — coarse grid (dx, dy from -9 to 9 step 3, dz 0) ───
    for dx in range(-9, 10, 3):
        for dy in range(-9, 10, 3):
            if dx == 0 and dy == 0:
                continue
            _add(_fmt(dx, dy, 0))

    # ── 4. Fine mouse for common look directions ───────────────────────────
    for v in (-1, 1, -2, 2, -5, 5):
        _add(_fmt(v, 0, 0))   # strafe-look left/right
        _add(_fmt(0, v, 0))   # pitch up/down

    # ── 5. WASD + mouse look combos ───────────────────────────────────────
    move_keys = ["w", "s", "a", "d"]
    look_dx = [-6, -3, 0, 3, 6]
    look_dy = [-4, -2, 0, 2, 4]
    for mk in move_keys:
        for dx in look_dx:
            for dy in look_dy:
                _add(_fmt(dx, dy, 0, mk))

    # ── 6. Sprint combos (lshift + WASD + mouse) ──────────────────────────
    for mk in move_keys:
        _add(_fmt(0, 0, 0, mk, "lshift"))  # group 1 = move, group 2 = shift
        for dx in [-3, 0, 3]:
            _add(f"0 0 0 ; {mk} ; lshift ;  ;  ;  ;")
            _add(f"{dx} 0 0 ; {mk} ; lshift ;  ;  ;  ;")

    # ── 7. Jump combos ────────────────────────────────────────────────────
    for mk in ["w", "a", "s", "d", ""]:
        key_part = mk if mk else ""
        _add(f"0 0 0 ; {key_part} ; space ;  ;  ;  ;")
        _add(f"3 0 0 ; {key_part} ; space ;  ;  ;  ;")
        _add(f"-3 0 0 ; {key_part} ; space ;  ;  ;  ;")

    # ── 8. Mouse buttons (click, aim, middle) ─────────────────────────────
    for mb in ["lbutton", "rbutton", "mbutton"]:
        _add(_fmt(0, 0, 0, mb))
        for dx in [-3, 0, 3]:
            for dy in [-3, 0, 3]:
                _add(_fmt(dx, dy, 0, mb))

    # ── 9. Aim + shoot ────────────────────────────────────────────────────
    for dx in [-5, -2, 0, 2, 5]:
        for dy in [-5, -2, 0, 2, 5]:
            _add(f"{dx} {dy} 0 ; rbutton ; lbutton ;  ;  ;  ;")

    # ── 10. Crouch combos ─────────────────────────────────────────────────
    for mk in move_keys:
        _add(_fmt(0, 0, 0, mk, "lctrl"))

    # ── 11. Scrollwheel (dz axis) ─────────────────────────────────────────
    for dz in [-3, -1, 1, 3]:
        _add(f"0 0 {dz} ;  ;  ;  ;  ;  ;")

    # ── 12. Common interact keys ──────────────────────────────────────────
    for ik in ["e", "f", "r", "g", "t", "tab", "esc", "enter"]:
        _add(_fmt(0, 0, 0, ik))

    # ── 13. Number keys (hotbar / weapon select) ──────────────────────────
    for nk in ["1", "2", "3", "4", "5"]:
        if f"key_{nk}" in vocab:
            _add(_fmt(0, 0, 0, nk))

    return candidates


# ────────────────────────────────────────────────────────────────────────────
# MODE 2: Dataset extraction (run on Lightning AI)
# ────────────────────────────────────────────────────────────────────────────

def generate_from_dataset(hdf5_path: str, vocab: dict[str, int],
                           top_n: int) -> list[str]:
    try:
        import h5py
    except ImportError:
        sys.exit("h5py not found. Install with: pip install h5py")

    i2t = id_to_tok(vocab)
    counter: Counter[str] = Counter()

    print(f"[dataset] reading {hdf5_path} …")
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"][:]  # [N, L] int64

    print(f"[dataset] {len(actions)} rows — extracting action strings …")
    for row in actions:
        s = tokens_to_action_str(row.tolist(), i2t)
        counter[action_str_to_key(s)] += 1

    print(f"[dataset] {len(counter)} unique action strings")

    # No-op always first
    noop_key = action_str_to_key("0 0 0 ;  ;  ;  ;  ;  ;")
    top = [noop_key]
    for s, _ in counter.most_common(top_n + 1):
        if s not in top:
            top.append(s)
        if len(top) >= top_n:
            break

    return top


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    vocab = load_vocab(args.vocab)

    if args.mode == "systematic":
        candidates = generate_systematic(vocab)
        print(f"[systematic] generated {len(candidates)} candidates")
    else:
        candidates = generate_from_dataset(args.hdf5, vocab, args.top_n)
        print(f"[dataset] extracted {len(candidates)} candidates")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# GameAgent candidate action strings for LeWM MPC planning.\n"
        "# Format: DX DY DZ ; group1 ; group2 ; group3 ; group4 ; group5 ; group6\n"
        "# No-op is always index 0 (required by evaluator).\n"
        f"# Generated by generate_candidates.py --mode {args.mode} "
        f"({len(candidates)} candidates)\n"
        "#\n"
    )
    out.write_text(header + "\n".join(candidates) + "\n")
    print(f"[done] written {len(candidates)} candidates → {out}")


if __name__ == "__main__":
    main()
