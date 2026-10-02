"""System 1 + System 2 routing on held-out steps: act on System 1 when its confidence >= t, else ask System 2.

    python eval_route.py --s2-preds checkpoints_s2/best/preds.jsonl

Uses the same records as train_s2's eval (same order as its preds file) and System 1 with the extension's
operation list, so the numbers match what the extension would do. Prints accuracy and share escalated per threshold.
"""
import argparse
import json

import torch

from model import load
from train_s2 import eval_sets

OPS = {"CLICK": "click an element on the page", "TYPE_TEXT": "type text into an input field",  # = extension/background.js
       "HOVER": "hover the mouse over an element", "SCROLL_DOWN": "scroll the page down", "SCROLL_UP": "scroll the page up",
       "GO_BACK": "go back to the previous page", "DONE": "the task is complete; stop and give the answer",
       "BLOCKED": "the task cannot be completed; give up"}
QS = {"operation": {"type": "choice", "instructions": "What is the next operation to perform to accomplish the `task`?", "criteria": OPS},
      "target": {"type": "choice", "instructions": "Which element should the next operation act on?", "criteria": "elements"}}
TARGETED = {"CLICK", "TYPE_TEXT", "HOVER"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s2-preds", default="checkpoints_s2/best/preds.jsonl")
    ap.add_argument("--n-text", type=int, default=300)
    ap.add_argument("--n-img", type=int, default=200)
    a = ap.parse_args()
    recs = eval_sets(a.n_text, a.n_img)
    s2 = [json.loads(line) for line in open(a.s2_preds, encoding="utf-8")]
    assert len(s2) == len(recs) and all(x["target"] == r["target"] for x, r in zip(s2, recs)), "preds don't match the eval set"
    m, tok = load("checkpoints/best")
    rows = []
    with torch.no_grad():
        for r, b in zip(recs, s2):
            ans, _ = m.answer(tok, {"task": r["task"], "url": r["url"], "history": r["history"], "page": r["page"]}, QS)
            op, tgt = ans["operation"]["choice"], ans["target"].get("choice")
            p = ans["operation"]["probabilities"][op] * (ans["target"]["probabilities"].get(tgt, 0) if op in TARGETED else 1)
            ok1 = op == r["op"] and (r["target"] is None or tgt == r["target"])
            rows.append((r["src"], p, ok1, b["ok"]))
    for src in sorted({x[0] for x in rows}) + ["all"]:
        part = [x for x in rows if src in ("all", x[0])]
        n = len(part)
        print(f"\n{src} (n={n})  System 1 alone {sum(x[2] for x in part) / n:.3f}   System 2 alone {sum(x[3] for x in part) / n:.3f}")
        print("  think below | accuracy | asks System 2")
        for t in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            acc = sum((x[2] if x[1] >= t else x[3]) for x in part) / n
            print(f"  {t:>10.2f} | {acc:8.3f} | {sum(x[1] < t for x in part) / n:6.1%}")


if __name__ == "__main__":
    main()
