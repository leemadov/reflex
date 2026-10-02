"""Train the decision model: LoRA on Qwen3-0.6B + pointer head, cross-entropy over each question's options.

    python train.py --steps 3000                  # optimizer steps, --accum sequences each
    python train.py --eval-only --ckpt checkpoints/best --n-val 2000
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model import BASE, Decider, encode, head_state, load

# canonical op -> (key, description) variants; variant 0 is what eval and the README use
OPS = {
    "CLICK": [("CLICK", "click an element on the page"), ("click", "Click a link, button or other element"),
              ("TAP", "tap or click a UI element")],
    "TYPE_TEXT": [("TYPE_TEXT", "type text into an input field"), ("type", "Type into a text box or search field"),
                  ("FILL", "fill in a form field with text")],
    "HOVER": [("HOVER", "hover the mouse over an element"), ("hover", "Move the pointer over an element without clicking")],
    "PRESS_KEY": [("PRESS_KEY", "press a key or keyboard shortcut"), ("press", "Press a key combination such as Enter")],
    "SCROLL_DOWN": [("SCROLL_DOWN", "scroll the page down"), ("scroll_down", "Scroll down to see more of the page")],
    "SCROLL_UP": [("SCROLL_UP", "scroll the page up"), ("scroll_up", "Scroll up toward the top of the page")],
    "GOTO_URL": [("GOTO_URL", "navigate directly to a URL"), ("goto", "Open a specific web address")],
    "GO_BACK": [("GO_BACK", "go back to the previous page"), ("back", "Return to the previously viewed page")],
    "GO_FORWARD": [("GO_FORWARD", "go forward in browser history"), ("forward", "Go forward to the next page in history")],
    "SWITCH_TAB": [("SWITCH_TAB", "open, close or switch browser tabs"), ("tab", "Change to another browser tab")],
    "DONE": [("DONE", "the task is complete; stop and give the answer"), ("stop", "Finish: the objective has been achieved"),
             ("FINISH", "the task is done")],
    "BLOCKED": [("BLOCKED", "the task cannot be completed; give up"), ("impossible", "The objective is impossible or blocked"),
                ("FAIL", "stop because the task cannot be done")],
}
EXTRA = {"SCROLL": [("SCROLL", "scroll the page"), ("scroll", "Scroll the page up or down")],
         "WAIT": [("WAIT", "wait for the page to finish loading"), ("wait", "Pause until the page is ready")]}
OP_Q = ["What is the next operation to perform to accomplish the `task`?", "Which operation should the agent perform next?",
        "Given the `page` and the `history`, what should the agent do next?"]
TGT_Q = ["Which element should the next operation act on?", "Which page element is the target of the next action?",
         "Which element on the page should the agent interact with next?"]
NOUL = [("done", "Is the `task` already accomplished, so the agent can stop?", lambda op: op == "DONE"),
        ("typing", "Does the next step require typing text?", lambda op: op == "TYPE_TEXT"),
        ("blocked", "Is the task impossible or blocked?", lambda op: op == "BLOCKED"),
        ("leave", "Should the agent leave this page (go back or navigate elsewhere)?", lambda op: op in ("GO_BACK", "GOTO_URL"))]
PROGRESS = ["just started", "about halfway", "almost finished"]


def make_example(r, rng=None):
    """record -> (state, questions, gold). rng=None gives the fixed eval format."""
    state = {"task": r["task"], "url": r["url"], "history": r["history"], "page": r["page"]}
    op = r["op"]
    if rng is None:
        crit = {v[0][0]: v[0][1] for v in OPS.values()}
        qs = {"operation": {"type": "choice", "instructions": OP_Q[0], "criteria": crit},
              "target": {"type": "choice", "instructions": TGT_Q[0], "criteria": "elements"}}
        return state, qs, {"operation": OPS[op][0][0], "target": r["target"]}  # target None -> not scored

    ops = list(OPS)
    if rng.random() < 0.3:  # callers may offer a single SCROLL, like Jev's op list
        ops = [o for o in ops if not o.startswith("SCROLL_")] + ["SCROLL"]
        op = "SCROLL" if op.startswith("SCROLL_") else op
    if rng.random() < 0.2:
        ops.append("WAIT")
    ops = [o for o in ops if o == op or rng.random() > 0.15]
    if rng.random() < 0.5:
        rng.shuffle(ops)
    lower = rng.random() < 0.2
    crit, gold_key = {}, None
    for o in ops:
        k, d = rng.choice(OPS.get(o) or EXTRA[o])
        k = k.lower() if lower else k
        crit[k] = d
        gold_key = k if o == op else gold_key
    qs = {rng.choice(["operation", "op", "next_action"]): {"type": "choice", "instructions": rng.choice(OP_Q), "criteria": crit}}
    gold = {next(iter(qs)): gold_key}
    if rng.random() < 0.85:  # asked whatever the gold op is: asking only for targeted ops leaks the op
        name = rng.choice(["target", "element", "ref"])
        qs[name] = {"type": "choice", "instructions": rng.choice(TGT_Q), "criteria": "elements"}
        gold[name] = r["target"]  # None for DONE/GO_BACK/... -> present but no loss
    for name, text, f in rng.sample(NOUL, rng.choice([0, 0, 1, 2])):
        q = {"type": "noul", "instructions": text}
        if rng.random() < 0.5:
            q["criteria"] = {"true": "yes, " + text[0].lower() + text[1:-1], "false": "no"}
        qs[name], gold[name] = q, "true" if f(r["op"]) else "false"
    if r.get("n_steps", 0) > 1 and rng.random() < 0.25:
        qs["progress"] = {"type": "score", "instructions": "How far along is the agent in completing the `task`?",
                          "criteria": PROGRESS}
        gold["progress"] = str(min(2, 3 * r["step"] // r["n_steps"]))
    items = list(qs.items())
    rng.shuffle(items)
    return state, dict(items), gold


def check(recs, n=4000):
    """Which questions get asked must not depend on the gold answer (a target question asked only on
    CLICK/TYPE steps once taught the model 'target question present -> not DONE')."""
    rng, seen = random.Random(0), {True: [], False: []}
    for r in (rng.choice(recs) for _ in range(n)):
        st, qs, gold = make_example(r, rng)
        op_q = next(q for q in qs.values() if q["type"] == "choice" and q.get("criteria") != "elements")
        assert gold[next(k for k, q in qs.items() if q is op_q)] in op_q["criteria"]
        seen[r["target"] is not None].append((any(q.get("criteria") == "elements" for q in qs.values()), len(qs)))
    for i, what in enumerate(("target-question rate", "questions per example")):
        a, b = (sum(x[i] for x in seen[k]) / len(seen[k]) for k in (True, False))
        assert abs(a - b) < 0.05 * max(a, b, 1), f"{what} leaks the op: targeted {a:.3f} vs untargeted {b:.3f}"
    print("check ok: question presence independent of the answer")


def losses(model, tok, ex, max_len):
    state, qs, gold = ex
    ids, specs = encode(tok, state, qs, max_len)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(ids, specs)
    out = []
    for sp, lg in zip(specs, logits):
        if gold[sp["name"]] in sp["keys"]:  # target can be truncated away
            out.append(F.cross_entropy(lg[None], torch.tensor([sp["keys"].index(gold[sp["name"]])], device=lg.device)))
    return out, len(ids)


@torch.no_grad()
def evaluate(model, tok, recs, max_len):
    model.eval()
    rows, ok_op, ok_t, n_t, ok_step, by_src, lat = [], 0, 0, 0, 0, {}, []
    for r in recs:
        t0 = time.perf_counter()
        state, qs, gold = make_example(r)
        ids, specs = encode(tok, state, qs, max_len)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids, specs)
        torch.cuda.synchronize()
        lat.append(time.perf_counter() - t0)
        good = True
        for sp, lg in zip(specs, logits):
            g = gold[sp["name"]]
            if g is None:  # target question on a DONE/GO_BACK/... step
                continue
            hit = bool(sp["keys"]) and sp["keys"][int(lg.argmax())] == g
            good &= hit
            if sp["name"] == "operation":
                ok_op += hit
            else:
                ok_t, n_t = ok_t + hit, n_t + 1
            if g in sp["keys"]:
                rows.append((sp["name"], lg.float().cpu(), sp["keys"].index(g)))
        ok_step += good
        by_src.setdefault(r["src"], []).append(good)
    model.train()
    n = len(recs)
    lat.sort()
    return {"op_acc": ok_op / n, "target_acc": ok_t / max(n_t, 1), "step_acc": ok_step / n, "n": n,
            "latency_ms_p50": round(1000 * lat[n // 2]), "latency_ms_p90": round(1000 * lat[int(n * 0.9)]),
            **{f"step_acc_{k}": sum(v) / len(v) for k, v in sorted(by_src.items())}}, rows


def ece(rows, T=1.0, bins=15):
    conf = torch.tensor([F.softmax(l / T, -1).max().item() for _, l, _ in rows])
    acc = torch.tensor([float(int(l.argmax()) == g) for _, l, g in rows])
    e = 0.0
    for i in range(bins):
        m = (conf > i / bins) & (conf <= (i + 1) / bins)
        if m.any():
            e += m.float().mean().item() * abs(conf[m].mean().item() - acc[m].mean().item())
    return e


def fit_temperature(rows):
    # ponytail: 1-D grid search on NLL; LBFGS if you ever need per-question temperatures
    nll = lambda T: sum(F.cross_entropy(l[None] / T, torch.tensor([g])).item() for _, l, g in rows)  # noqa: E731
    return min((0.5 + 0.05 * i for i in range(51)), key=nll)


def save(model, path, metrics):
    path.mkdir(parents=True, exist_ok=True)
    model.backbone.save_pretrained(path)  # LoRA adapter only
    torch.save(head_state(model), path / "head.pt")
    (path / "metrics.json").write_text(json.dumps(metrics, indent=1))


def read(path, n=None, seed=0, exclude=0):
    recs = [json.loads(line) for line in open(path, encoding="utf-8")]
    if exclude:  # drop the checkpoint-selection sample (read(path, exclude)) so final numbers are unbiased
        drop = set(map(id, random.Random(0).sample(recs, exclude)))
        recs = [r for r in recs if id(r) not in drop]
    return random.Random(seed).sample(recs, n) if n and n < len(recs) else recs


def selective(rows, name, T, thr=0.9):
    """Jev-style use: act only when confident. -> (top5 acc, share of steps with p>=thr, accuracy on those)"""
    ps = [(F.softmax(l / T, -1), g) for n, l, g in rows if n == name]
    hi = [int(p.argmax()) == g for p, g in ps if p.max() >= thr]
    top5 = sum(g in p.topk(min(5, len(p))).indices.tolist() for p, g in ps) / max(len(ps), 1)
    return round(top5, 4), round(len(hi) / max(len(ps), 1), 4), round(sum(hi) / max(len(hi), 1), 4)


def report(model, tok, val, max_len, tag):
    m, rows = evaluate(model, tok, val, max_len)
    T = fit_temperature(rows)
    m.update(ece=round(ece(rows), 4), ece_calibrated=round(ece(rows, T), 4), T=round(T, 2))
    for q in ("operation", "target"):
        m[f"{q}_top5"], m[f"{q}_cover@0.9"], m[f"{q}_acc@0.9"] = selective(rows, q, T)
    print(tag, json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}), flush=True)
    return m, T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--eval-len", type=int, default=16384)
    ap.add_argument("--n-val", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    val = read("data/test.jsonl", a.n_val)
    if a.check:
        return check(read("data/test.jsonl"))
    print("val majority-op baseline:", round(max(sum(r["op"] == o for r in val) for o in OPS) / len(val), 4), flush=True)

    if a.eval_only:
        model, tok = load(a.ckpt)
        report(model, tok, read("data/test.jsonl", a.n_val, seed=1, exclude=400), a.eval_len, "eval (held out)")
        return

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(0)
    rng = random.Random(0)
    tok = AutoTokenizer.from_pretrained(BASE)
    bb = AutoModel.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa")
    bb.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    bb = get_peft_model(bb, LoraConfig(r=64, lora_alpha=128, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model = Decider(bb).cuda().train()
    lora = [p for p in bb.parameters() if p.requires_grad]
    head = [*model.q.parameters(), *model.k.parameters()]
    opt = torch.optim.AdamW([{"params": lora, "lr": a.lr}, {"params": head, "lr": a.lr * 5}], weight_decay=0.0)
    warm = 100
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.45 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))))
    train = read("data/train.jsonl")
    print(f"train {len(train)} records, val {len(val)}; lora params {sum(p.numel() for p in lora) / 1e6:.1f}M", flush=True)

    out, best, t0, toks, run = Path(a.out), -1.0, time.time(), 0, []
    for step in range(1, a.steps + 1):
        for _ in range(a.accum):
            ls, n = losses(model, tok, make_example(rng.choice(train), rng), a.max_len)
            if ls:
                (sum(ls) / len(ls) / a.accum).backward()
                run.append(sum(ls).item() / len(ls))
            toks += n
        torch.nn.utils.clip_grad_norm_(lora + head, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if step % 20 == 0:
            dt = time.time() - t0
            print(f"step {step} loss {sum(run[-160:]) / len(run[-160:]):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"tok/s {toks / dt:.0f} elapsed {dt / 60:.1f}m mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % a.eval_every == 0 or step == a.steps:
            m, T = report(model, tok, val, a.eval_len, f"eval step {step}")
            if m["step_acc"] > best:
                best = m["step_acc"]
                model.T.fill_(T)
                save(model, out / "best", {**m, "step": step})
            save(model, out / "last", {**m, "step": step})
    print("best step_acc", best, flush=True)


if __name__ == "__main__":
    main()
