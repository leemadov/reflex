"""Does cd-0.6b behave like a System One model? (TypeSafe's name for Jev: a model that decides instead of writing.)

    python verify.py [--ckpt checkpoints/best] [--n 300]

Measures each property on held-out pages and prints PASS / FAIL / INFO with the numbers. Exit code 1 on any FAIL.
"""
import argparse
import json
import random
import statistics
import sys
import time

import torch
import torch.nn.functional as F

from model import BASE, ELEM, load
from train import ece, evaluate, make_example, read

FIELDS = {"choice": {"type", "choice", "probabilities", "confidence"}, "noul": {"type", "noul"},
          "score": {"type", "score", "legend", "probabilities", "confidence"}}  # Jev's documented answer shapes
WORDS = ["approve", "reject", "escalate", "billing", "technical", "shipping", "spam", "urgent", "later", "refund"]
results = []


def report(status, name, detail):
    results.append(status)
    print(f"[{status}] {name}: {detail}", flush=True)


def random_questions(r, rng):
    """One question of every type, with option sets the model never saw in training."""
    ids = ELEM.findall(r["page"])
    qs = {"route": {"type": "choice", "instructions": "Which team should handle the `task`?",
                    "criteria": {f"{w}_{i}": f"the {w} team" for i, w in enumerate(rng.sample(WORDS, rng.randint(2, 10)))}},
          "element": {"type": "choice", "instructions": "Which element should the next operation act on?",
                      "criteria": {i: "" for i in rng.sample(ids, min(len(ids), rng.randint(1, 40)))}
                      if ids and rng.random() < 0.5 else "elements"},
          "has_search": {"type": "noul", "instructions": "Does the `page` have a search box?"},
          "progress": {"type": "score", "instructions": "How close is the agent to finishing the `task`?",
                       "criteria": [f"level {i}" for i in range(rng.randint(2, 10))]}}
    if rng.random() < 0.5:
        qs["has_search"]["criteria"] = {"true": "a search field is on the page", "false": "no search field"}
    return dict(rng.sample(list(qs.items()), len(qs)))


def violations(r, qs, ans):
    """Every way an answer could break the typed contract."""
    bad, ids = [], set(ELEM.findall(r["page"]))
    for name, q in qs.items():
        a, t = ans[name], q["type"]
        if set(a) != FIELDS[t] or a["type"] != t:
            bad.append(f"{name}: fields {sorted(a)}")
            continue
        if t == "noul":
            bad += [f"{name}: noul {a['noul']}"] if not 0 <= a["noul"] <= 1 else []
            continue
        p = a["probabilities"]
        allowed = (ids if q["criteria"] == "elements" else set(map(str, range(len(q["criteria"])))) if t == "score"
                   else set(q["criteria"]))
        if not p or not set(p) <= allowed or abs(sum(p.values()) - 1) > 0.02 or not 0 <= a["confidence"] <= 1:
            bad.append(f"{name}: probabilities off the options or not summing to 1")
        if t == "choice" and (a["choice"] not in p or p[a["choice"]] < max(p.values())):
            bad.append(f"{name}: choice {a['choice']!r} is not the top option")
        if t == "score" and (not 0 <= a["score"] <= len(q["criteria"]) - 1 or list(a["legend"].values()) != q["criteria"]
                             or abs(a["score"] - sum(int(k) * v for k, v in p.items())) > 0.02):
            bad.append(f"{name}: score {a['score']} is not the expected level")
    return bad


def timed(fn):
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/best")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    m, tok = load(a.ckpt)
    rng = random.Random(0)
    recs = read("data/test.jsonl", a.n, seed=3, exclude=400)  # held out from training and from checkpoint selection
    state = lambda r: {"task": r["task"], "url": r["url"], "history": r["history"], "page": r["page"]}  # noqa: E731

    report("PASS" if not hasattr(m.backbone, "lm_head") else "FAIL", "Decides, never writes",
           f"backbone is a bare {type(m.backbone).__name__} (no language-model head); the only output layer scores the "
           "options it is given, so every answer is a probability over those options")

    calls, bad, lat, toks, passes = [], [], [], [], []
    hook = m.backbone.register_forward_hook(lambda *_: calls.append(1))
    for r in recs:
        qs = random_questions(r, rng)
        calls.clear()
        (ans, n), dt = timed(lambda: m.answer(tok, state(r), qs))
        lat.append(1000 * dt), toks.append(n), passes.append(len(calls))
        bad += violations(r, qs, ans)
    hook.remove()
    report("PASS" if not bad else "FAIL", "Answers stay inside the supplied options",
           f"{4 * len(recs)} answers (choice, noul, score; 2 to 600+ options; option names never seen in training): "
           f"{len(bad)} violations" + (f", e.g. {bad[:3]}" if bad else ""))
    report("PASS" if set(passes) == {1} else "FAIL", "One forward pass per request",
           f"backbone passes per 4-question request: {sorted(set(passes))}; nothing is decoded token by token")
    lat.sort()
    p50 = lat[len(lat) // 2]
    report("PASS" if p50 < 500 else "FAIL", "Fast", f"p50 {p50:.0f} ms, p90 {lat[int(len(lat) * 0.9)]:.0f} ms per 4-question request "
           f"on real pages (median {statistics.median(toks):.0f} tokens); Jev quotes 70-500 ms")

    r = recs[0]
    forms = {"string": r["page"], "object": state(r), "array": [f"task: {r['task']}", {"url": r["url"]}, r["page"]]}
    yes_no = {"done": {"type": "noul", "instructions": "Is the task done?"}}
    forms_ok = all(m.answer(tok, s, yes_no)[0]["done"]["type"] == "noul" for s in forms.values())
    big = m.answer(tok, state(r), {"pick": {"type": "choice", "instructions": "Pick the option that matches the page.",
                                            "criteria": {f"o{i}": f"option number {i}" for i in range(255)}}})[0]["pick"]
    qs = random_questions(r, random.Random(9))
    same = json.dumps(m.answer(tok, state(r), qs)) == json.dumps(m.answer(tok, state(r), qs))
    report("PASS" if forms_ok and len(big["probabilities"]) == 255 and same else "FAIL", "Jev request/response contract",
           f"state as string/object/array: {forms_ok}; 255-option choice answered: {len(big['probabilities']) == 255}; "
           f"same request twice gives the identical answer: {same}")

    _, rows = evaluate(m, tok, recs, 16384)
    m.eval()
    T = float(m.T)
    conf = [(F.softmax(lg / T, -1).max().item(), int(lg.argmax()) == g) for _, lg, g in rows]
    bins = []
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        b = [(c, ok) for c, ok in conf if lo < c <= lo + 0.2]
        if b:
            bins.append(f"{lo:.1f}-{lo + 0.2:.1f}: says {sum(c for c, _ in b) / len(b):.2f}, right {sum(ok for _, ok in b) / len(b):.2f} (n={len(b)})")
    e = ece(rows, T)
    report("PASS" if e < 0.05 else "FAIL", "Calibrated", f"ECE {e:.3f} on {len(rows)} held-out decisions | " + " | ".join(bins))

    diffs, flips = [], 0
    for r in recs[:60]:
        st, qs, _ = make_example(r)
        both = m.answer(tok, st, qs)[0]
        others = [{k: m.answer(tok, st, {k: q})[0][k] for k, q in qs.items()}, m.answer(tok, st, dict(reversed(qs.items())))[0]]
        for k in qs:
            for o in others:
                diffs.append(max(abs(v - o[k]["probabilities"].get(x, 0)) for x, v in both[k]["probabilities"].items()))
                flips += both[k]["choice"] != o[k]["choice"]
    report("INFO", "Question independence", f"asking a question alone or in the other order moves its probabilities by "
           f"median {statistics.median(diffs):.3f}, max {max(diffs):.3f}; top answer changed in {flips}/{len(diffs)} cases "
           "(Jev answers each question separately; here they share one pass)")

    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import AutoModelForCausalLM
    llm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).cuda().eval()
    s1, s2, n2, cap = [], [], [], 768
    for r in recs[:3]:
        st, qs, _ = make_example(r)
        s1.append(timed(lambda: m.answer(tok, st, qs))[1])
        prompt = (f"Task: {r['task']}\nPage (accessibility tree):\n{tok.decode(tok(r['page']).input_ids[:4000])}\n\n"
                  "Which element should the agent click or type into next? Reason it out, then give the element id.")
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, enable_thinking=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            out, dt = timed(lambda: llm.generate(**enc, max_new_tokens=cap, do_sample=False))
        s2.append(dt), n2.append(out.shape[1] - enc["input_ids"].shape[1])
    report("INFO", "System 1 vs System 2", f"same 3 decisions: this model {1000 * statistics.mean(s1):.0f} ms, one pass each; "
           f"base Qwen3-0.6B reasoning in text {statistics.mean(s2):.1f} s and {statistics.mean(n2):.0f} generated tokens each "
           f"(one pass per token{', hit the ' + str(cap) + '-token cap' if max(n2) >= cap else ''})")

    print(f"\n{results.count('PASS')} PASS, {results.count('FAIL')} FAIL, {results.count('INFO')} INFO")
    sys.exit(1 if "FAIL" in results else 0)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
