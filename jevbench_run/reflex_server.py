"""Reflex behind JevBench's /v1/systemone wire format, so JevBench's own typesafe adapter can measure it.

    python jevbench_run/reflex_server.py --mode reason|routed [--port 8766] [--temperature T] [--escalate-below C]
                                        [--log jevbench_run/reason/log.jsonl]

reason  Reflex Reason writes its reasoning inside <think>, as it was trained to; then the options are lettered A, B,
        C... and one forward pass after "</think>```" gives its probability for each letter (softened by --temperature,
        fitted by calibrate.py), so every option gets a probability.
routed  Reflex Instinct answers first in one pass; when its confidence is below --escalate-below, Reason decides.
        Confidence = top probability rescaled so chance is 0 and certainty 1: (top - 1/k) / (1 - 1/k), which puts
        two-option and six-option questions on one scale.
--log appends each decision's raw letter scores, reasoning and routing to a file, for calibrate.py.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import model as s1  # noqa: E402 - Reflex Instinct; also sets the project's attention backends
import train_s2 as s2  # noqa: E402 - Reflex Reason

SYSTEM = ("You make one decision. Given the situation, a question and lettered options, think step by step about which "
          "option is right, then give the letter of exactly one option in triple backticks.")
LETTERS = "ABCDEFGHIJ"


def options(q):
    """-> [(key, text shown)], the order the options are lettered in"""
    t, c = q.get("type", "choice"), q.get("criteria")
    if t == "noul":
        c = c or {}
        return [("yes", f"yes: {c.get('true', 'yes')}"), ("no", f"no: {c.get('false', 'no')}")]
    if t == "score":
        return [(str(i), f"level {i}: {d}") for i, d in enumerate(c)]
    if isinstance(c, dict):
        return [(str(k), f"{k}: {v}") for k, v in c.items()]
    return [(str(k), str(k)) for k in c]


def as_answer(q, probs):
    t = q.get("type", "choice")
    if t == "noul":
        return {"type": "noul", "noul": probs["yes"]}
    if t == "score":
        return {"type": "score", "score": round(sum(int(k) * v for k, v in probs.items()), 3), "probabilities": probs}
    return {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}


@torch.no_grad()
def reason(m, proc, state, q, temperature=1.0, max_new=384):
    """-> (probs over option keys, raw letter logits, reasoning, prompt tokens, reasoning tokens)"""
    opts = options(q)
    body = (f"{state if isinstance(state, str) else s2.render_state(state)}\n\nQuestion: {q.get('instructions', '')}\n"
            "Options:\n" + "\n".join(f"{LETTERS[i]}. {text}" for i, (_, text) in enumerate(opts)))
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    tok = proc.tokenizer
    enc = {"input_ids": tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()}
    thought = s2.generate(m, proc, enc, max_new).split("</think>")[0].strip()
    ids = tok(prompt + thought + "\n</think>\n\n```", add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    pad = -ids.shape[1] % 256  # one cuDNN plan per 256-token bucket, as in generate(); causal, so padding changes nothing
    s2.base_of(m).model.rope_deltas = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(input_ids=torch.cat([ids, ids.new_zeros((1, pad))], 1), logits_to_keep=pad + 1).logits[0, 0].float()
    letters = logits[[tok.convert_tokens_to_ids(LETTERS[i]) for i in range(len(opts))]]
    p = torch.softmax(letters / temperature, -1).tolist()
    probs = {k: round(v, 4) for (k, _), v in zip(opts, p)}
    return probs, letters.tolist(), thought, enc["input_ids"].shape[1], len(tok(thought, add_special_tokens=False).input_ids)


def instinct(m, tok, state, q):
    """-> (probs over JevBench's option keys, prompt tokens)"""
    a, n = m.answer(tok, state, {"decision": q})
    a = a["decision"]
    probs = {"yes": a["noul"], "no": round(1 - a["noul"], 4)} if a["type"] == "noul" else a["probabilities"]
    return probs, n


def confidence(probs):
    k = len(probs)
    return (max(probs.values()) - 1 / k) / (1 - 1 / k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reason", "routed"], default="reason")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--temperature", type=float, default=3.176)  # calibrate.py fit (settings.json); 1 = raw
    ap.add_argument("--escalate-below", type=float, default=0.5)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    import uvicorn
    from fastapi import FastAPI
    m2, proc = s2.load(str(ROOT / "checkpoints_s2/best"), merge=True)
    m1, tok1 = s1.load(str(ROOT / "checkpoints/best")) if a.mode == "routed" else (None, None)
    app = FastAPI(title=f"reflex-{a.mode}")

    @app.post("/v1/systemone")
    def systemone(req: dict):
        answers, usage = {}, {"input_tokens": 0, "output_tokens": 0, "instinct_input_tokens": 0}
        for name, q in req["questions"].items():
            rec = {"key": hashlib.sha256(json.dumps([req["state"], q], sort_keys=True).encode()).hexdigest()[:16]}
            probs = None
            if m1 is not None:
                probs, n = instinct(m1, tok1, req["state"], q)
                usage["instinct_input_tokens"] += n
                rec.update(instinct=probs, confidence=round(confidence(probs), 4))
            if probs is None or confidence(probs) < a.escalate_below:
                probs, letters, thought, n_in, n_out = reason(m2, proc, req["state"], q, a.temperature)
                usage["input_tokens"] += n_in
                usage["output_tokens"] += n_out
                rec.update(reason=probs, letters=letters, thought=thought, escalated=True)
            answers[name] = as_answer(q, probs)
            if a.log:
                with open(a.log, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
        return {"model": f"reflex-{a.mode}", "answers": answers, "usage": usage}

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
