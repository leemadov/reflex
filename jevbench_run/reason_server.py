"""Reflex Reason behind JevBench's /v1/systemone wire format, so JevBench's own typesafe adapter can measure it.

    python jevbench_run/reason_server.py [--port 8766] [--ckpt checkpoints_s2/best]

Reason writes its reasoning inside <think>, as it was trained to, then the answer is read as a distribution: the options
are lettered A, B, C... and after "</think>```" one forward pass gives the model's probability for each letter, so
every option gets a probability (JevBench scores calibration). {"think": false} in the request skips the reasoning.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import train_s2 as s2  # noqa: E402 - also sets the project's attention backends (model.py)

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


def state_text(state):
    return state if isinstance(state, str) else s2.render_state(state)


@torch.no_grad()
def decide(m, proc, state, q, think=True, max_new=384):
    opts = options(q)
    body = (f"{state_text(state)}\n\nQuestion: {q.get('instructions', '')}\nOptions:\n"
            + "\n".join(f"{LETTERS[i]}. {text}" for i, (_, text) in enumerate(opts)))
    prompt = f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n<think>\n"
    tok = proc.tokenizer
    thought = ""
    if think:
        enc = {"input_ids": tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()}
        thought = s2.generate(m, proc, enc, max_new).split("</think>")[0].strip()
    ids = tok(prompt + thought + "\n</think>\n\n```", add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    n = ids.shape[1]
    pad = -n % 256  # one cuDNN plan per 256-token bucket, as in generate(); causal, so padding can't change the answer
    s2.base_of(m).model.rope_deltas = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(input_ids=torch.cat([ids, ids.new_zeros((1, pad))], 1), logits_to_keep=pad + 1).logits[0, 0].float()
    letter_ids = [tok.convert_tokens_to_ids(LETTERS[i]) for i in range(len(opts))]
    p = torch.softmax(logits[letter_ids], -1).tolist()
    probs = {k: round(v, 4) for (k, _), v in zip(opts, p)}
    out_tokens = len(tok(thought, add_special_tokens=False).input_ids) if thought else 0
    best = max(probs, key=probs.get)
    t = q.get("type", "choice")
    if t == "noul":
        ans = {"type": "noul", "noul": probs["yes"]}
    elif t == "score":
        ans = {"type": "score", "score": round(sum(int(k) * v for k, v in probs.items()), 3), "probabilities": probs}
    else:
        ans = {"type": "choice", "choice": best, "probabilities": probs}
    return ans, len(tok(prompt, add_special_tokens=False).input_ids), out_tokens, thought  # input = the prompt, once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--ckpt", default="checkpoints_s2/best")
    a = ap.parse_args()
    import uvicorn
    from fastapi import FastAPI
    m, proc = s2.load(a.ckpt, merge=True)
    app = FastAPI(title="reflex-reason-2b")

    @app.post("/v1/systemone")
    def systemone(req: dict):
        answers, n_in, n_out = {}, 0, 0
        for name, q in req["questions"].items():
            ans, i, o, _ = decide(m, proc, req["state"], q, think=req.get("think", True))
            answers[name] = ans
            n_in, n_out = n_in + i, n_out + o
        return {"model": "reflex-reason-2b", "answers": answers, "usage": {"input_tokens": n_in, "output_tokens": n_out}}

    uvicorn.run(app, host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
