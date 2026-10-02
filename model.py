"""Jev-style decision model for computer use: state + typed questions -> calibrated probabilities.

Backbone: Qwen3-0.6B + LoRA. Every answer option is a position in one input sequence:
  - text option  "- CLICK: click an element"            -> last token of that option line
  - element      "[815] link 'Electronics'" in the page -> last token of that page line (no copying)
A question's query is the last token of its block; logit = <Wq h_query, Wk h_option> / sqrt(d).
noul = 2-option choice, score = k-level choice (+ expected level), same three answer types as Jev.

    python model.py check                      # offset/pointer self-check (no GPU needed)
    python model.py serve --ckpt checkpoints/best [--port 8765] [--ckpt2 checkpoints_s2/best]
        /v1/systemone (fast), /v1/systemtwo (screenshot + reasoning), /v1/plan, /v1/text
"""
import argparse
import bisect
import json
import math
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import compact

# Windows torch has no flash kernel and the mem-efficient one rejects GQA, so SDPA silently falls back to
# O(L^2) math attention (OOM at 8k tokens). cuDNN handles GQA + causal, 2.5x faster than repeat_kv + efficient.
# Set globally (not a context manager) so gradient-checkpoint recomputation picks the same kernel.
torch.backends.cuda.enable_cudnn_sdp(True)
for off in (torch.backends.cuda.enable_flash_sdp, torch.backends.cuda.enable_mem_efficient_sdp,
            torch.backends.cuda.enable_math_sdp):
    off(False)

BASE = "Qwen/Qwen3-0.6B"
NAME = "cd-0.6b"
NAME2 = "cd-s2-vl-2b"
ELEM = re.compile(r"^[ \t]*\[([^\]\s]+)\] .*$", re.M)


def render_state(state) -> str:
    if isinstance(state, str):
        return compact(state)
    if isinstance(state, list):
        return "\n".join(compact(x) if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in state)
    out = []
    for k, v in state.items():
        if isinstance(v, list):
            v = "\n".join("- " + (x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)) for x in v) or "none"
        elif not isinstance(v, str):
            v = json.dumps(v, ensure_ascii=False)
        v = compact(v)
        out.append(f"{k}:\n{v}" if "\n" in v else f"{k}: {v}")
    return "\n".join(out)


def options(q):
    """-> list of (key, description), or the string 'elements' for every [id] line of the state."""
    t, c = q.get("type", "choice"), q.get("criteria")
    if t == "noul":
        c = c or {}
        return [("true", c.get("true", "yes")), ("false", c.get("false", "no"))]
    if t == "score":
        return [(str(i), d) for i, d in enumerate(c)]
    if c is None or c == "elements":
        return "elements"
    if isinstance(c, list):
        return [(str(x), "") for x in c]
    return [(str(k), str(v)) for k, v in c.items()]


def encode(tok, state, questions, max_len):
    """-> (input_ids, specs); spec = {name, type, keys, qpos, kpos}. Truncates the state, never the questions."""
    s = render_state(state)
    elems = {m.group(1): m.end() - 1 for m in ELEM.finditer(s)}  # id -> char of its line's last character

    qtext, marks = "", []
    for name, q in questions.items():
        opts = options(q)
        # choice keys that are page ids point into the page; score/noul keys ("0", "true") never do
        pointer = opts == "elements" or (q.get("type", "choice") == "choice" and all(k in elems for k, _ in opts))
        qtext += f"\n\nquestion {name}: {q.get('instructions', '')}\noptions:"
        if pointer:
            keys = list(elems) if opts == "elements" else [k for k, _ in opts]
            qtext += " elements on the page"
            chars = None
        else:
            keys, chars = [], []
            for k, d in opts:
                qtext += f"\n- {k}: {d}" if d and d != k else f"\n- {k}"
                keys.append(k)
                chars.append(len(qtext) - 1)
        qtext += "\nanswer:"
        marks.append((name, q.get("type", "choice"), keys, chars, len(qtext) - 1))

    se = tok(s, add_special_tokens=False, return_offsets_mapping=True)
    qe = tok(qtext, add_special_tokens=False, return_offsets_mapping=True)
    ids, starts = se.input_ids, [a for a, _ in se.offset_mapping]
    qstarts = [a for a, _ in qe.offset_mapping]
    budget = max_len - len(qe.input_ids)
    if len(ids) > budget:  # cut whole page lines off the end
        cut = s.rfind("\n", 0, starts[budget])
        ids = ids[:bisect.bisect_left(starts, max(cut, 0))]
    n = len(ids)
    tpos = lambda st, c: bisect.bisect_right(st, c) - 1  # noqa: E731 - token holding char c

    specs = []
    for name, typ, keys, chars, qchar in marks:
        if chars is None:  # element pointers; drop the ones truncated away
            kept = [(k, tpos(starts, elems[k])) for k in keys if tpos(starts, elems[k]) < n]
            keys, kpos = [k for k, _ in kept], [p for _, p in kept]
        else:
            kpos = [n + tpos(qstarts, c) for c in chars]
        specs.append({"name": name, "type": typ, "keys": keys, "kpos": kpos, "qpos": n + tpos(qstarts, qchar)})
    return ids + qe.input_ids, specs


class Decider(nn.Module):
    def __init__(self, backbone, dim=256):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.q, self.k = nn.Linear(h, dim), nn.Linear(h, dim)
        self.register_buffer("T", torch.ones(()))  # calibration temperature, fitted after training

    def forward(self, ids, specs):
        dev = self.q.weight.device
        # right-pad to a multiple of 256: cuDNN builds a plan per shape (~0.9s each); causal attention means
        # real tokens never see the padding, so outputs are unchanged
        pad = [0] * (-len(ids) % 256)
        h = self.backbone(input_ids=torch.tensor([ids + pad], device=dev)).last_hidden_state[0]
        out = []
        for sp in specs:
            q = self.q(h[sp["qpos"]].float())
            k = self.k(h[torch.tensor(sp["kpos"], device=dev, dtype=torch.long)].float())
            out.append(k @ q / math.sqrt(q.numel()))
        return out

    @torch.no_grad()
    def answer(self, tok, state, questions, max_len=16384):
        ids, specs = encode(tok, state, questions, max_len)
        res = {}
        for sp, lg in zip(specs, self(ids, specs)):
            if not sp["keys"]:
                res[sp["name"]] = {"type": sp["type"], "error": "no options (all elements truncated?)"}
                continue
            p = F.softmax(lg / self.T, -1)
            probs = {k: round(v, 4) for k, v in zip(sp["keys"], p.tolist())}
            conf = 1.0 if len(p) < 2 else float(1 + (p * p.clamp_min(1e-12).log()).sum() / math.log(len(p)))
            if sp["type"] == "noul":
                res[sp["name"]] = {"type": "noul", "noul": probs["true"]}
            elif sp["type"] == "score":
                exp = float((p * torch.arange(len(p), device=p.device)).sum())
                res[sp["name"]] = {"type": "score", "score": round(exp, 3), "legend": dict(zip(sp["keys"], questions[sp["name"]]["criteria"])),
                                   "probabilities": probs, "confidence": round(conf, 3)}
            else:
                res[sp["name"]] = {"type": "choice", "choice": sp["keys"][int(p.argmax())], "probabilities": probs,
                                   "confidence": round(conf, 3)}
        return res, len(ids)


def head_state(m):
    return {k: v for k, v in m.state_dict().items() if not k.startswith("backbone.")}


def load(ckpt=None, device="cuda"):
    """Inference model: base + merged LoRA + head. ckpt=None gives an untrained head."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    bb = AutoModel.from_pretrained(BASE, dtype=torch.bfloat16, attn_implementation="sdpa")
    if ckpt:
        from peft import PeftModel
        bb = PeftModel.from_pretrained(bb, ckpt).merge_and_unload()
    m = Decider(bb)
    if ckpt:
        import os
        from safetensors.torch import load_file  # head.safetensors is the published form; head.pt is what train.py writes
        st = f"{ckpt}/head.safetensors"
        m.load_state_dict(load_file(st) if os.path.exists(st) else torch.load(f"{ckpt}/head.pt", map_location="cpu"), strict=False)
    return m.to(device).eval(), tok


def serve(ckpt, port, ckpt2=None, preload=False):
    import uvicorn
    from fastapi import FastAPI
    m, tok = load(ckpt)
    warm = {"done": 0, "total": 64}
    # One lock around every GPU call. PyTorch's attention-backend switch (sdpa_kernel) is process-global, not per
    # thread: System 2 decoding on the math kernel while another thread runs a 16k-token System 1 pass made that pass
    # use the math kernel too (a 9 GB allocation). The agent is sequential anyway, so serialising costs nothing.
    lock = __import__("threading").Lock()

    def build_plans():  # cuDNN plans for every 256-token bucket (~0.9 s each), common page sizes first, in the background:
        with torch.no_grad():  # requests are served between buckets and build any missing plan on demand
            for n in [*range(256, 8192 + 1, 256), *range(8448, 16384 + 1, 256)]:
                with lock:
                    m.backbone(input_ids=torch.zeros((1, n), dtype=torch.long, device=m.q.weight.device))
                warm["done"] += 1
    if not preload:  # with --preload, System 2's warm-up goes first (it is what makes a first task slow), then these
        __import__("threading").Thread(target=build_plans, daemon=True).start()
    app = FastAPI(title=NAME)

    @app.post("/v1/systemone")
    def systemone(req: dict):
        with lock:
            ans, n = m.answer(tok, req["state"], req["questions"])
        return {"model": NAME, "answers": ans, "usage": {"input_tokens": n, "output_tokens": 0}}

    gen, gen_lock = {}, __import__("threading").Lock()

    def writer():
        with gen_lock:
            if not gen:
                from transformers import AutoModelForCausalLM
                gen["m"] = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16).to(m.q.weight.device).eval()
        return gen["m"]

    @app.post("/v1/text")
    def text(req: dict):
        """The decision model never writes; like Jev, free text (what to type, the final answer) comes from a
        small LLM: here the plain base model, loaded on first use. Raw one-line completion of a few-shot prompt,
        because the chat template makes a 0.6B model answer in sentences instead of giving the bare text."""
        from torch.nn.attention import SDPBackend, sdpa_kernel
        w = writer()
        enc = tok(req["prompt"], return_tensors="pt").to(m.q.weight.device)
        # generate() passes an attention mask, which the global cuDNN-only setting can't take; allow the others here
        with lock, torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
            out = w.generate(**enc, max_new_tokens=int(req.get("max_tokens", 48)), do_sample=False)
        return {"text": tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip().split("\n")[0]}

    s2, s2_load = {}, __import__("threading").Lock()

    def system2():
        """Qwen3-VL-2B + System 2 LoRA, loaded on first use and kept unmerged so plan() can switch the LoRA off."""
        with s2_load:  # its own lock: loading weights doesn't need the GPU lock, so System 1 keeps answering meanwhile
            return _load_s2()

    def _load_s2():
        if not s2:
            import os
            import train_s2
            ckpt = next((c for c in (ckpt2, "checkpoints_s2/last") if c and os.path.exists(f"{c}/adapter_config.json")), None)
            print("System 2 checkpoint:", ckpt or "none (base model, untrained)", flush=True)
            s2["m"], s2["proc"] = train_s2.load(ckpt)
            s2["t"], s2["ckpt"] = train_s2, ckpt
        return s2

    def image(b64):
        if not b64:
            return None
        import base64
        import io
        from PIL import Image
        return Image.open(io.BytesIO(base64.b64decode(b64.split(",")[-1])))

    @app.post("/v1/systemtwo")
    def systemtwo(req: dict):
        """Deliberate step: look at the screenshot + page, think, pick one action. Slow (seconds), for when
        System 1 is unsure. -> thinking text + action {op, target, arg} (None if it wrote no valid action)"""
        with lock:
            s = system2()
            text, act = s["t"].think(s["m"], s["proc"], req["state"], image(req.get("screenshot")))
        thinking = text.split("</think>")[0].strip()
        return {"model": NAME2, "checkpoint": s["ckpt"], "thinking": thinking,
                "action": dict(zip(("op", "target", "arg"), act)) if act else None, "raw": text}

    @app.post("/v1/plan")
    def plan_(req: dict):
        """{task, context?: [{role, text}], plan?: bool} -> the task made standalone using the chat context, + its plan."""
        with lock:
            s = system2()
            task = s["t"].resolve(s["m"], s["proc"], req["task"], req.get("context") or [])
            steps = s["t"].plan(s["m"], s["proc"], task) if req.get("plan", True) else [task]
            return {"model": NAME2, "task": task, "steps": steps}

    @app.post("/v1/summarize")
    def summarize(req: dict):
        """{task, screenshot} -> one sentence on the result, read off the final page by the vision model (better than
        the 0.6B writer guessing from the history)."""
        with lock:
            s = system2()
            return {"model": NAME2, "text": s["t"].summarize(s["m"], s["proc"], req["task"], image(req.get("screenshot")))}

    @app.get("/v1/status")
    def status():
        return {"system1": True, "system2": bool(s2), "writer": bool(gen), "checkpoint2": s2.get("ckpt"),
                "warm": round(warm["done"] / warm["total"], 2)}

    if preload:  # the desktop app: load System 2 + the writer in the background now, so no request waits ~30 s for them
        def preload_models():
            from PIL import Image
            blank = Image.new("RGB", (1024, 640), "white")
            s = system2()
            with lock:  # one tiny pass through each path compiles the GPU kernels now, not on the user's first task
                s["t"].think(s["m"], s["proc"], {"task": "warm up", "url": "", "history": [], "page": "[1] button 'OK'"}, blank, max_new=8)
                s["t"].summarize(s["m"], s["proc"], "warm up", blank, max_new=4)
                s["t"].plan(s["m"], s["proc"], "warm up", max_new=8)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            w = writer()
            with lock, torch.no_grad(), sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
                w.generate(**tok("warm up", return_tensors="pt").to(m.q.weight.device), max_new_tokens=2, do_sample=False)
            print("READY system2", flush=True)
            build_plans()
        __import__("threading").Thread(target=preload_models, daemon=True).start()
    print("READY system1", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=port)  # ponytail: localhost only; put auth in front before exposing


def check():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE)
    page = "RootWebArea 'Shop'\n\t[12] link 'Home'\n\t[15] textbox 'Search'\n\t\tStaticText 'Search'\n\t[27] button 'Go'"
    state = {"task": "search for shoes", "history": [], "page": page}
    qs = {"op": {"type": "choice", "instructions": "Next operation?", "criteria": {"CLICK": "click", "TYPE_TEXT": "type"}},
          "target": {"type": "choice", "instructions": "Which element?", "criteria": "elements"},
          "sub": {"type": "choice", "instructions": "Which button?", "criteria": {"27": "", "12": ""}},
          "done": {"type": "noul", "instructions": "Is the task done?"}}
    ids, specs = encode(tok, state, qs, 4096)
    txt = lambda i: tok.decode(ids[i])  # noqa: E731
    sp = {s["name"]: s for s in specs}
    assert sp["target"]["keys"] == ["12", "15", "27"], sp["target"]
    assert [txt(i).strip() for i in sp["target"]["kpos"]] == ["'", "'", "'"]  # last token of each element line
    upto = lambda i: tok.decode(ids[:i + 1]).rstrip("\n")  # noqa: E731 - Qwen merges "'\n" into one token
    assert upto(sp["target"]["kpos"][1]).endswith("[15] textbox 'Search'")
    assert sp["sub"]["keys"] == ["27", "12"] and sp["sub"]["kpos"] == [sp["target"]["kpos"][2], sp["target"]["kpos"][0]]
    assert upto(sp["op"]["kpos"][1]).endswith("- TYPE_TEXT: type")
    assert all(txt(s["qpos"]).rstrip("\n").endswith(":") for s in specs) and sp["done"]["keys"] == ["true", "false"]
    ids2, specs2 = encode(tok, state, qs, len(ids) - 12)  # truncation drops trailing page lines, keeps questions
    t2 = {s["name"]: s for s in specs2}["target"]["keys"]
    assert len(ids2) <= len(ids) - 12 and t2 and "27" not in t2, t2
    print("check ok:", len(ids), "tokens")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "serve"])
    ap.add_argument("--ckpt", default="checkpoints/best")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--ckpt2", default="checkpoints_s2/best", help="System 2 LoRA")
    ap.add_argument("--preload", action="store_true", help="load System 2 and the writer at startup (background)")
    a = ap.parse_args()
    check() if a.cmd == "check" else serve(a.ckpt, a.port, a.ckpt2, a.preload)
