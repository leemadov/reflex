"""System 2: LoRA on Qwen3-VL-2B to look at the screenshot and the page, think step by step, then emit one action.

Text steps come from NNetNav (data/*.jsonl, `why` = its step-by-step reasoning), screenshot steps from
Multimodal-Mind2Web (data/m2w_*.jsonl, prepare_m2w.py). The answer is "{why}\n</think>\n\n```{action}```" after a
prompt ending in "<think>\n"; loss on the answer only. Saves every 100 steps; --resume continues a stopped run.

    python train_s2.py --hours 10 [--resume]             -> checkpoints_s2/{last,best}
    python train_s2.py --eval-only --ckpt checkpoints_s2/best
    python train_s2.py --check                            # format round-trip + prompt lengths (CPU)
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import model as s1  # noqa: F401 - sets the cuDNN-only attention backend (see model.py)
from model import render_state
from prepare import parse_action
from train import read

BASE2 = "Qwen/Qwen3-VL-2B-Instruct"
IMG = (1024, 640)  # every screenshot is resized to this: 640 image tokens, one cuDNN plan for the vision tower
SYSTEM = ("You are a web agent. Given the task, the URL, your previous actions, the page's accessibility tree and "
          "possibly a screenshot, think step by step about what to do next, then give exactly one action in triple "
          "backticks. Actions: click [id], type [id] [text], select [id] [option], hover [id], press [key_comb], "
          "scroll [down|up], goto [url], go_back, go_forward, new_tab, tab_focus [index], close_tab, "
          "stop [answer] (stop [N/A] if the task is impossible).")
HEAD = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
VISION = "<|vision_start|><|image_pad|><|vision_end|>\n"
TAIL = "<|im_end|>\n<|im_start|>assistant\n<think>\n"
PLAN_HEAD = ("<|im_start|>system\nYou split a web task into steps for a web agent. Rules: use as few steps as possible. "
             "Most tasks are ONE step: then repeat the task as the single step. Split only when the task itself names "
             "several separate things to do (joined by 'and', 'then', commas). Never invent steps such as searching, "
             "comparing or picking options unless the task says so. One numbered step per line.<|im_end|>\n")
PLAN_SHOTS = [("Add a Desk Lamp to the cart", "1. Add a Desk Lamp to the cart"),  # tuned: 10/10 step counts on a mixed set
              ("Find the population of Lyon", "1. Find the population of Lyon"),
              ("Show only Home products, then add the Desk Lamp to the cart",
               "1. Show only Home products\n2. Add the Desk Lamp to the cart"),
              ("Book a table for two at Nobu tonight", "1. Book a table for two at Nobu tonight"),
              ("Add a Coffee Mug and a Notebook to the cart", "1. Add a Coffee Mug to the cart\n2. Add a Notebook to the cart"),
              ("Search for \"usb hub\", sort by rating, then open the top result",
               "1. Search for \"usb hub\"\n2. Sort the results by rating\n3. Open the top result")]


def action_text(r):
    op, t, a = r["op"], r["target"], r["arg"]
    return {"CLICK": f"click [{t}]", "TYPE_TEXT": f"type [{t}] [{a or ''}]", "SELECT": f"select [{t}] [{a or ''}]",
            "HOVER": f"hover [{t}]", "PRESS_KEY": f"press [{a}]", "SCROLL_DOWN": "scroll [down]", "SCROLL_UP": "scroll [up]",
            "GOTO_URL": f"goto [{a}]", "GO_BACK": "go_back", "GO_FORWARD": "go_forward",
            "SWITCH_TAB": f"tab_focus [{a}]" if a else "new_tab", "DONE": f"stop [{a or ''}]", "BLOCKED": "stop [N/A]"}[op]


def state_of(r):
    return {"task": r["task"], "url": r["url"], "history": r["history"], "page": r["page"]}


def screenshot(img):
    """Path / PIL image / None -> RGB image at IMG size (the browser window, as the extension captures it)."""
    if img is None:
        return None
    im = (Image.open(img) if isinstance(img, (str, Path)) else img).convert("RGB")
    if im.size == IMG:
        return im
    # a live window can be any shape: fit inside IMG without stretching, black bars for the rest
    s = min(IMG[0] / im.width, IMG[1] / im.height)
    canvas = Image.new("RGB", IMG)
    canvas.paste(im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BICUBIC), (0, 0))
    return canvas


def encode(proc, state, image, max_prompt, need=None):
    """Chat prompt up to the opening <think> -> model kwargs (on cuda). Trims whole page lines to fit; None if that
    trimmed away element `need` (a training target the model could no longer see)."""
    tok = proc.tokenizer
    fixed = len(tok(HEAD + TAIL, add_special_tokens=False).input_ids) + (IMG[0] * IMG[1] // 32 ** 2 + 3 if image else 0)
    text = render_state(state)
    body = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    budget = max_prompt - fixed
    if len(body.input_ids) > budget:
        cut = text.rfind("\n", 0, body.offset_mapping[budget][0])
        text = text[:max(cut, 0)] + "\n[page truncated]"
    if need and f"[{need}]" not in text:
        return None
    prompt = HEAD + (VISION if image else "") + text + TAIL
    if image:
        enc = proc(text=[prompt], images=[image], return_tensors="pt")
        enc.pop("attention_mask", None)
    else:
        enc = {"input_ids": tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids}
    return {k: v.cuda() for k, v in enc.items()}


def example(proc, r, max_len, max_answer=384):
    ans = proc.tokenizer(r["why"] + "\n</think>\n\n```" + action_text(r) + "```<|im_end|>", add_special_tokens=False).input_ids
    ans = ans[-max_answer:]  # very long reasoning: keep its end, which holds the decision
    enc = encode(proc, state_of(r), screenshot(r.get("image")), max_len - len(ans), need=r["target"])
    return (enc, ans) if enc else None


def base_of(m):
    return m.get_base_model() if hasattr(m, "get_base_model") else m  # LoRA layers are injected in place


def loss_of(m, enc, a):
    b = base_of(m)
    p = enc["input_ids"].shape[1]
    ids = torch.cat([enc["input_ids"], torch.tensor([a], device="cuda")], 1)
    extra = {k: v for k, v in enc.items() if k != "input_ids"}
    if "mm_token_type_ids" in extra:  # image steps: the answer tokens are text (type 0)
        extra["mm_token_type_ids"] = torch.cat([extra["mm_token_type_ids"], torch.zeros_like(ids[:, p:])], 1)
    b.model.rope_deltas = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = b.model(input_ids=ids, **extra).last_hidden_state[0, p - 1:-1]  # logits only where the answer is
        logits = b.lm_head(h).float()
    return F.cross_entropy(logits, torch.tensor(a, device="cuda"))


@torch.no_grad()
def generate(m, proc, enc, max_new=384):
    """Greedy decode with a cache and no attention mask (batch 1). Prefill on cuDNN; decode steps on the math kernel,
    since cuDNN builds a plan per shape (~0.3 s) and the cache grows every token."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import DynamicCache
    base_of(m).model.rope_deltas = None  # stale M-RoPE offsets from a previous image prompt would shift positions
    end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Prefill padded to a multiple of 256 so cuDNN reuses its plans (the math fallback on a ~7k prompt needs GBs and
    # Windows silently spills that to system RAM: 60 s). Causal attention: real tokens never see the padding, and the
    # cache is cropped back to the real length before decoding. M-RoPE offsets are unchanged (padding is text).
    n = enc["input_ids"].shape[1]
    pad = -n % 256
    kw = {k: (torch.cat([v, v.new_zeros((1, pad))], 1) if k in ("input_ids", "mm_token_type_ids") else v) for k, v in enc.items()}
    cache, out = DynamicCache(), []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(max_new):
            with sdpa_kernel([SDPBackend.CUDNN_ATTENTION] if i == 0 else [SDPBackend.MATH]):
                o = m(**kw, past_key_values=cache, use_cache=True, logits_to_keep=pad + 1 if i == 0 else 1)
            if i == 0:
                cache.crop(n)
            nxt = int(o.logits[0, 0 if i == 0 else -1].argmax())
            if nxt == end:
                break
            out.append(nxt)
            kw = {"input_ids": torch.tensor([[nxt]], device="cuda")}
    return proc.tokenizer.decode(out)


def think(m, proc, state, image=None, max_len=7168, max_new=384):
    """-> (reasoning + action text, parsed action (op, target, arg) or None)"""
    text = generate(m, proc, encode(proc, state, screenshot(image), max_len - max_new), max_new)
    return text, parse_action(text)


RESOLVE_HEAD = ("<|im_start|>system\nYou rewrite the user's latest message to a web agent as ONE standalone instruction, "
                "using the earlier conversation to fill in what it refers to (items, dates, sites, numbers). Keep it short "
                "and keep the user's wording where you can. If the message is already standalone, repeat it unchanged."
                "<|im_end|>\n")
RESOLVE_SHOTS = [  # tuned on follow-ups: quantities, "same for X", changing one field of an earlier task, unrelated new tasks
    ('User: Add a Coffee Mug to the cart.\nAgent: Did "Add a Coffee Mug to the cart" (done)', "add two more",
     "Add 2 more Coffee Mugs to the cart"),
    ('User: Search for flights from Athens to London on October 4\nAgent: Did "Search for flights from Athens to London on '
     'October 4" (done)', "make it the 6th instead", "Search for flights from Athens to London on October 6"),
    ('User: Book a hotel in Rome from May 3 to May 5\nAgent: Did "Book a hotel in Rome from May 3 to May 5" (stopped)',
     "change the checkout to the 9th", "Book a hotel in Rome from May 3 to May 9"),
    ('User: Show only Home products\nAgent: Did "Show only Home products" (done)', "do the same for Electronics",
     "Show only Electronics products"),
    ('User: Show only Electronics products\nAgent: Did "Show only Electronics products" (done)', "now sort them by price",
     "Sort the Electronics products by price"),
    ('User: Add a Desk Lamp to the cart\nAgent: Did "Add a Desk Lamp to the cart" (done)', "Search for the weather in Athens",
     "Search for the weather in Athens"),
]


def _chat(head, shots, user):
    p = head + "".join(f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n" for q, a in shots)
    return p + f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def _base(m, proc, prompt, max_new):
    enc = {"input_ids": proc.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.cuda()}
    with (m.disable_adapter() if hasattr(m, "disable_adapter") else torch.no_grad()):  # base model: LoRA off
        return generate(m, proc, enc, max_new)


def resolve(m, proc, task, context, max_new=60):
    """Follow-up + recent turns [{role, text}] -> one standalone task (the base model, few-shot)."""
    if not context:
        return task
    convo = "\n".join(f"{'User' if t['role'] == 'user' else 'Agent'}: {t['text']}" for t in context[-6:])
    shots = [(f"Conversation:\n{c}\n\nLatest message: {q}", a) for c, q, a in RESOLVE_SHOTS]
    out = _base(m, proc, _chat(RESOLVE_HEAD, shots, f"Conversation:\n{convo}\n\nLatest message: {task}"), max_new)
    out = out.strip().split("\n")[0].strip().strip('"')
    return task if not out or out.startswith("Did ") or "(done)" in out else out  # an echo of the transcript is no rewrite


def summarize(m, proc, task, image, max_new=60):
    """One sentence on how the task ended, read off a screenshot of the final page (base model, LoRA off)."""
    if image is None:
        return ""
    p = ("<|im_start|>user\n" + VISION + f"A web agent was given this task: {task}\nThe screenshot shows the page after it "
         "finished. In one short sentence, report the result: the answer if the task asked for information, otherwise "
         "what is now done. If the screenshot shows the task is not done, say so.<|im_end|>\n<|im_start|>assistant\n")
    enc = {k: v.cuda() for k, v in proc(text=[p], images=[screenshot(image)], return_tensors="pt").items() if k != "attention_mask"}
    with (m.disable_adapter() if hasattr(m, "disable_adapter") else torch.no_grad()):
        return generate(m, proc, enc, max_new).strip().split("\n")[0]


def plan(m, proc, task, max_new=160):
    """Short numbered plan from the base model (LoRA off: planning isn't what the adapter learned). -> [steps]"""
    text = _base(m, proc, _chat(PLAN_HEAD, PLAN_SHOTS, task), max_new)
    steps = [s.split(".", 1)[1].strip() for s in text.split("\n") if s.strip()[:1].isdigit() and "." in s]
    return steps[:6] or [task]


def evaluate(m, proc, recs, preds_path=None):
    m.eval()
    rows = []
    for r in recs:
        t0 = time.perf_counter()
        text, got = think(m, proc, state_of(r), r.get("image"))
        got = got or (None, None, None)
        rows.append({"src": r["src"], "op": r["op"], "target": r["target"], "pred_op": got[0], "pred_target": got[1],
                     "pred_arg": got[2], "s": round(time.perf_counter() - t0, 2), "think": text,
                     "ok": got[0] == r["op"] and (r["target"] is None or got[1] == r["target"])})
    m.train()
    if preds_path:
        with open(preds_path, "w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
    out = {}
    for src in sorted({x["src"] for x in rows}):
        part = [x for x in rows if x["src"] == src]
        tg = [x for x in part if x["target"] is not None]
        lat = sorted(x["s"] for x in part)
        out[src] = {"n": len(part), "op_acc": round(sum(x["pred_op"] == x["op"] for x in part) / len(part), 4),
                    "target_acc": round(sum(x["pred_target"] == x["target"] for x in tg) / max(len(tg), 1), 4),
                    "step_acc": round(sum(x["ok"] for x in part) / len(part), 4), "latency_s_p50": lat[len(lat) // 2]}
    return out


def eval_sets(n_text, n_img):
    return read("data/test.jsonl", n_text, seed=1, exclude=400) + read("data/m2w_test.jsonl", n_img, seed=1)


def load(ckpt, merge=False):
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(BASE2)
    m = Qwen3VLForConditionalGeneration.from_pretrained(BASE2, dtype=torch.bfloat16, attn_implementation="sdpa")
    if ckpt:
        m = PeftModel.from_pretrained(m, ckpt)
        m = m.merge_and_unload() if merge else m
    return m.cuda().eval(), proc


def check(n=3000):
    """The answer format must parse back to the gold action, and prompts must fit (CPU only)."""
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(BASE2)
    recs = read("data/test.jsonl", n) + read("data/m2w_test.jsonl", n // 3)
    bad = [r for r in recs if (parse_action(f"```{action_text(r)}```") or (None, None))[:2] != (r["op"], r["target"])]
    assert len(bad) < 0.02 * len(recs), (len(bad), [(r["op"], action_text(r)) for r in bad[:5]])
    tok = proc.tokenizer
    for r in recs[:50] + recs[-50:]:
        text = render_state(state_of(r))
        assert len(tok(HEAD + TAIL + text).input_ids) > 0
    print(f"check ok: {len(bad)}/{len(recs)} actions don't round-trip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=7168)
    ap.add_argument("--img-frac", type=float, default=0.3, help="share of training steps drawn from Mind2Web screenshots")
    ap.add_argument("--n-text", type=int, default=300)
    ap.add_argument("--n-img", type=int, default=200)
    ap.add_argument("--out", default="checkpoints_s2")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        return check()
    out = Path(a.out)

    if a.eval_only:
        m, proc = load(a.ckpt, merge=True)
        print("eval (held out)", json.dumps(evaluate(m, proc, eval_sets(a.n_text, a.n_img), out / "preds.jsonl")), flush=True)
        return

    from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.manual_seed(0)
    rng = random.Random(0)
    proc = AutoProcessor.from_pretrained(BASE2)
    m = Qwen3VLForConditionalGeneration.from_pretrained(BASE2, dtype=torch.bfloat16, attn_implementation="sdpa")
    m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m.enable_input_require_grads()
    m = get_peft_model(m, LoraConfig(r=32, lora_alpha=64, target_modules=[  # language model only; vision tower frozen
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])).cuda().train()
    lora = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora, lr=a.lr, weight_decay=0.0)
    step, done_s, run = 0, 0.0, []
    if a.resume and (out / "last" / "state.pt").exists():
        set_peft_model_state_dict(m, load_file(out / "last" / "adapter_model.safetensors"))
        st = torch.load(out / "last" / "state.pt", weights_only=False)
        opt.load_state_dict(st["opt"])
        step, done_s, run = st["step"], st["elapsed"], st["run"]
        rng.setstate(st["rng"])
        print(f"resumed at step {step}, {done_s / 3600:.2f}h done", flush=True)
    text = [r for r in read("data/train.jsonl") if r.get("why")]
    imgs = read("data/m2w_train.jsonl")
    budget = a.hours * 3600
    print(f"train {len(text)} text + {len(imgs)} screenshot records; lora {sum(p.numel() for p in lora) / 1e6:.1f}M; "
          f"budget {a.hours}h", flush=True)

    t0 = time.time() - done_s
    while time.time() - t0 < budget:
        step += 1
        frac = min(1.0, (time.time() - t0) / budget)  # cosine on wall-clock, so the time budget ends the schedule
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 50) * (0.1 + 0.45 * (1 + math.cos(math.pi * frac)))
        for _ in range(a.accum):
            ex = None
            while ex is None:  # skip the rare step whose target was trimmed off a very long page
                ex = example(proc, rng.choice(imgs) if rng.random() < a.img_frac else rng.choice(text), a.max_len)
            loss = loss_of(m, *ex)
            (loss / a.accum).backward()
            run.append(loss.item())
        torch.nn.utils.clip_grad_norm_(lora, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            dt = time.time() - t0
            print(f"step {step} loss {sum(run[-80:]) / len(run[-80:]):.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"elapsed {dt / 60:.1f}m mem {torch.cuda.max_memory_allocated() / 2**30:.1f}G", flush=True)
        if step % a.save_every == 0:
            m.save_pretrained(out / "last")
            torch.save({"opt": opt.state_dict(), "step": step, "elapsed": time.time() - t0, "run": run[-200:],
                        "rng": rng.getstate()}, out / "last" / "state.pt")
    m.save_pretrained(out / "best")
    del opt  # give the eval the optimizer's memory: near the 12 GB limit Windows spills to system RAM and crawls
    torch.cuda.empty_cache()
    metrics = {**evaluate(m, proc, eval_sets(a.n_text, a.n_img), out / "best" / "preds.jsonl"),
               "step": step, "hours": round((time.time() - t0) / 3600, 2)}
    (out / "best" / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("eval (held out)", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
