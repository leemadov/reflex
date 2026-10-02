# Computer-Desicion: a local Jev-style decision model for computer use

A small **decision model** in the style of TypeSafe's Jev ("System One"): it takes a text **state** and
typed **questions**, and returns calibrated probabilities, never generated text. For computer use the state is the
task, the action history and the page's **accessibility tree**. The model picks the next **operation** and the
**element** it acts on.

- Backbone: `Qwen/Qwen3-0.6B` + LoRA (r=64), plus a 2x256-d pointer head (~40M trainable parameters).
- Every answer option is a position in one input sequence: a text option is the last token of its line in the
  question block; an element option is the last token of that element's line **inside the page**, so elements are
  never copied. `logit = <Wq h_question, Wk h_option>`; softmax over the question's options; temperature-calibrated.
- Answer types are Jev's three: `choice` (up to all elements on a page), `noul` (yes/no probability),
  `score` (ordered levels -> expected level).

## Data

| Dataset | Steps (train / test) | What |
|---|---|---|
| [stanfordnlp/nnetnav-wa](https://huggingface.co/datasets/stanfordnlp/nnetnav-wa) | 44,431 / 4,924 | WebArena's self-hosted sites (shop, shop admin, GitLab, Reddit clone, maps) |
| [stanfordnlp/nnetnav-live](https://huggingface.co/datasets/stanfordnlp/nnetnav-live) | 45,809 / 5,050 | Live websites (ESPN, OpenStreetMap, dictionaries, ...) |

Both are Apache-2.0, from [NNetNav](https://arxiv.org/abs/2410.02907): an LLM explores websites, and its trajectories
are relabeled with instructions. Training uses only the train splits.

`prepare.py` strips per-link URLs, icon-font glyphs and StaticText that repeats its parent's name (a live-web page
goes from a median of 11k to 4.5k tokens), and drops malformed agent outputs. The same cleanup runs at serve time.
[Mind2Web](https://huggingface.co/datasets/osunlp/Mind2Web) is downloaded to `data/raw/mind2web` but not used yet.

Operations learned: `CLICK, TYPE_TEXT, HOVER, PRESS_KEY, SCROLL_DOWN, SCROLL_UP, GOTO_URL, GO_BACK, GO_FORWARD,
SWITCH_TAB, DONE, BLOCKED`. Training randomizes option names, descriptions, order and subsets, and question names and
order, so a caller can pass its own op list (for example Jev's `CLICK, TYPE_TEXT, SELECT, SCROLL, WAIT, DONE, BLOCKED`).

## Results

Measured on 2,000 held-out test steps, disjoint from the 400 used to pick the checkpoint (`final_eval.log`):

| | Model | Reference |
|---|---|---|
| Operation, top-1 / top-5 | **62.1%** / 95.2% | 56.0% always CLICK; 61% TF-IDF + logistic regression trained on 20k steps |
| Target element, top-1 / top-5 | **41.6%** / 70.2% | ~0.3% random guess (about 300 elements per page) |
| Step: operation and element both right | **31.0%** (WebArena 32.0%, live web 30.0%) | |
| Calibration error (ECE, 15 bins) | **0.025** | fitted temperature came out at 1.0 |
| Decisions with p >= 0.9 | operation: 6.3% of steps, 90.4% right; element: 6.7% of steps, 95.1% right | |
| Latency, RTX 5070 | 162 ms p50 / 452 ms p90 in-process; 365 ms p50 over HTTP on 6k-token pages | Jev: ~100 ms |

How to read this: the labels come from an LLM explorer, so a step often has several reasonable next actions, and
the logged one is only one of them. Even a classifier trained on 20k steps gets just 61% of operations right. Top-1
against these labels understates usefulness; the calibrated probabilities are the point. Act automatically above a
high confidence, and hand low-confidence steps to a bigger model or to the user. A probability of 0.9 means about 90%.

Training: 2,000 steps x 8 sequences, i.e. 16k steps, 18% of one epoch. It took 4 h 20 min at about 4.9k tokens/s
with a 4 GB peak. Accuracy was still rising at the end (`train.log`), so a longer run should help.

## Is it a System One model?

"System One" is TypeSafe's name for Jev's kind of model: it decides instead of writing. `verify.py` measures each
property on held-out pages (`verify.log`):

| Property | Measured |
|---|---|
| Decides, never writes | No language-model head at all (bare `Qwen3Model`); the only output layer scores the supplied options |
| Answers only from the supplied options | 1,200 answers to random choice/noul/score requests (2 to 600+ options, unseen option names): 0 violations |
| One forward pass per request | exactly 1 backbone pass for 4 questions; no token-by-token decoding |
| Fast | 191 ms p50, 605 ms p90 per 4-question request (Jev: 70-500 ms) |
| Jev's API contract | state as string/object/array, 255-option choices, Jev's answer fields, deterministic |
| Calibrated | ECE 0.050 on 511 decisions: "0.89" is right 94% of the time, "0.49" right 51% |
| System 1 vs System 2 | 0.19 s in one pass, vs 14 s and 380 generated tokens for the same-size LLM reasoning it out in text |

Where it differs from Jev: questions share one pass instead of being answered separately. Asking alone or in
another order moves probabilities by a median of 0.001 and changed 2 of 240 answers. Calibration comes from
temperature scaling, not TypeSafe's RL method. It is a 0.6B model specialized for web computer use, and handles
16k tokens of context, not 64k.

## Use

```bash
python prepare.py                                   # data/raw -> data/train.jsonl, data/test.jsonl
python train.py --check                             # question presence must not leak the answer
python train.py --steps 2000 --accum 8              # ~4.3 h on an RTX 5070, writes checkpoints/best
python train.py --eval-only --ckpt checkpoints/best --n-val 2000
python model.py serve --ckpt checkpoints/best       # http://127.0.0.1:8765/v1/systemone (46 s warm-up)
python model.py check                               # tokenizer/pointer self-check
python verify.py                                    # System One property checks, ~5 min (verify.log)
```

Request (same shape as Jev's API; `"criteria": "elements"` means "every `[id]` element in the state"):

```json
{
  "state": {
    "task": "Find reviews for a product in the Electronics category.",
    "url": "http://shop.local/",
    "history": [],
    "page": "RootWebArea 'One Stop Market'\n\t[227] link 'My Account'\n\t[815] menuitem 'Electronics'\n\t[272] combobox 'Search'"
  },
  "questions": {
    "operation": {"type": "choice", "instructions": "What is the next operation?",
                  "criteria": {"CLICK": "click an element", "TYPE_TEXT": "type text into a field",
                               "SCROLL": "scroll the page", "DONE": "the task is complete"}},
    "target": {"type": "choice", "instructions": "Which element should the next operation act on?",
               "criteria": "elements"},
    "done": {"type": "noul", "instructions": "Is the task already accomplished?"}
  }
}
```

Actual response:

```json
{"model": "cd-0.6b",
 "answers": {"operation": {"type": "choice", "choice": "CLICK", "confidence": 0.368,
                           "probabilities": {"CLICK": 0.5216, "TYPE_TEXT": 0.4281, "SCROLL": 0.0417, "DONE": 0.0086}},
             "target": {"type": "choice", "choice": "272", "confidence": 0.465,
                        "probabilities": {"227": 0.005, "815": 0.2463, "272": 0.7487}},
             "done": {"type": "noul", "noul": 0.0061}},
 "usage": {"input_tokens": 158, "output_tokens": 0}}
```

On a three-element toy page it hesitates between searching (272) and the category menu (815). That split shows
up in the probabilities instead of being hidden. `confidence` is `1 - normalized entropy`.

The page tree should look like WebArena/BrowserGym accessibility trees (`[id] role 'name' props`, tab-indented),
because that is what the model was trained on.

The server also has `POST /v1/text` (`{"prompt", "max_tokens"}` -> `{"text"}`): a one-line completion from the
plain base Qwen3-0.6B. The decision model never writes text; the browser agent uses this endpoint for what to type
and for the final answer, the way Jev hands writing to a small LLM.

## Browser agent extension (`extension/`)

A Chromium extension (Opera, Chrome, Edge) that runs the model as an agent on your current tab.

Each step:

1. Read the tab's accessibility tree through the browser's debugger API and print it the way BrowserGym does,
   the format the model was trained on.
2. Ask the model for the operation and the element.
3. Act with real mouse and keyboard events.

When the model's confidence (`p(operation) x p(element)`) is below your threshold, the popup shows its top
choices, the target is outlined on the page, and nothing happens until you click **Do it**. You can pick a
different operation or element, or edit the text to type.

A glowing blue cursor (`cursor.js`) glides along a soft arc to each target, leaving a short light trail, and the
target gets a blue glow. The cursor ripples on click and fades out when the run ends. It is decoration only:

- It's hidden from the accessibility tree, so the model's view of the page is unchanged.
- It ignores the mouse, so real clicks land underneath.
- It works on pages with strict security policies.
- It jumps instead of gliding if the system's reduced-motion setting is on.

Install:

1. Start the model: `python model.py serve --ckpt checkpoints/best`
2. Open `opera://extensions`, `chrome://extensions` or `edge://extensions` and turn on Developer mode.
3. Click **Load unpacked** and pick the `extension` folder.

Use: open a page, click the extension's icon, describe the task, and click **Run on this tab**. While it runs, the
browser shows a "started debugging this browser" bar; closing that bar or pressing **Stop** ends the run.

Settings (in the popup):

- **Model server**: default `http://127.0.0.1:8765`.
- **Max steps**: default 15.
- **Act without asking when confidence ≥**: default 0.9, so almost every step waits for you; 0 means fully
  automatic.

Safety:

- It refuses to type into password fields.
- It never repeats the same action without asking. The model rarely notices on its own that it's finished, and
  will press "Add to cart" again and again.
- It talks only to the local server.
- It acts inside your real, logged-in browser, so keep the default threshold on sites where a click can buy, send
  or delete something.

Tested end to end in Chrome for Testing 153 against the local demo shop from `computer-use-agent`
(`python apps/server.py`), fully automatic (threshold 0), with repeat pauses answered "done":

| Task | Result |
|---|---|
| Search for "Coffee Mug" / "Notebook" and add it to your cart | ✅ typed the name, clicked the one remaining button, stopped |
| Filter the catalog to Electronics / Apparel | ✅ one click |
| Add "Running Shoes" to your wishlist | ✅ but it also clicked *Add to cart* before stopping |
| Add "Wireless Mouse" / "Desk Lamp" / "Backpack" to your cart (no search) | ❌ clicked the first of 19 identical *Add to cart* buttons |

That's 5 of 8 with no help. `agent_demo.mp4` is a 67-second screen recording of four of these tasks in a visible
browser window (1600x1000 instead of 1280x800). There the Electronics filter *failed*: the model typed
"Electronics" into the search box. Its decisions shift with page layout. The final line the popup shows is
labelled "Summary (unverified)", because the small LLM that writes it claimed success on the failed runs.

The failures are the model's: it can't yet tie a button to the product name a few lines
above it. Searching first sidesteps this, and in confirm mode the popup labels duplicates ("button 'Add to cart'
(5 of 19)") next to the highlight on the page. Other limits: only the main frame (no iframes), one tab at a time,
and no `SELECT`, `GOTO_URL` or tab operations.

## Reflex desktop app (`desktop/`)

A Windows app (Electron) with a built-in browser, the chat panel and the local models in one window, in the ad's
pink design. Launch it with the **Reflex** shortcut on the desktop, `Reflex.cmd`, or `cd desktop && npm start`.

- **One agent, two hosts.** The app runs the extension's own `background.js` against a small `chrome.*` shim
  (`desktop/agent-host.js`), and its chat is the extension's own `popup.html` (`desktop/preload-chat.js`). A fix
  in either file lands in both.
- **The models are managed for you.** If no model server is running, the app starts `python model.py serve
  --preload` and stops it on quit. System 2 and the writer load and compile their GPU kernels in the background. The
  top bar shows their status.
- **Fast / Balanced / Best** sets the System 2 cutoff (0 / 0.2 / 0.3) and the step budget. Balanced is the
  eval-backed default (`eval_route.py`). Best is slower and, on average, no more accurate.
- **Speed work:**
  - The page wait watches for the DOM to go quiet instead of sleeping a fixed 0.8 s.
  - The browser connection stays attached between tasks.
  - The cursor glides faster in Fast/Balanced.
  - One GPU lock serialises every model call. PyTorch's attention-backend switch is process-global, and a concurrent
    System 1 warm-up once ran on the math kernel and needed 9 GB.
- **Quality work:** the final answer is read off a screenshot of the final page by the vision model (`/v1/summarize`).
  The app also keeps rendering when its window is covered, because hidden windows could drop the agent's clicks.
- **Measured on the demo shop (`REFLEX_SELFTEST=<dir>`), from a cold start:** models ready in about 31 s. "Add Wool
  Socks to the cart" took 7.4 s and the follow-up "now add a backpack too" took 3.4 s. With warm models the times are
  4.8 s and 2.0 s.

## Limits

- Text only: it reads accessibility trees, not screenshots. Canvas apps and unlabeled icons are invisible to it.
- It chooses and never writes. The text for `TYPE_TEXT` and the answer for `DONE` must come from elsewhere, such as
  the task string or a small LLM. Jev works the same way.
- It was trained on web pages only. Windows desktop UI Automation trees are a different format and are untested.
- The labels are noisy and LLM-generated (see Results). `SELECT` and `WAIT` never appear as correct answers in the
  data. Yes/no and score questions other than the handful used in training work zero-shot and are unvalidated.
- Windows note: this PyTorch build has no FlashAttention, so `model.py` forces cuDNN attention globally and pads
  inputs to multiples of 256 tokens. Remove both if you move to Linux with flash-attn.
