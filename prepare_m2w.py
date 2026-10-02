"""Multimodal-Mind2Web -> System 2 records with a screenshot (the browser window around the target).

    python prepare_m2w.py        -> data/m2w_train.jsonl, data/m2w_test.jsonl (5% of tasks), data/m2w_img/*.jpg

The page becomes the same tab-indented "[id] role 'name'" tree the extension sends (ids = backend_node_id), and
the full-page screenshot is cropped to a 1280x800 window holding the target, resized to train_s2.IMG.
Records match prepare.py's, plus "image"; "why" is a one-line statement of the gold step (Mind2Web has no
reasoning traces).
"""
import glob
import io
import json
import random
import re
import sys
from pathlib import Path

import lxml.html
import pyarrow.parquet as pq
from PIL import Image

SRC = "data/raw/mm-mind2web/data/train-*.parquet"
OUT, IMGS = Path("data"), Path("data/m2w_img")
IMG, WIN = (1024, 640), (1280, 800)
ROLE = {"a": "link", "button": "button", "select": "combobox", "option": "option", "textarea": "textbox", "img": "img",
        "label": "LabelText", "li": "listitem", "h1": "heading", "h2": "heading", "h3": "heading", "h4": "heading",
        "table": "table", "tr": "row", "td": "cell", "th": "columnheader", "nav": "navigation", "form": "form"}
INPUT = {"checkbox": "checkbox", "radio": "radio", "submit": "button", "button": "button", "search": "searchbox",
         "image": "button", "reset": "button"}
VERB = {"CLICK": "click", "TYPE": "type", "SELECT": "select"}


def py(s):
    s = re.sub(r"\s+", " ", s).strip()[:120]
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def tree(html):
    """cleaned_html -> (BrowserGym-style lines, ids present). Keeps elements with an id that have a name or are
    interactive, and text that no kept ancestor already shows."""
    root = lxml.html.fromstring(html)
    lines, ids = [], set()

    def name_of(e):
        a = e.attrib
        own = " ".join(t.strip() for t in e.itertext() if t.strip())
        return a.get("aria_label") or a.get("aria-label") or own or a.get("value") or a.get("placeholder") \
            or a.get("alt") or a.get("title") or a.get("name") or ""

    def walk(e, depth, shown):
        if not isinstance(e.tag, str):
            return
        tag, bid = e.tag.lower(), e.attrib.get("backend_node_id")
        role = INPUT.get(e.attrib.get("type", "text"), "textbox") if tag == "input" else ROLE.get(tag, "")
        role = e.attrib.get("role") or role
        name = name_of(e)
        keep = bid and (role in ("link", "button", "textbox", "searchbox", "combobox", "checkbox", "radio", "option")
                        or (name and name not in shown and len(list(e)) == 0))
        if keep:
            lines.append("\t" * depth + f"[{bid}] {role or 'generic'} {py(name)}")
            ids.add(bid)
            shown = shown + " " + name
        elif e.text and e.text.strip() and e.text.strip() not in shown and not bid:
            lines.append("\t" * depth + f"StaticText {py(e.text)}")
        for c in e:
            walk(c, depth + (1 if keep else 0), shown)

    walk(root, 0, "")
    return "\n".join(lines), ids


def window(page, bid, rng, cap=16000):
    """Long pages: a cap-sized run of whole lines holding the target at a random spot, so truncation never drops it."""
    if len(page) <= cap:
        return page
    at = re.search(rf"^\t*\[{bid}\] ", page, re.M).start()
    start = page.rfind("\n", 0, max(0, at - rng.randint(0, cap // 2))) + 1
    end = page.rfind("\n", 0, start + cap)
    if end <= at:  # the target line itself sits past the cap: end right after it
        end = page.find("\n", at) if page.find("\n", at) >= 0 else len(page)
    return ("[page truncated above]\n" if start else "") + page[start:end]


def history(reprs, idx):
    out = []
    for r in reprs[:idx]:  # "[button]  Search -> CLICK" / "[textbox]  From -> TYPE: Boston"
        m = re.match(r"\[(.*?)\]\s*(.*?)\s*->\s*(\w+)(?::\s*(.*))?$", r)
        if m:
            role, name, op, val = m.groups()
            out.append(f"{op.lower()} {role} {py(name)}" + (f" [{val}]" if val else ""))
    return out


def crop(png, box, rng):
    """Window of WIN size holding the target box, resized to IMG; None if the box doesn't fit."""
    im = Image.open(io.BytesIO(png)).convert("RGB")
    x, y, w, h = box
    if w <= 0 or h <= 0 or h > WIN[1] or y + h > im.height:
        return None
    if im.width < WIN[0]:
        im = im.crop((0, 0, WIN[0], max(im.height, WIN[1])))
    lo, hi = max(0, int(y + h) - WIN[1]), min(int(y), max(0, im.height - WIN[1]))
    top = rng.randint(lo, hi) if hi >= lo else lo
    return im.crop((0, top, WIN[0], top + WIN[1])).resize(IMG, Image.BICUBIC)


def main():
    rng = random.Random(0)
    IMGS.mkdir(parents=True, exist_ok=True)
    recs, skip = [], {}
    for shard in sorted(glob.glob(SRC)):
        t = pq.read_table(shard, columns=["action_uid", "annotation_id", "confirmed_task", "website", "operation",
                                          "pos_candidates", "cleaned_html", "screenshot", "action_reprs",
                                          "target_action_index"]).to_pylist()
        for r in t:
            op = json.loads(r["operation"])
            pos = [json.loads(c) for c in r["pos_candidates"]]
            if not pos or op["op"] not in VERB:
                skip["no target"] = skip.get("no target", 0) + 1
                continue
            bid = pos[0]["backend_node_id"]
            attrs = json.loads(pos[0]["attributes"])
            try:
                box = [float(v) for v in attrs["bounding_box_rect"].split(",")]
            except (KeyError, ValueError):
                skip["no box"] = skip.get("no box", 0) + 1
                continue
            page, ids = tree(r["cleaned_html"])
            if bid not in ids:
                skip["target not in tree"] = skip.get("target not in tree", 0) + 1
                continue
            page = window(page, bid, rng)
            shot = r["screenshot"] and r["screenshot"].get("bytes")
            im = crop(shot, box, rng) if shot else None
            if im is None:
                skip["screenshot"] = skip.get("screenshot", 0) + 1
                continue
            path = IMGS / f"{r['action_uid']}.jpg"
            im.save(path, quality=85)
            label = re.search(rf"^\t*\[{bid}\] (.*)$", page, re.M).group(1)
            val = op.get("value") or ""
            o = {"CLICK": "CLICK", "TYPE": "TYPE_TEXT", "SELECT": "SELECT"}[op["op"]]
            why = {"CLICK": f"To {r['confirmed_task'][0].lower() + r['confirmed_task'][1:]}, the next step on this "
                            f"page is to click [{bid}], the {label}.",
                   "TYPE_TEXT": f"The next step is to type \"{val}\" into [{bid}], the {label}.",
                   "SELECT": f"The next step is to choose \"{val}\" in [{bid}], the {label}."}[o]
            recs.append({"src": "mind2web", "aid": r["annotation_id"], "task": r["confirmed_task"],
                         "url": f"https://www.{r['website']}.com", "history": history(r["action_reprs"], int(r["target_action_index"])),
                         "page": page, "op": o, "target": bid, "arg": val if o != "CLICK" else None,
                         "step": int(r["target_action_index"]), "why": why, "image": str(path).replace("\\", "/")})
        print(f"{shard}: {len(recs)} records so far; skipped {skip}", flush=True)
    tasks = sorted({x["aid"] for x in recs})
    test = set(random.Random(1).sample(tasks, max(1, len(tasks) // 20)))
    for name, part in (("train", [x for x in recs if x["aid"] not in test]), ("test", [x for x in recs if x["aid"] in test])):
        with open(OUT / f"m2w_{name}.jsonl", "w", encoding="utf-8") as f:
            for x in part:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(name, len(part), "records", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
