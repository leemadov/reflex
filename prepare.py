"""Convert NNetNav (WebArena + live web) trajectories into compact decision records.

    python prepare.py            -> data/train.jsonl, data/test.jsonl

One record per agent step:
    {"src", "task", "url", "history": [str], "page": a11y tree,
     "op": CLICK|TYPE_TEXT|..., "target": element id or None, "arg": str or None,
     "step": i, "n_steps": n or None, "why": the agent's step-by-step reasoning (System 2 training text)}
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data")

ACT = re.compile(r"```\s*(click|type|select|hover|press|scroll|new_tab|tab_focus|close_tab|goto|go_back|go_forward|stop)\b([^`]*)```")
OPS = {"click": "CLICK", "type": "TYPE_TEXT", "select": "SELECT", "hover": "HOVER", "press": "PRESS_KEY", "goto": "GOTO_URL",
       "go_back": "GO_BACK", "go_forward": "GO_FORWARD", "new_tab": "SWITCH_TAB", "tab_focus": "SWITCH_TAB",
       "close_tab": "SWITCH_TAB"}
ICON_ONLY = re.compile(r"\s*StaticText '[-\s]*'")


def compact(tree: str) -> str:
    """Label-agnostic a11y-tree cleanup; serve time applies the same function."""
    out, parents = [], []  # parents: stack of (indent, name) for dedup of echoed StaticText
    for ln in tree.split("\n"):
        ln = re.sub(r", url='[^']*'", "", ln)
        if ICON_ONLY.fullmatch(ln):
            continue
        ind = len(ln) - len(ln.lstrip("\t"))
        while parents and parents[-1][0] >= ind:
            parents.pop()
        m = re.fullmatch(r"\s*StaticText '(.*)'", ln)
        if m and m.group(1).strip() and any(m.group(1).strip() in p for _, p in parents):
            continue  # "[12] button 'Save'" followed by "StaticText 'Save'" says nothing new
        name = re.search(r"'(.*)'", ln)
        parents.append((ind, name.group(1) if name else ""))
        out.append(ln)
    return "\n".join(out)


def parse_action(text: str):
    m = ACT.findall(text)
    if not m:
        return None
    name, rest = m[-1]
    args = re.findall(r"\[([^\]]*)\]", rest)
    if name == "stop":
        ans = rest[rest.find("[") + 1: rest.rfind("]")].strip() if "[" in rest else ""
        return ("BLOCKED" if ans.upper() in ("N/A", "NA", "") else "DONE"), None, ans
    if name == "scroll":
        d = (args[0] if args else "down").lower()
        return ("SCROLL_UP" if "up" in d else "SCROLL_DOWN"), None, None
    op = OPS[name]
    if op in ("CLICK", "TYPE_TEXT", "SELECT", "HOVER"):
        if not args or not args[0].strip().isdigit():
            return None
        return op, args[0].strip(), (args[1] if op in ("TYPE_TEXT", "SELECT") and len(args) > 1 else None)
    return op, None, (args[0] if args else None)


def nnetnav(path: Path, src: str):
    recs = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        user, asst = r["messages"][1]["content"], r["messages"][-1]["content"]
        if "OBSERVATION:\n" not in user or "\nOBJECTIVE: " not in user:
            continue
        act = parse_action(asst)
        if act is None:
            continue
        op, target, arg = act
        tree = user.split("OBSERVATION:\n", 1)[1].rsplit("\nURL: ", 1)[0]
        page = compact(tree)
        if target and not re.search(rf"^\t*\[{target}\] ", page, re.M):
            continue  # agent pointed at an id that is not on the page
        url = user.rsplit("\nURL: ", 1)[1].split("\n", 1)[0].strip()
        task = user.split("\nOBJECTIVE: ", 1)[1].split("\nPREVIOUS ACTIONS:", 1)[0].strip()
        hist = [h.split(": ", 1)[1] for h in user.split("PREVIOUS ACTIONS:", 1)[1].strip().split("\n")
                if ": " in h and h.split(": ", 1)[1].strip() != "None"]
        why = re.split(r"\s*In summary,? the next action", asst.split("```")[0], maxsplit=1)[0].strip()
        recs.append({"src": src, "tid": r["id"], "task": task, "url": url, "history": hist, "page": page,
                     "op": op, "target": target, "arg": arg, "step": len(hist), "why": why})
    # trajectory length -> progress labels (only when steps of one trajectory share an id)
    n = defaultdict(int)
    for x in recs:
        n[(x["tid"], x["task"])] = max(n[(x["tid"], x["task"])], x["step"] + 1)
    for x in recs:
        x["n_steps"] = n[(x.pop("tid"), x["task"])]
    return recs


def main():
    OUT.mkdir(exist_ok=True)
    for split in ("test", "train"):
        recs = []
        for name in ("nnetnav-wa", "nnetnav-live"):
            got = nnetnav(RAW / name / f"{split}.jsonl", name)
            print(f"{split} {name}: {len(got)} records", flush=True)
            recs += got
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for x in recs:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(split, "ops", Counter(x["op"] for x in recs).most_common(), flush=True)
        print(split, "n_steps", Counter(min(x["n_steps"], 10) for x in recs).most_common(), flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
