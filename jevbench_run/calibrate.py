"""Calibrate Reflex Reason, and route Instinct -> Reason, without letting any item tune its own score.

    python jevbench_run/calibrate.py

The 231 public items are split in two halves by paraphrase group (reworded pairs stay together). Each half's settings
(Reason's temperature, and the confidence below which Instinct hands a question to Reason) are fitted on the other
half only. Both models are deterministic, so the routed system is computed exactly from the two models' own runs:
Instinct's answer and latency, plus Reason's when it is called. reflex_server.py --mode routed runs the same thing live.
"""
import hashlib
import json
import math
import statistics
from pathlib import Path

import score
from reflex_server import confidence, options

HERE = Path(__file__).parent
PRICE = {"instinct_in": 0.01, "reason_in": 0.03, "reason_out": 0.15}  # $/M tokens, the board's size-class references
T_GRID = [0.25 * 1.03 ** i for i in range(240)]  # 0.25 .. ~300
C_GRID = [i / 20 for i in range(21)] + [1.01]  # 1.01: always ask Reason


def key(t):
    q = {"type": t["question"]["type"], "instructions": t["question"]["instructions"]}
    if t["question"].get("criteria") is not None:
        q["criteria"] = t["question"]["criteria"]
    return hashlib.sha256(json.dumps([t["state"], q], sort_keys=True).encode()).hexdigest()[:16]


def softmax(xs, T):
    m = max(xs)
    e = [math.exp((x - m) / T) for x in xs]
    return [v / sum(e) for v in e]


def fit_temperature(items):
    """items: [(letter logits, index of the right option)] -> T with the lowest mean negative log-likelihood"""
    nll = lambda T: statistics.mean(-math.log(max(softmax(l, T)[e], 1e-12)) for l, e in items)  # noqa: E731
    return min(T_GRID, key=nll)


def main():
    score.setup()
    from jevbench.scoring import score_task
    from jevbench.tasks import Task
    tasks = score.load_tasks()
    inst = score.load_rows(HERE)
    reas = score.load_rows(HERE / "reason-logged")
    log = {r["key"]: r for r in map(json.loads, open(HERE / "reason/log.jsonl", encoding="utf-8"))}
    fold = {i: int(hashlib.sha256((t["group"] or i).encode()).hexdigest(), 16) % 2 for i, (_, t) in tasks.items()}
    keys = {i: [k for k, _ in options(t["question"])] for i, (_, t) in tasks.items()}
    right = {i: keys[i].index(str(t["expected"])) for i, (_, t) in tasks.items()}
    letters = {i: log[key(t)]["letters"] for i, (_, t) in tasks.items()}

    def graded(i, probs, latency, cost):
        g = score_task(probs, Task(**tasks[i][1]))
        return {"correct": g["correct"], "probs": g["probs"], "latency_s": latency, "cost": cost}

    def reason_row(i, T):
        p = softmax(letters[i], T)
        cost = reas[i]["usage"]["input_tokens"] * PRICE["reason_in"] + reas[i]["usage"]["output_tokens"] * PRICE["reason_out"]
        return graded(i, {k: v for k, v in zip(keys[i], p)}, reas[i]["latency_s"], cost)

    def routed_row(i, T, c):
        cost = inst[i]["usage"]["input_tokens"] * PRICE["instinct_in"]
        if confidence(inst[i]["probs"]) >= c:
            return {**graded(i, inst[i]["probs"], inst[i]["latency_s"], cost), "escalated": False}
        r = reason_row(i, T)
        return {**r, "latency_s": inst[i]["latency_s"] + r["latency_s"], "cost": cost + r["cost"], "escalated": True}

    usd = lambda rows: 1000 * statistics.mean(r["cost"] for r in rows.values()) / 1e6  # noqa: E731
    est = lambda rows: score.evaluate(tasks, rows, "", usd(rows), board=False)["score_public_only_estimate"]  # noqa: E731

    # settings per half, each fitted on the other half
    T, c = {}, {}
    for f in (0, 1):
        train = [i for i in tasks if fold[i] != f]
        T[f] = fit_temperature([(letters[i], right[i]) for i in train])
        c[f] = max(C_GRID, key=lambda cc: est({i: routed_row(i, T[f], cc) for i in train}))
    calibrated = {i: reason_row(i, T[fold[i]]) for i in tasks}
    routed = {i: routed_row(i, T[fold[i]], c[fold[i]]) for i in tasks}

    out = {}
    for name, folder, rows in (("Reflex Reason 2B, calibrated (Atlas AI)", "reason-calibrated", calibrated),
                               ("Reflex Instinct -> Reason, routed (Atlas AI)", "routed", routed)):
        s = score.evaluate(tasks, rows, name, usd(rows))
        s["settings_per_half"] = {"temperature": T, "escalate_below": c}
        if folder == "routed":
            s["escalated_share"] = statistics.mean(r["escalated"] for r in rows.values())
        (HERE / folder).mkdir(exist_ok=True)
        (HERE / folder / "summary.json").write_text(json.dumps(s, indent=1))
        out[folder] = s
        print(f"\n=== {name}")
        score.report(s)

    # the settings to ship, fitted on all 231 (the numbers above never saw their own items)
    T_all = fit_temperature([(letters[i], right[i]) for i in tasks])
    c_all = max(C_GRID, key=lambda cc: est({i: routed_row(i, T_all, cc) for i in tasks}))
    print(f"\nship: --temperature {T_all:.3f} --escalate-below {c_all}")
    (HERE / "settings.json").write_text(json.dumps({"temperature": T_all, "escalate_below": c_all}, indent=1))


if __name__ == "__main__":
    main()
