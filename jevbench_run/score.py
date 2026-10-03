"""Score a JevBench public run of ours with JevBench's own scoring code, and place it on the published board.

    python jevbench_run/score.py                                   # Reflex Instinct (jevbench_run/*.jsonl)
    python jevbench_run/score.py --run reason --name "Reflex Reason 2B (Atlas AI)" --price-in 0.03 --price-out 0.15

Public items only (easy 48, standard 72, hard 111). The official score also uses the judge tier, held-out items and
a sealed set that only the benchmark's operators run, so the composite here is an estimate, labelled as such.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
BENCH = HERE.parent.parent / "jevbench"
TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}


def setup(bench=BENCH):
    sys.path.insert(0, str(bench))


def load_tasks(bench=BENCH):
    """-> {id: (tier, task record)}"""
    tasks = {}
    for f, tier in TIERS.items():
        for line in open(Path(bench) / "datasets/public" / f"{f}.jsonl", encoding="utf-8"):
            t = json.loads(line)
            tasks[t["id"]] = (tier, t)
    return tasks


def load_rows(run):
    return {r["task_id"]: r for f in TIERS for r in map(json.loads, open(Path(run) / f"{f}.jsonl", encoding="utf-8"))}


def evaluate(tasks, rows, name, usd_1000, bench=BENCH, board=True):
    """rows: {id: {correct, probs, latency_s}} over any subset of the tasks -> summary with axes and estimated score"""
    setup(bench)
    from jevbench import composite_v13 as v13, composite_v14 as v14
    from jevbench.metrics import ece_top_label, percentile

    acc, chance, by_family = {}, {}, {}
    for tier in TIERS.values():
        ids = [i for i in rows if tasks[i][0] == tier]
        if not ids:
            continue
        acc[tier] = statistics.mean(bool(rows[i]["correct"]) for i in ids)
        chance[tier] = statistics.mean(1 / len(tasks[i][1]["labels"]) for i in ids)
        for i in ids:
            fam = by_family.setdefault(f"{tier}/{tasks[i][1]['family']}", [0, 0])
            fam[0] += bool(rows[i]["correct"])
            fam[1] += 1
    hard = [rows[i] for i in rows if tasks[i][0] == "hard"]
    ece = ece_top_label([(max(r["probs"].values()), r["correct"]) for r in hard])["ece"]
    lat = [r["latency_s"] for r in rows.values()]
    p50, p95 = percentile(lat, 0.5), percentile(lat, 0.95)
    axes = {"intelligence": v13.intelligence(acc, chance), "calibration": v13.calibration(ece),
            "speed": v13.speed(p50, p95, "gpu"), "cost": v13.cost(usd_1000)}
    out = {"system": name, "accuracy": acc, "chance": chance, "hard_ece": ece,
           "latency_s": {"p50_raw": p50, "p95_raw": p95, "p50_adjusted": v13.adjusted_latency(p50, "gpu"),
                         "p95_adjusted": v13.adjusted_latency(p95, "gpu")},
           "usd_per_1000_estimate": usd_1000, "axes": axes, "score_public_only_estimate": v14.harmonic(axes),
           "by_family": {k: round(c / n, 3) for k, (c, n) in sorted(by_family.items())}}
    if board:
        place(out, rows, bench)
    return out


def place(out, rows, bench=BENCH):
    """Where the estimate would sit on the current board, and accuracy rank on the same items against every complete
    run whose per-item answers were published (the v1.2 round)."""
    score = out["score_public_only_estimate"]
    systems = json.load(open(Path(bench) / "results/v1.4.2.2/jevbench-v1.4.2.2-results.json", encoding="utf-8"))["systems"]
    ranked = [s for s in systems if s.get("ranked") and s.get("jevbench_score") is not None]
    out["would_place_among_ranked"] = 1 + sum(s["jevbench_score"] > score for s in ranked)
    out["ranked_of"] = len(ranked) + 1
    per = json.load(open(Path(bench) / "results/v1.2/jevbench-v1.2-per-task.json", encoding="utf-8"))
    tier_of = {t["id"]: t["tier"] for t in per["tasks"]}

    def public_acc(ok):
        r = {t: statistics.mean(ok.get(i, False) for i in tier_of if tier_of[i] == t) for t in ("easy", "standard", "hard")}
        return {**r, "all": statistics.mean(ok.get(i, False) for i in tier_of)}
    mine = public_acc({i: bool(r["correct"]) for i, r in rows.items()})
    others = [public_acc({i: v[0] == "c" for i, v in s["public_tasks"].items()}) for s in per["systems"].values() if not s.get("partial")]
    out["same_items_rank"] = {k: f"#{1 + sum(o[k] > mine[k] for o in others)} of {len(others) + 1}" for k in mine}
    keep = ("imajev", "jev 1.13", "raw qwen3 0.6b", "kev 0.6b", "laya", "reflex 4b")
    out["neighbours"] = [{"system": s["display"], "score": s.get("jevbench_score"), "rank": s.get("rank"),
                          "easy": s["tiers"].get("easy"), "standard": s["tiers"].get("standard")}
                         for s in systems if any(k in s["display"].lower() for k in keep)]


def report(s):
    print(json.dumps({k: v for k, v in s.items() if k not in ("neighbours", "by_family")}, indent=1))
    print("\nby family:", json.dumps(s["by_family"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=".", help="folder under jevbench_run/ holding easy/original/hard.jsonl")
    ap.add_argument("--name", default="Reflex Instinct 0.6B (Atlas AI)")
    # the board's size-class reference prices: 0.6B -> DeepInfra Qwen3-Embedding-0.6B $0.01/M in, $0 out (Instinct
    # generates nothing); ~2B -> DeepInfra Qwen3.5-4B $0.03/M in, $0.15/M out (no hosted ~2B listed; errs high)
    ap.add_argument("--price-in", type=float, default=0.01)
    ap.add_argument("--price-out", type=float, default=0.0)
    a = ap.parse_args()
    run = HERE / a.run
    tasks, rows = load_tasks(), load_rows(run)
    usd = 1000 * statistics.mean(r["usage"]["input_tokens"] * a.price_in + r["usage"].get("output_tokens", 0) * a.price_out
                                 for r in rows.values()) / 1e6
    s = evaluate(tasks, rows, a.name, usd)
    (run / "summary.json").write_text(json.dumps(s, indent=1))
    report(s)


if __name__ == "__main__":
    main()
