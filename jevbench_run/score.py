"""Score Reflex Instinct's JevBench public run with JevBench's own scoring code, and place it on the published board.

    python jevbench_run/score.py [--bench ../jevbench]

Public items only (easy 48, standard 72, hard 111). The official score also uses the judge tier, held-out items and
a sealed set that only the benchmark's operators run, so the composite here is an estimate, labelled as such.
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
TIERS = {"easy": "easy", "original": "standard", "hard": "hard"}
PRICE_IN_PER_M = 0.01  # the board's rule for 0.6B models: DeepInfra Qwen3-Embedding-0.6B size class, $0.01/M input, $0 output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=str(HERE.parent.parent / "jevbench"))
    a = ap.parse_args()
    sys.path.insert(0, a.bench)
    from jevbench import composite_v13 as v13, composite_v14 as v14
    from jevbench.metrics import ece_top_label, percentile

    tasks, rows = {}, {}
    for f, tier in TIERS.items():
        for line in open(Path(a.bench) / "datasets/public" / f"{f}.jsonl", encoding="utf-8"):
            t = json.loads(line)
            tasks[t["id"]] = (tier, t)
        for line in open(HERE / f"{f}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            rows[r["task_id"]] = r

    acc, chance, by_family = {}, {}, {}
    for tier in TIERS.values():
        ids = [i for i, (t, _) in tasks.items() if t == tier]
        acc[tier] = sum(bool(rows[i]["correct"]) for i in ids) / len(ids)
        chance[tier] = statistics.mean(1 / len(tasks[i][1]["labels"]) for i in ids)
        for i in ids:
            fam = by_family.setdefault(f"{tier}/{tasks[i][1]['family']}", [0, 0])
            fam[0] += bool(rows[i]["correct"])
            fam[1] += 1
    hard = [rows[i] for i, (t, _) in tasks.items() if t == "hard"]
    ece = ece_top_label([(max(r["probs"].values()), r["correct"]) for r in hard])["ece"]
    lat = [r["latency_s"] for r in rows.values()]
    p50, p95 = percentile(lat, 0.5), percentile(lat, 0.95)
    tokens = statistics.mean(r["usage"]["input_tokens"] for r in rows.values())
    usd_1000 = tokens * 1000 * PRICE_IN_PER_M / 1e6

    axes = {"intelligence": v13.intelligence(acc, chance), "calibration": v13.calibration(ece),
            "speed": v13.speed(p50, p95, "gpu"), "cost": v13.cost(usd_1000)}
    score = v14.harmonic(axes)
    ours = {"system": "Reflex Instinct 0.6B (Atlas AI)", "accuracy": acc, "chance": chance, "hard_ece": ece,
            "latency_s": {"p50_raw": p50, "p95_raw": p95, "p50_adjusted": v13.adjusted_latency(p50, "gpu"),
                          "p95_adjusted": v13.adjusted_latency(p95, "gpu")},
            "input_tokens_mean": tokens, "usd_per_1000_estimate": usd_1000, "axes": axes,
            "score_public_only_estimate": score,
            "by_family": {k: round(c / n, 3) for k, (c, n) in sorted(by_family.items())}}

    board = json.load(open(Path(a.bench) / "results/v1.4.2.2/jevbench-v1.4.2.2-results.json", encoding="utf-8"))["systems"]
    hp = sorted((s for s in board if s["tiers"].get("hard_public") is not None), key=lambda s: -s["tiers"]["hard_public"])
    ours["hard_public_rank"] = 1 + sum(s["tiers"]["hard_public"] > acc["hard"] for s in hp)
    ours["hard_public_of"] = len(hp) + 1
    ranked = sorted((s for s in board if s.get("ranked") and s.get("jevbench_score") is not None), key=lambda s: -s["jevbench_score"])
    ours["would_place_among_ranked"] = 1 + sum(s["jevbench_score"] > score for s in ranked)
    ours["ranked_of"] = len(ranked) + 1
    keep = ("imajev", "jev 1.13", "raw qwen3 0.6b", "kev 0.6b", "qwen3.5-0.8b decision", "simplejev", "laya", "reflex 4b", "needle 3 (")
    ours["neighbours"] = [{"system": s["display"], "score": s.get("jevbench_score"), "rank": s.get("rank"),
                           "easy": s["tiers"].get("easy"), "standard": s["tiers"].get("standard"),
                           "hard_public": s["tiers"].get("hard_public"), "axes": s.get("axes")}
                          for s in board if any(k in s["display"].lower() for k in keep)]
    (HERE / "summary.json").write_text(json.dumps(ours, indent=1))
    print(json.dumps({k: v for k, v in ours.items() if k not in ("neighbours", "by_family")}, indent=1))
    print("\nby family:", json.dumps(ours["by_family"]))
    print(f"\n{'system':58} {'score':>6} {'rank':>5} {'easy':>6} {'std':>6} {'hard':>6}")
    print(f"{ours['system']:58} {score:6.2f} {'~' + str(ours['would_place_among_ranked']):>5} {acc['easy']:6.3f} {acc['standard']:6.3f} {acc['hard']:6.3f}")
    for n in ours["neighbours"]:
        f = lambda v: f"{v:6.3f}" if isinstance(v, float) else f"{'-':>6}"
        print(f"{n['system'][:58]:58} {(n['score'] or 0):6.2f} {str(n['rank'] or '-'):>5} {f(n['easy'])} {f(n['standard'])} {f(n['hard_public'])}")


if __name__ == "__main__":
    main()
