"""mai-jev：对 laya-train 的 checkpoint 做三原语评测，输出可比对基线的指标。

为什么不用 laya-train 自带的 before/after 表：
  1. 它按 question type 聚合（choice/score/noul），我们按**问题 id** 聚合
     （route_team/urgency/needs_refund_or_escalation）—— 名字不一样，报告没法并排看。
  2. 它的 accuracy 对 score 是"argmax 命中"，对 urgency 我们要的是
     off-by-one 容差（差一级算对），这是客服分诊的常规口径。
  3. 它不跑规则基线，而验收要求"超过同规模基线"。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  python scripts\\eval_checkpoint.py ^
      --checkpoint $WORK\\runs\\smoke ^
      --data $WORK\\data\\v0.1_test.jsonl ^
      --baseline --out $WORK\\runs\\smoke_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mai_jev.questions import SCORE_OFFSET, label_space  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_rules import evaluate as evaluate_baseline, load_jsonl  # noqa: E402


def load_agent(checkpoint: str):
    import laya

    return laya.load(checkpoint)


def argmax_label(qid: str, answer: dict) -> tuple[object, float]:
    """从 laya 的 answer 里取预测标签与置信度。"""
    t = answer.get("type")
    if t == "choice":
        probs = answer.get("probabilities") or {}
        if probs:
            label = max(probs, key=lambda k: probs[k])
            return label, float(probs[label])
        return answer.get("choice"), float(answer.get("confidence", 0.0))
    if t == "noul":
        probs = answer.get("probabilities") or {}
        if probs:
            label = "true" if probs.get("true", 0.0) >= probs.get("false", 0.0) else "false"
            return label, float(probs.get(label, 0.0))
        return ("true" if answer.get("choice") else "false"), float(answer.get("confidence", 0.0))
    if t == "score":
        # laya 的 score 输出是 [0, n_levels-1] 的连续值；我们对外是 1..5。
        return int(round(float(answer.get("score", 0.0)) + SCORE_OFFSET)), float(answer.get("confidence", 0.0))
    raise ValueError(f"unknown answer type {t!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="+", required=True, help="build_dataset 产出的 jsonl")
    ap.add_argument("--baseline", action="store_true", help="同时跑规则基线")
    ap.add_argument("--out", help="把指标写成 json")
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.data:
        rows.extend(load_jsonl(Path(path)))
    if not rows:
        print("no rows", file=sys.stderr)
        raise SystemExit(2)

    agent = load_agent(args.checkpoint)
    questions = rows[0]["questions"]

    from collections import Counter

    per_q: dict[str, Counter] = {qid: Counter() for qid in questions}
    totals: Counter = Counter()
    urgency_off_by_one = 0
    urgency_n = 0
    confusion: dict[str, dict[str, int]] = {}

    for row in rows:
        gold = row["labels"]
        # predict() 返回 {answers: {qid: {...}}, usage: {...}}；decide() 是 structured
        # 那条路（要 JSON schema），我们这里是三原语 questions dict，用 predict。
        payload = agent.predict(row["state"], questions)
        answers = payload.get("answers", payload)
        if not isinstance(answers, dict):
            raise TypeError(f"predict() returned {type(answers).__name__}, expected dict of qid -> answer")
        for qid in questions:
            a = answers[qid]
            label, confidence = argmax_label(qid, a)
            g = gold[qid]
            hit = label == g
            per_q[qid]["hit" if hit else "miss"] += 1
            per_q[qid]["conf_sum"] += confidence
            totals[f"{qid}_hit" if hit else f"{qid}_miss"] += 1
            if qid == "urgency":
                urgency_n += 1
                if abs(int(label) - int(g)) <= 1:
                    urgency_off_by_one += 1
            if qid == "route_team":
                confusion.setdefault(str(g), {}).setdefault(str(label), 0)
                confusion[str(g)][str(label)] += 1

    report: dict = {"n": len(rows), "questions": {}}
    for qid in questions:
        hit = per_q[qid]["hit"]
        miss = per_q[qid]["miss"]
        n = hit + miss
        entry = {
            "n": n,
            "accuracy": hit / n if n else 0.0,
            "mean_confidence": (per_q[qid]["conf_sum"] / n) if n else 0.0,
        }
        if qid == "urgency" and urgency_n:
            entry["off_by_one"] = urgency_off_by_one / urgency_n
        if qid == "route_team":
            entry["confusion"] = confusion
        report["questions"][qid] = entry

    if args.baseline:
        report["baseline_rules"] = evaluate_baseline(rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
