"""#17：在人工反例集（antibody set）上量模型对 `noul` 新口径的遵守程度。

反例集（`mai_jev_noul_antibody.jsonl`）每行**只带 `noul` 这一个问题** —— 它是人写的、
专门针对 rubric 的反例（关键词会命中、判定却应当相反的句子）。所以 `calibrate_eval.py`
不能直接吃它：那个脚本按三原语逐行取 `gold`，遇到缺原语的行会 `KeyError`。

这个探针只做一件事：把 checkpoint 在这些行上的 `noul` 输出与**人写的判定**比一次，
并同时报规则基线，让「模型 / 关键词表 / 掷硬币」在同一批行上可比、可复现。

判据仍是 **balanced accuracy**（反例集 37 行里 false 27 / true 10，偏斜）。
反例集**不参与训练**，所以它量的是泛化，不是记忆。

用法：
  python scripts/eval_noul_antibody.py ^
      --checkpoint  $WORK\\runs\\v04_rlcd_noul ^
      --antibody    $WORK\\data\\adapted_v04\\mai_jev_noul_antibody.jsonl ^
      --calibration $WORK\\runs\\v04_rlcd_noul\\paper_eval\\calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_rules import predict_noul  # noqa: E402

NOUL_QID = "needs_refund_or_escalation"
NOUL_QTYPE = 2  # laya 的 QTYPES = {'choice': 0, 'score': 1, 'noul': 2}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def softmax(z, scale: float):
    import numpy as np

    z = np.asarray(z, dtype=float) / max(scale, 1e-4)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def temperature_from_calibration(path: Path) -> tuple[list[float], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [float(t) for t in payload["temperature"]], dict(payload.get("temperature_by_options") or {})


def balanced(gold: list[int], pred: list[int]) -> float:
    """每类召回率的宏平均（类别只有 true/false 两类）。"""
    recalls = []
    for cls in (0, 1):
        idx = [i for i, g in enumerate(gold) if g == cls]
        if idx:
            recalls.append(sum(1 for i in idx if pred[i] == cls) / len(idx))
    return sum(recalls) / len(recalls) if recalls else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--antibody", required=True)
    ap.add_argument("--calibration", help="calibration.json（不给就用温度 1.0）")
    ap.add_argument("--list-wrong", action="store_true", help="逐行列出判错的条目")
    ap.add_argument("--out", help="把三行对照写成 JSON（make_paper_tables.py 的 --antibody-json）")
    args = ap.parse_args()

    import laya
    from laya.common import temp_bucket
    from laya.train import calibration_records, items_from_rows

    rows = [r for r in load_jsonl(Path(args.antibody)) if NOUL_QID in (r.get("gold") or {})]
    if not rows:
        raise SystemExit(f"{args.antibody} 里没有带 {NOUL_QID} 的行")

    gold = []
    for row in rows:
        probs = row["gold"][NOUL_QID]["probabilities"]
        gold.append(1 if max(probs, key=lambda k: probs[k]) == "true" else 0)

    if args.calibration:
        temps, by_options = temperature_from_calibration(Path(args.calibration))
    else:
        temps, by_options = [1.0, 1.0, 1.0], {}
    scale = float(by_options.get(temp_bucket(NOUL_QTYPE, 2), temps[NOUL_QTYPE]))

    agent = laya.load(args.checkpoint)
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 64))
    items, skipped = items_from_rows(agent.tok, rows, max_len, head_max_len)
    recs = list(calibration_records(agent.model, agent.tok, items, agent.device,
                                    max_len, head_max_len, batch_size=16))

    pred, conf = [], []
    for qt, logits, _target, k in recs:
        assert int(qt) == NOUL_QTYPE, f"反例集里出现了非 noul 记录：qtype={qt}"
        p = softmax(logits[:k], scale)
        pred.append(int(p.argmax()))
        conf.append(float(p.max()))

    n = len(pred)
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    # 规则基线在同一批行上（`predict_noul` 返回 bool，单原语、不吃 gold 结构）
    rules_pred = [1 if predict_noul(r["state"]) else 0 for r in rows]
    rules_correct = sum(1 for g, p in zip(gold, rules_pred) if g == p)

    print(f"checkpoint   : {args.checkpoint}")
    print(f"antibody rows: {len(rows)}  (built {n} records, skipped={skipped})")
    print(f"temperature  : {scale:.4f} (noul)")
    print(f"gold         : false={gold.count(0)} true={gold.count(1)}")
    print()
    print(f"{'scorer':<34}{'accuracy':>10}{'balanced':>10}{'mean conf':>11}")
    print(f"{'model (' + Path(args.checkpoint).name + ')':<34}"
          f"{correct / n:>10.4f}{balanced(gold, pred):>10.4f}{sum(conf) / n:>11.4f}")
    print(f"{'rules baseline (baseline_rules.py)':<34}"
          f"{rules_correct / n:>10.4f}{balanced(gold, rules_pred):>10.4f}{'-':>11}")
    print(f"{'majority class (always false)':<34}"
          f"{gold.count(0) / n:>10.4f}{0.5:>10.4f}{'-':>11}")

    if args.list_wrong:
        print()
        print("model errors:")
        for row, g, p in zip(rows, gold, pred):
            if g != p:
                print(f"  gold={['false', 'true'][g]:<5} pred={['false', 'true'][p]:<5} {row['state']}")

    if args.out:
        # 论文的宏由 make_paper_tables.py 从这个 JSON 生成（正文里不出现手写小数）。
        payload = {
            "checkpoint": str(args.checkpoint),
            "antibody": str(args.antibody),
            "temperature": scale,
            "rows": n,
            "skipped": skipped,
            "gold": {"false": gold.count(0), "true": gold.count(1)},
            "model": {
                "accuracy": correct / n,
                "balanced_accuracy": balanced(gold, pred),
                "mean_confidence": sum(conf) / n,
            },
            "rules": {
                "accuracy": rules_correct / n,
                "balanced_accuracy": balanced(gold, rules_pred),
            },
            "majority_class": {
                "accuracy": gold.count(0) / n,
                "balanced_accuracy": 0.5,
            },
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
