"""#15：在**部署变体**（ONNX 图）上跑一遍 test split，产出论文用的量化退化数字。

为什么不用 `calibrate_eval.py`：那个脚本走 eager torch + `laya.load`，测的是权重；
论文的 Efficiency 一节要回答的是「把权重换成 int8-gather 图之后还准不准」，
而那条部署路径是 `mai_jev.deploy.Predictor`（onnxruntime + tokenizers，无 torch）。
两者的 tokenizer 加载方式不同（`tokenizers.Tokenizer.from_file` vs HF wrapper），
所以必须在部署侧实测，不能把 eager 的数字搬过去。

这个脚本只做一件事：给一个 export 目录，算出三原语的 argmax accuracy，并可选地
与 eager 的 `calibration_eval.json` 比对，写出量化退化。

用法：
  python scripts/eval_export.py --export-dir $WORK/export/v03-int8-ml256 ^
      --data $WORK/data/adapted_v03/mai_jev_test.laya.jsonl ^
      --eager-eval $WORK/runs/v03_rlcd_bal/paper_eval/calibration_eval.json ^
      --threads 4 --out $WORK/export/quant_report.json

gold 的取法必须分原语，这是踩过的坑（见 docs/research/v03-quant-degradation.md）：
choice 的键是 criteria 名、score 的键是 0-based 下标字符串、noul 的键是 "false"/"true"。
一律按整数下标取会让 choice 之外全取成 0，然后你会以为自己测出了「量化崩了」。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PRIMS = ("route_team", "urgency", "needs_refund_or_escalation")

# calibrate_eval.py 的分区键是原语类型名，不是我们的问题 id。
PRIM_TYPE = {"route_team": "choice", "urgency": "score", "needs_refund_or_escalation": "noul"}


def gold_key(question: str, prob: dict) -> str:
    """gold 概率向量的 argmax 键。分原语，见模块 docstring。"""
    return max(prob.items(), key=lambda kv: kv[1])[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--data", required=True, help="laya 格式的 test jsonl")
    ap.add_argument("--eager-eval", help="calibration_eval.json，用来算量化退化")
    ap.add_argument("--model-mb", type=float, help="写进报告的 model.onnx 体积（MB）")
    ap.add_argument("--fp32-model-mb", type=float, help="对应的 fp32 图体积（MB），用于对比")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--limit", type=int, help="只跑前 N 行；不给出就跑全量")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from mai_jev.deploy import Predictor

    rows = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    pred = Predictor(args.export_dir, threads=args.threads)

    hit = {p: 0 for p in PRIMS}
    conf = {p: 0.0 for p in PRIMS}
    for row in rows:
        state = row["state"]
        gold = row.get("gold") or row.get("expected")
        for qid in PRIMS:
            if qid not in gold:
                continue
            ans = pred.predict_question(qid, state)
            want = gold_key(qid, gold[qid]["probabilities"])
            # gold 键与部署侧的 label 是三套不同的约定（choice = criteria 名、
            # score = 0-based 下标、noul = "true"/"false"），而部署侧的 label 是
            # 选项文本的第一段（score 是 "level 0"）。统一做法：**不看约定，只看候选集**——
            # gold 键必须能在该问题的候选 label 集合里找到等价项，否则先报错。
            labels = list(ans["labels"])
            if want in labels:
                target = want
            elif pred.questions[qid]["type"] == "score" and f"level {want}" in labels:
                # score：gold 键是 0-based 下标 "0".."K-1"，部署侧 label 是 "level 0".."level K-1"。
                target = f"level {want}"
            else:
                raise SystemExit(
                    f"gold 键 {want!r} 在 {qid} 的候选集 {labels} 里找不到等价项；"
                    "多半是 gold 键约定或 head 渲染对不上，别硬凑，先查清楚。"
                )
            if ans["label"] == target:
                hit[qid] += 1
            conf[qid] += float(ans["confidence"])

    n = len(rows)
    out: dict = {
        "export_dir": args.export_dir,
        "data": args.data,
        "n": n,
        "threads": args.threads,
        "accuracy": {p: hit[p] / n for p in PRIMS},
        "mean_confidence": {p: conf[p] / n for p in PRIMS},
        "macro_accuracy": sum(hit[p] / n for p in PRIMS) / len(PRIMS),
    }
    if args.model_mb:
        out["model_mb"] = args.model_mb
    if args.fp32_model_mb:
        out["fp32_model_mb"] = args.fp32_model_mb

    if args.eager_eval:
        ev = json.loads(Path(args.eager_eval).read_text(encoding="utf-8"))
        eager = ev["test_after_calibration"]
        out["eager_accuracy"] = {p: eager[PRIM_TYPE[p]]["accuracy"] for p in PRIMS}
        out["degradation"] = {
            p: round(eager[PRIM_TYPE[p]]["accuracy"] - out["accuracy"][p], 4) for p in PRIMS
        }
        out["note"] = (
            "degradation = eager accuracy - 部署变体 accuracy。"
            "两个温度不同（各自变体必须在自己的 calib 上重拟合），"
            "所以这不是纯量化误差，但它是上线口径真正关心的数。"
        )

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())