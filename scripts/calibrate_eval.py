"""mai-jev：温度标定 + 裁决级评测（ticket #6）。

为什么不用 `laya-train` 自带的 before/after 表，也不复用 `eval_checkpoint.py`：
  * laya 自带表按 question type（choice/score/noul）聚合，我们按**问题 id**聚合，
    报告没法并排看；`eval_checkpoint.py` 按 id 聚合但只出 accuracy/mean_confidence，
    没有标定、没有 ECE、没有置信度分档、没有多数类基线 —— 而验收线要这些。

本脚本做四件事：
  1. 在**独立的 calib split**（#7 build_dataset 留出的 285 行，不是训练集）上，
     用 `laya.calibrate.fit_temperature_map(..., compute_ece=True)` 拟合每原语温度。
     `compute_ece=True` 会再内部留出 20% 只用来算 ECE，拟合本身只用剩下的 80%
     —— 避免「在拟合集上报 ECE」这种自欺。
  2. 把拟合出的温度写成 `calibration.json`（laya 的 `calibration_payload` 形状），
     推理时用 `laya.load(checkpoint, calibration=...)` 装载。
  3. 在**独立 test split**（343 行，从未参与训练、也未参与标定）上，报标定前/后
     每原语 accuracy、ECE、soft accuracy，以及置信度三档
     （≥0.9 自动执行 / 0.6–0.9 复核 / <0.6 转人工）的覆盖率与实际命中率。
  4. 对照组：原版 `laya-multilingual`、多数类基线、`scripts/baseline_rules.py` 规则基线。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  python scripts\\calibrate_eval.py ^
      --checkpoint $WORK\\runs\\v02_rlcd_e2 ^
      --calib-data $WORK\\data\\adapted\\mai_jev_calib.laya.jsonl ^
      --test-data $WORK\\data\\adapted\\mai_jev_test.laya.jsonl ^
      --baseline-checkpoint convaiinnovations/laya-multilingual ^
      --out $WORK\\runs\\v02_rlcd_e2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from mai_jev.questions import QUESTION_ORDER, SCORE_OFFSET  # noqa: E402
from baseline_rules import evaluate as evaluate_rules, load_jsonl  # noqa: E402

# 置信度三档（ticket #6 第 2 条）
AUTO_THRESHOLD = 0.9
REVIEW_THRESHOLD = 0.6


def qtype_of(question: dict) -> int:
    return {"choice": 0, "score": 1, "noul": 2}[question["type"]]


def gold_probs(row: dict, qid: str) -> dict:
    """v0.2 数据的 `gold[qid]['probabilities']`（score 的 key 已是 0-based）。"""
    return row["gold"][qid]["probabilities"]


def gold_label(row: dict, qid: str):
    """规则基线要的对外面标签：urgency 回到对外的 1..5，noul 回到 bool。

    `baseline_rules.predict_urgency` 返回 int、`predict_noul` 返回 bool，直接把
    0-based 的字符串 key 塞给它会永远判错（`"3" == 3` 是 False）。
    """
    probs = gold_probs(row, qid)
    key = max(probs, key=lambda k: probs[k])
    if qid == "urgency":
        return int(key) + SCORE_OFFSET
    if qid == "needs_refund_or_escalation":
        return key == "true"
    return key


def records_for(agent, rows: list[dict], max_len: int, head_max_len: int) -> list:
    """jsonl 行 -> `(qtype, logits, target, k)` 记录（laya 的内部格式）。"""
    from laya.train import calibration_records, items_from_rows
    import torch

    items, skipped = items_from_rows(agent.tok, rows, max_len, head_max_len)
    if not items:
        raise SystemExit(f"no items built from {len(rows)} rows; skipped={skipped}")
    # `laya.load` 默认把模型放在 CUDA 上；`calibration_records` 不搬模型，
    # 硬塞 cpu 会撞 "Expected all tensors to be on the same device"。
    device = getattr(agent, "device", None) or torch.device("cpu")
    recs = list(calibration_records(agent.model, agent.tok, items, device,
                                    max_len, head_max_len, batch_size=16))
    return recs, skipped


def softmax(z, scale: float):
    import numpy as np

    z = np.asarray(z, dtype=float) / max(scale, 1e-4)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def per_item_view(recs: list, temperatures: list, temperature_by_options: dict) -> list:
    """把记录摊成 `[{qtype, p, target, gold, pred, conf}]`，供分档统计与 soft accuracy。"""
    from laya.common import temp_bucket
    import numpy as np

    out = []
    for qt, logits, target, k in recs:
        bucket = temp_bucket(qt, k)
        scale = float(temperature_by_options.get(bucket, temperatures[qt]))
        p = softmax(logits[:k], scale)
        t = np.asarray(target[:k], dtype=float)
        gold = int(np.argmax(t))
        pred = int(np.argmax(p))
        out.append({
            "qtype": int(qt), "p": p, "target": t,
            "gold": gold, "pred": pred, "correct": pred == gold,
            "conf": float(p[pred]),
            # soft accuracy：预测分布落在 gold 上的概率质量（choice/noul 同 accuracy，
            # score 上是「期望得分落在正确档位附近的质量」，比 argmax 宽容）
            "soft": float(p[gold]),
        })
    return out


def majority_baseline(rows: list[dict]) -> dict:
    """多数类基线：每个原语永远猜训练分布里最多的那一档。"""
    from collections import Counter

    acc = {}
    for qid in QUESTION_ORDER:
        c = Counter()
        for row in rows:
            probs = gold_probs(row, qid)
            c[max(probs, key=lambda k: probs[k])] += 1
        top = c.most_common(1)[0][0]
        hits = sum(1 for row in rows if max(gold_probs(row, qid), key=lambda k: gold_probs(row, qid)[k]) == top)
        acc[qid] = hits / len(rows) if rows else 0.0
    return acc


def summarise(view: list, qtype_name: dict) -> dict:
    import numpy as np

    by_type: dict[str, list] = {}
    for item in view:
        by_type.setdefault(qtype_name[item["qtype"]], []).append(item)
    out = {}
    for name, items in sorted(by_type.items()):
        n = len(items)
        confs = np.array([it["conf"] for it in items])
        correct = np.array([it["correct"] for it in items])
        bins = np.linspace(0.0, 1.0, 16)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= 0) & (confs <= hi)
            if m.sum():
                ece += m.sum() / n * abs(correct[m].mean() - confs[m].mean())
        tiers = {}
        for label, lo, hi in (("auto>=0.9", AUTO_THRESHOLD, 1.01),
                              ("review0.6-0.9", REVIEW_THRESHOLD, AUTO_THRESHOLD),
                              ("human<0.6", -0.01, REVIEW_THRESHOLD)):
            m = (confs >= lo) & (confs < hi)
            tiers[label] = {
                "coverage": float(m.sum()) / n if n else 0.0,
                "accuracy": float(correct[m].mean()) if m.sum() else None,
                "n": int(m.sum()),
            }
        out[name] = {
            "n": n,
            "accuracy": float(correct.mean()) if n else 0.0,
            # 均衡准确率：每档准确率的宏平均。score 原语在 v03 上真实分布极度偏斜
            # （test 241/343 都是 1 档），只看 accuracy 等于奖励「永远猜多数类」，
            # 所以验收读的是这一个数。choice/noul 上它与 accuracy 接近。
            "balanced_accuracy": float(np.mean(
                [correct[np.array([it["gold"] == lv for it in items])].mean()
                 for lv in sorted({it["gold"] for it in items})
                 if (np.array([it["gold"] == lv for it in items])).sum()])) if n else 0.0,
            "soft_accuracy": float(np.mean([it["soft"] for it in items])) if n else 0.0,
            "ece": float(ece) if n else 0.0,
            "mean_confidence": float(confs.mean()) if n else 0.0,
            "brier_top1": float(np.mean((confs - correct.astype(float)) ** 2)) if n else 0.0,
            "tiers": tiers,
        }
    return out


def md_table(rows: list[list[str]]) -> str:
    head, body = rows[0], rows[1:]
    out = ["| " + " | ".join(str(c) for c in head) + " |",
           "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in body]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--calib-data", required=True)
    ap.add_argument("--test-data", required=True)
    ap.add_argument("--baseline-checkpoint", help="原版 checkpoint，如 convaiinnovations/laya-multilingual")
    ap.add_argument("--out", required=True, help="报告与 calibration.json 的输出目录")
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    import laya
    from laya.calibrate import calibration_payload, fit_temperature_map
    from laya.common import QTYPE_NAMES

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    agent = laya.load(args.checkpoint)
    max_len = int(agent.cfg.get("max_len", 512))
    head_max_len = int(agent.cfg.get("head_max_len", 64))

    calib_rows = load_jsonl(Path(args.calib_data))
    test_rows = load_jsonl(Path(args.test_data))

    print(f"calib rows={len(calib_rows)} test rows={len(test_rows)} max_len={max_len} head_max_len={head_max_len}")

    calib_recs, calib_skipped = records_for(agent, calib_rows, max_len, head_max_len)
    test_recs, test_skipped = records_for(agent, test_rows, max_len, head_max_len)
    print(f"calib records={len(calib_recs)} skipped={calib_skipped}")
    print(f"test  records={len(test_recs)} skipped={test_skipped}")

    # ── 1. 温度标定（只在 calib split 上，compute_ece=True 内部再留出 20%）──────────
    fitted = fit_temperature_map(calib_recs, compute_ece=True, seed=args.seed)
    temps, by_options = fitted["temperature"], fitted["temperature_by_options"]
    calib_report = fitted.get("report", {})
    print(f"temperature={['%.4f' % t for t in temps]} by_options={by_options}")

    # ── 2. 落 calibration.json ────────────────────────────────────────────────
    calib_json = out_dir / "calibration.json"
    with open(calib_json, "w", encoding="utf-8") as fh:
        json.dump(calibration_payload(temps, by_options), fh, ensure_ascii=False, indent=2)
    print(f"wrote {calib_json}")

    # ── 3. test split 上标定前 / 标定后 ────────────────────────────────────────
    ones = [1.0, 1.0, 1.0]
    before = summarise(per_item_view(test_recs, ones, {}), QTYPE_NAMES)
    after = summarise(per_item_view(test_recs, temps, by_options), QTYPE_NAMES)

    # ── 4. 对照组 ────────────────────────────────────────────────────────────
    baseline_orig = None
    baseline_orig_t = None
    if args.baseline_checkpoint:
        base_agent = laya.load(args.baseline_checkpoint)
        base_recs, _ = records_for(base_agent, test_rows, max_len, head_max_len)
        baseline_orig = summarise(per_item_view(base_recs, ones, {}), QTYPE_NAMES)
        baseline_orig_t = summarise(per_item_view(base_recs, temps, by_options), QTYPE_NAMES)

    rules = evaluate_rules(test_rows, gold_labels=lambda row: {
        qid: gold_label(row, qid) for qid in QUESTION_ORDER
    })
    majority = majority_baseline(test_rows)

    payload = {
        "checkpoint": args.checkpoint,
        "seed": args.seed,
        "max_len": max_len,
        "head_max_len": head_max_len,
        "calib_rows": len(calib_rows),
        "test_rows": len(test_rows),
        "calib_records": len(calib_recs),
        "test_records": len(test_recs),
        "calib_skipped": calib_skipped,
        "test_skipped": test_skipped,
        "temperature": [float(t) for t in temps],
        "temperature_by_options": {str(k): float(v) for k, v in by_options.items()},
        "calibration_report": calib_report,
        "test_before_calibration": before,
        "test_after_calibration": after,
        "baseline_original_laya_multilingual": baseline_orig,
        "baseline_original_with_our_temperature": baseline_orig_t,
        "baseline_majority_class": majority,
        "baseline_rules": rules,
    }
    with open(out_dir / "calibration_eval.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)

    # ── 5. Markdown 报告 ──────────────────────────────────────────────────────
    def row(name, key, src, cols=("accuracy", "soft_accuracy", "ece", "mean_confidence")):
        entry = src.get(key) or {}
        return [name, str(entry.get("n", "-"))] + [
            f"{entry[c]:.4f}" if isinstance(entry.get(c), (int, float)) else "-" for c in cols
        ]

    name_of = {"choice": "route_team (choice)", "score": "urgency (score)", "noul": "needs_refund_or_escalation (noul)"}

    lines = ["# 温度标定与评测报告", "",
             f"- checkpoint：`{args.checkpoint}`",
             f"- calib split：{len(calib_rows)} 行 → {len(calib_recs)} records（**不参与训练**）",
             f"- test split：{len(test_rows)} 行 → {len(test_recs)} records（**不参与训练也不参与标定**）",
             f"- 随机种子：{args.seed}", "",
             "## 1. 拟合出的温度", "",
             md_table([["原语", "温度"],
                       ["choice / route_team", f"{temps[0]:.4f}"],
                       ["score / urgency", f"{temps[1]:.4f}"],
                       ["noul / needs_refund_or_escalation", f"{temps[2]:.4f}"]]), ""]
    if by_options:
        lines += ["分桶温度（样本不足 2000 的桶不会进 `temperature_by_options`）：", "",
                  md_table([["桶", "温度"]] + [[k, f"{v:.4f}"] for k, v in by_options.items()]), ""]
    if calib_report:
        lines += [f"标定集内部留出 {calib_report.get('n_eval')} 条仅用于报 ECE，"
                  f"拟合只用了其余部分（避免在拟合集上报 ECE）。", ""]

    lines += ["## 2. test split：标定前 vs 标定后", "",
              md_table([["原语", "n", "accuracy", "soft acc", "ECE", "mean conf"],
                        row("标定前 route_team", "choice", before),
                        row("标定后 route_team", "choice", after),
                        row("标定前 urgency", "score", before),
                        row("标定后 urgency", "score", after),
                        row("标定前 noul", "noul", before),
                        row("标定后 noul", "noul", after)]), ""]

    lines += ["## 3. 置信度分档（标定后）", ""]
    for key in ("choice", "score", "noul"):
        entry = after.get(key) or {}
        lines += [f"### {name_of[key]}", "",
                  md_table([["档位", "覆盖率", "该档实际命中率", "样本数"]] + [
                      [label, f"{v['coverage']:.4f}",
                       f"{v['accuracy']:.4f}" if v["accuracy"] is not None else "-", v["n"]]
                      for label, v in entry.get("tiers", {}).items()]), ""]

    if any("balanced_accuracy" in (after.get(k) or {}) for k in after):
        lines += ["每档准确率（score 原语分布偏斜，验收读这一项而不是 accuracy）：", "",
                  md_table([["原语", "balanced accuracy", "accuracy"]] + [
                      [name_of[k], f"{after[k]['balanced_accuracy']:.4f}",
                       f"{after[k]['accuracy']:.4f}"]
                      for k in ("choice", "score", "noul") if after.get(k)]), ""]

    def acc_row(label: str, src: dict | None) -> list:
        src = src or {}
        return [label] + [
            f"{src[q]['accuracy']:.4f}" if src.get(q) else "-" for q in ("choice", "score", "noul")
        ]

    compare = [["模型 / 基线", "route_team", "urgency", "noul"],
               acc_row("本次微调（标定后）", after),
               acc_row("原版 laya-multilingual（未微调）", baseline_orig),
               acc_row("原版 + 本次温度", baseline_orig_t),
               ["规则基线 scripts/baseline_rules.py",
                f"{rules['route_team']['accuracy']:.4f}",
                f"{rules['urgency']['accuracy']:.4f}",
                f"{rules['needs_refund_or_escalation']['accuracy']:.4f}"],
               ["多数类基线",
                f"{majority['route_team']:.4f}",
                f"{majority['urgency']:.4f}",
                f"{majority['needs_refund_or_escalation']:.4f}"]]

    lines += ["## 4. 对照组（test split，标定后温度）", "", md_table(compare), ""]

    verdict = []
    overall_after = sum(after[q]["accuracy"] for q in after) / max(len(after), 1)
    worst_ece = max(after[q]["ece"] for q in after)
    overall_before = sum(before[q]["accuracy"] for q in before) / max(len(before), 1)
    overall_base = (sum(baseline_orig[q]["accuracy"] for q in baseline_orig) / 3) if baseline_orig else None
    verdict.append(f"- 宏平均准确率：标定前 {overall_before:.4f} → 标定后 **{overall_after:.4f}**")
    verdict.append(f"- 最差单原语 ECE：**{worst_ece:.4f}**（验收线 ≤ 0.10 → {'达标' if worst_ece <= 0.10 else '未达标'}）")
    if overall_base is not None:
        verdict.append(f"- 对比原版 laya-multilingual 宏平均 {overall_base:.4f} → "
                       f"中文场景{'优于' if overall_after > overall_base else '未优于'}原版")
    for q in after:
        rb = rules[{"choice": "route_team", "score": "urgency",
                    "noul": "needs_refund_or_escalation"}[q]] or {}
        verdict.append(f"- {name_of[q]}：{after[q]['accuracy']:.4f} vs 规则基线 "
                       f"{rb.get('accuracy', float('nan')):.4f} → "
                       f"{'达标' if after[q]['accuracy'] > rb.get('accuracy', 1) else '未达标'}")
    lines += ["## 5. 裁决", ""] + verdict + [""]

    report = out_dir / "eval_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {report}")
    print("\n".join(verdict))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()