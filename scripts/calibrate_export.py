"""为 ONNX 导出变体在它自己的 calib 划分上重拟合温度。

#8 的结论：温度是「权重 + 推理后端」的耦合属性 —— 同一套流程在 torch 与 fp32 ONNX
上跑出的温度差别很大。所以每个量化变体必须配一份在**该图上**拟合的 calibration.json，
跨变体复用会让 ECE 悄悄失效。

用法：
    python scripts/calibrate_export.py --model-dir $WORK\\export\\v03-int8-ml256 \\
        --calib ...\\mai_jev_calib.jsonl --test ...\\mai_jev_test.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mai_jev.deploy import Predictor  # noqa: E402


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def gold_vector(row: dict, qid: str, k: int, labels: list[str], qtype: str) -> np.ndarray:
    """`gold[qid]['probabilities']` -> 长度 k 的概率向量。

    三种原语的 gold key 语义不同，不能一律按整数排：
      - choice：key 是 criteria 名字（`售后` / `退款处理` …），要按 head 的选项顺序对齐；
      - score：key 是 0-based 档位下标 `"0".."4"`（`laya/train.py:187`），按整数排；
      - noul：key 是 `"false"` / `"true"`，按选项顺序排。
    """
    probs = row["gold"][qid]["probabilities"]
    vec = np.zeros(k, dtype=float)
    for i, label in enumerate(labels):
        if qtype == "choice":
            # key 是 criteria 名字（`售后` / `退款处理` …）
            value = probs.get(label, 0.0)
        elif qtype == "score":
            # key 是 0-based 档位下标 `"0".."K-1"`（`laya/train.py:187`）
            value = probs.get(str(i), 0.0)
        else:
            # noul 的 key 就是选项名 `"false"` / `"true"`；head 里渲染成
            # `false: no, ...` / `true: yes, ...`，所以要按名字取而不是按下标。
            value = probs.get(label, 0.0)
        if i < k:
            vec[i] = float(value)
    if vec.sum() <= 0:
        vec[:] = 1.0 / k
    return vec / vec.sum()


def records_for(predictor: Predictor, rows: list[dict]) -> list:
    """jsonl 行 -> `(qtype, logits, target, k)`，与 laya 的内部记录格式一致。"""
    recs = []
    for row in rows:
        state = row["state"]
        for qid in predictor.questions:
            feed = predictor._feed(qid, state)
            logits, _ = predictor.session.run(["logits", "act_logits"], feed)
            k = len(predictor._heads[qid].markers)
            qtype = predictor.qtype[qid]
            labels = [o.split(":", 1)[0] for o in predictor._heads[qid].options]
            recs.append((qtype, logits[0][:k].astype(float),
                         gold_vector(row, qid, k, labels, predictor.questions[qid]["type"]), k))
    return recs


def ece(view: list, bins: int = 15) -> float:
    total, acc = len(view), 0.0
    if not total:
        return float("nan")
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [r for r in view if lo <= r["conf"] < hi or (b == bins - 1 and r["conf"] == 1.0)]
        if not bucket:
            continue
        acc += len(bucket) / total * abs(
            float(np.mean([r["conf"] for r in bucket])) - float(np.mean([r["correct"] for r in bucket])))
    return acc


def summarize(recs: list, temps: list, by_options: dict) -> list[dict]:
    out = []
    for qtype in (0, 1, 2):
        sub = [r for r in recs if r[0] == qtype]
        if not sub:
            continue
        view = []
        for _qt, logits, target, k in sub:
            bucket = Predictor._bucket(qtype, k)
            scale = float(by_options.get(bucket, temps[qtype]))
            z = logits[:k] / max(scale, 1e-4)
            z = z - z.max()
            e = np.exp(z)
            p = e / e.sum()
            pred = int(np.argmax(p))
            view.append({"correct": float(pred == int(np.argmax(target))), "conf": float(p[pred])})
        out.append({
            "qtype": ["choice", "score", "noul"][qtype],
            "n": len(view),
            "accuracy": round(float(np.mean([v["correct"] for v in view])), 4),
            "mean_confidence": round(float(np.mean([v["conf"] for v in view])), 4),
            "ece": round(ece(view), 4),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calib", required=True, help="calib jsonl（v0.2 schema）")
    ap.add_argument("--test", help="给了就顺便报 test 上的准确率与 ECE")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--write", action="store_true", help="覆盖 <model-dir>/calibration.json")
    args = ap.parse_args()

    from laya.calibrate import fit_temperature_map

    predictor = Predictor(args.model_dir, threads=args.threads)
    old = list(predictor.temperature)

    calib_recs = records_for(predictor, load_jsonl(args.calib))
    fitted = fit_temperature_map(calib_recs, compute_ece=True, seed=args.seed)
    temps = [float(t) for t in fitted["temperature"]]
    by_options = dict(fitted.get("temperature_by_options") or {})

    print(f"model_dir  : {args.model_dir}")
    print(f"max_len    : {predictor.max_len}   threads: {args.threads}")
    print(f"旧温度(torch 侧带过来的): {[round(t, 4) for t in old]}")
    print(f"新温度(本图上拟合)      : {[round(t, 4) for t in temps]}")
    if by_options:
        print(f"分桶温度: {by_options}")

    print("\ncalib（拟合集，%d 条记录）" % len(calib_recs))
    print(f"  {'primitive':<12}{'n':>6}{'acc':>9}{'conf':>9}{'ece':>9}")
    for row in summarize(calib_recs, temps, by_options):
        print(f"  {row['qtype']:<12}{row['n']:>6}{row['accuracy']:>9.4f}"
              f"{row['mean_confidence']:>9.4f}{row['ece']:>9.4f}")

    if args.test:
        test_recs = records_for(predictor, load_jsonl(args.test))
        print("\ntest（旧温度 vs 新温度）")
        print(f"  {'primitive':<12}{'n':>6}{'acc':>9}{'ece(旧)':>10}{'ece(新)':>10}")
        before = {r["qtype"]: r for r in summarize(test_recs, old, {})}
        after = {r["qtype"]: r for r in summarize(test_recs, temps, by_options)}
        for key in after:
            b, a = before.get(key), after[key]
            print(f"  {key:<12}{a['n']:>6}{a['accuracy']:>9.4f}"
                  f"{b['ece']:>10.4f}{a['ece']:>10.4f}")
        worst = max(r["ece"] for r in after.values())
        print(f"  最差单原语 ECE = {worst:.4f} {'PASS' if worst <= 0.10 else 'FAIL'} (目标 ≤0.10)")

    if args.write:
        payload = {
            "version": 2,
            "temperature": temps,
            "temperature_by_options": by_options,
            "model_id_or_path": None,
            "subfolder": None,
            "config": None,
        }
        out = Path(args.model_dir) / "calibration.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()