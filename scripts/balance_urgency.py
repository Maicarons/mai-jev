"""按 urgency 各档均衡重采样训练集（ticket #16 的必要配套）。

为什么需要它：`scripts/label_urgency.py` 在真实语料上判出 73% 是「无任何紧急证据」
（evidence=`no_evidence` 2336/3209 行）。这个分布是真实的，但在 score 原语上，
多数类基线直接给 0.7022（test split 241/343），**准确率验收线 0.75 会变成一句废话** ——
学成「永远输出 1」就能过。

所以本脚本做两件事：

1. 把 train split 按 urgency 档位**过采样到最大类**（保留 `cap` 上限，避免无脑复制），
   calib/test 一律不动 —— 评测集必须保持真实分布，否则均衡后的准确率是自欺。
2. 报告 `majority_baseline` 与 `balanced_accuracy` 两个数，让任何读报告的人
   一眼看出模型是不是只会猜多数类。

用 `make_row` 的 `gold.urgency.probabilities` 取 argmax 定档（key 已是 0-based）。
过采样是**精确复制**而非加权 loss，所以不改动训练脚本，只多产出一个 jsonl。
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def urgency_level(rec: dict) -> str:
    probs = (rec.get("gold") or {}).get("urgency", {}).get("probabilities") or {}
    if not probs:
        return "?"
    return max(probs, key=lambda k: probs[k])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True, help="均衡后的 train jsonl")
    ap.add_argument("--prefix", default="mai_jev")
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--target", type=int, help="每档目标行数（默认取最大档）")
    ap.add_argument("--cap", type=int, default=4, help="少数档最多复制几倍")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = [json.loads(l) for l in
            (in_dir / f"{args.prefix}_train.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rng = random.Random(args.seed)

    buckets: dict[str, list[dict]] = defaultdict(list)
    for rec in rows:
        buckets[urgency_level(rec)].append(rec)

    natural = {k: len(v) for k, v in sorted(buckets.items())}
    target = args.target or max(natural.values())
    out: list[dict] = []
    for level in sorted(buckets):
        pool = buckets[level]
        if not pool:
            continue
        if len(pool) >= target:
            out.extend(pool)
            continue
        reps = min(args.cap, -(-target // len(pool)))
        out.extend(pool * reps)
        out.extend(rng.choices(pool, k=target - len(pool) * reps))
    rng.shuffle(out)

    with Path(args.out).open("w", encoding="utf-8") as fh:
        for rec in out:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    balanced = Counter(urgency_level(r) for r in out)
    print(f"train rows {len(rows)} -> {len(out)}")
    print("  natural  ", natural)
    print("  balanced ", {k: balanced.get(k, 0) for k in sorted(natural)})

    for split in ("calib", "test"):
        p = in_dir / f"{args.prefix}_{split}.jsonl"
        rows_s = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        c = Counter(urgency_level(r) for r in rows_s)
        n = sum(c.values())
        top = max(c.values())
        print(f"  {split}: n={n} dist={dict(sorted(c.items()))} majority_baseline={top / n:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())