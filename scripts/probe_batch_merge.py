"""Probe: 能不能把同一 state 的三原语合成一次前向？

#8 报告 1.5 说右 padding 的混合 batch 会让短的 qtype=2 行返回 NaN，根因指向
ModernBERT sliding-window attention 与 padding 的交互。这里直接测三件事：

1. 三原语拼成 batch=3（右侧 padding）后是否出现非有限值；
2. 与逐条 batch=1 相比 argmax 是否翻转；
3. 一次 batch=3 前向 vs 三次 batch=1 前向各花多少时间。

不修改 deploy.py，纯探针。
"""
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, r".")
from mai_jev.deploy import QTYPE, Predictor  # noqa: E402

EXPORT = sys.argv[1] if len(sys.argv) > 1 else r"$WORK\export\v03-int8-ml256"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

p = Predictor(EXPORT, threads=THREADS)
qids = list(p.questions)
rows = [json.loads(l) for l in
        open(r"$WORK\data\adapted\mai_jev_test.jsonl", encoding="utf-8")][:N]


def batch_feed(text):
    """把三原语拼成 batch=3：序列右侧 padding，marker 右侧补 False。"""
    feeds = [p._feed(q, text) for q in qids]
    L = max(len(f["input_ids"][0]) for f in feeds)
    M = max(len(f["marker_pos"][0]) for f in feeds)
    ids, am, mp, mm, qt = [], [], [], [], []
    for q, f in zip(qids, feeds):
        n = len(f["input_ids"][0])
        pad = L - n
        ids.append(np.concatenate([f["input_ids"][0], np.zeros(pad, dtype=np.int64)]))
        am.append(np.concatenate([f["attention_mask"][0], np.zeros(pad, dtype=np.int64)]))
        mp.append(np.concatenate([f["marker_pos"][0],
                                  np.zeros(M - len(f["marker_pos"][0]), dtype=np.int64)]))
        mm.append(np.concatenate([f["marker_mask"][0],
                                  np.zeros(M - len(f["marker_mask"][0]), dtype=bool)]))
        qt.append(QTYPE[p.questions[q]["type"]])
    return {"input_ids": np.stack(ids), "attention_mask": np.stack(am),
            "marker_pos": np.stack(mp), "marker_mask": np.stack(mm),
            "qtype": np.array(qt, dtype=np.int64)}


# 预热
for r in rows[:3]:
    p.session.run(["logits", "act_logits"], batch_feed(r["state"]))
    for q in qids:
        p.predict_question(q, r["state"])

nonfinite = flips = total = 0
for r in rows:
    f = batch_feed(r["state"])
    lg, _ = p.session.run(["logits", "act_logits"], f)
    for i, q in enumerate(qids):
        ref = p.predict_question(q, r["state"])
        k = len(ref["labels"])
        total += 1
        if not np.all(np.isfinite(lg[i][:k])):
            nonfinite += 1
        elif int(np.argmax(lg[i][:k])) != int(np.argmax(np.asarray(ref["probs"]))):
            flips += 1

print("rows=%d  non-finite %d/%d   argmax flips vs batch=1: %d"
      % (len(rows), nonfinite, total, flips))

t_batch, t_loop = [], []
for r in rows:
    f = batch_feed(r["state"])
    t = time.perf_counter()
    p.session.run(["logits", "act_logits"], f)
    t_batch.append((time.perf_counter() - t) * 1000)

    t = time.perf_counter()
    for q in qids:
        fq = p._feed(q, r["state"])
        p.session.run(["logits", "act_logits"], fq)
    t_loop.append((time.perf_counter() - t) * 1000)

for name, ts in (("batch=3 一次前向", t_batch), ("三次 batch=1", t_loop)):
    ts.sort()
    print("%-16s p50=%7.1f  p95=%7.1f  mean=%7.1f"
          % (name, statistics.median(ts), ts[int(0.95 * len(ts))], statistics.mean(ts)))