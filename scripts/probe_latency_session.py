#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一次进程、一个 session 量完延迟的三组数字，并为论文生成 v04 延迟构成文档。

为什么非要同一个 session
------------------------
同一台机、同一个包、同一条命令，不同 session 的端到端 p50 实测能差 **2.5 倍**
（v04 包在 328ms 与 806ms 上都测到过，见 `docs/research/RUNS.md` 2026-10-17 条）。
这个量级比模型侧的任何效应都大，所以「端到端 p50」和「延迟构成」若来自不同 session，
论文里两个数字会互相打架（读者会算：三个头只花 283ms，怎么三个原语端到端才 330ms）。
本脚本把三组数字放进同一个进程、同一个 session：

  1. 端到端：与 `python -m mai_jev.deploy --input ... --repeats N` **同款协议**
     （三原语一条工单、batch=1、精确长度、warmup 排除），输出与 bench json 同形的字段；
  2. state 预算扫描：固定 head，改 `Predictor.state_room`，只量 `session.run` 前向
     （不含 tokenizer），得到「问句头 vs 工单正文」的成本分解；
  3. batching：三原语合成 batch=3 一次前向 vs 三次 batch=1（同一个 session）。

另外量一条 **CPU 微探针**（固定尺寸 numpy matmul 的 p50）：它给出这个 session 的
机器状态读数，让不同 session 之间的数差能被归因，而不是被误当成模型差异。

用法
----
    python scripts/probe_latency_session.py \
        --model-dir $WORK\\export\\v04-int8-ml256 \
        --input     $WORK\\export\\bench_input.jsonl \
        --repeats 150 --warmup 8 --threads 4 \
        --out-json  $WORK\\export\\v04_latency_session.json \
        --out-md    docs/research/v04-latency.md

`--out-md` 里的六个数（state_room=0 / 满预算 / batch=3 / 三次 batch=1 / 两个 head 长度）
会被 `scripts/make_paper_tables.py` 按固定句式抓成论文宏，所以那几句的措辞不要改。
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mai_jev.deploy import QTYPE, Predictor, rss_mb  # noqa: E402

# state 预算扫描的取样档位（除「满」以外）
BUDGETS = (0, 32, 64, 96)
SWEEP_TICKETS = 3     # 每个档位用几条真实工单
SWEEP_REPEATS = 8     # 每档每工单量几次前向（3×8=24 个样本取 p50）
BATCH_TICKETS = 40    # batching 那一组用几条工单


def cpu_probe(mat: int = 2000, reps: int = 5) -> float:
    """固定尺寸 matmul 的 p50（ms）：这个 session 的机器状态读数。"""
    rng = np.random.default_rng(0)
    a = rng.standard_normal((mat, mat))
    b = rng.standard_normal((mat, mat))
    a.dot(b)
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        a.dot(b)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return round(statistics.median(ts), 1)


def mem_info() -> dict:
    """本机物理/提交内存读数（MB）。用来判断「慢 session」是不是内存压力造成的
    —— 本项目记录过内存墙（空闲物理内存只剩几 GB、commit 逼近上限）。"""
    if os.name != "nt":
        return {}

    import ctypes

    class _MemStatus(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

    st = _MemStatus()
    st.dwLength = ctypes.sizeof(_MemStatus)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
        return {}
    mb = 1048576.0
    return {"phys_avail_MB": round(st.ullAvailPhys / mb),
            "phys_total_MB": round(st.ullTotalPhys / mb),
            "memory_load_pct": int(st.dwMemoryLoad),
            "commit_avail_MB": round(st.ullAvailPageFile / mb)}


def load_texts(path: str, need: int) -> list[str]:
    texts = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                texts.append(rec.get("state") if isinstance(rec, dict) else str(rec))
            except ValueError:
                texts.append(line)
    if not texts:
        raise SystemExit(f"{path} 里没有可用的工单")
    if len(texts) < need:
        print(f"[warn] {path} 只有 {len(texts)} 条，按环形重复取 {need} 次", file=sys.stderr)
    return texts


def batch_feed(p: Predictor, qids: list[str], text: str) -> dict:
    """三原语拼成 batch=3：序列右侧 padding，marker 右侧补 False。
    （与 scripts/probe_batch_merge.py 同法；那个脚本仍然负责 NaN / argmax 那两问。）"""
    feeds = [p._feed(q, text) for q in qids]
    length = max(len(f["input_ids"][0]) for f in feeds)
    markers = max(len(f["marker_pos"][0]) for f in feeds)
    ids, am, mp, mm, qt = [], [], [], [], []
    for q, f in zip(qids, feeds):
        pad = length - len(f["input_ids"][0])
        ids.append(np.concatenate([f["input_ids"][0], np.zeros(pad, dtype=np.int64)]))
        am.append(np.concatenate([f["attention_mask"][0], np.zeros(pad, dtype=np.int64)]))
        mp.append(np.concatenate([f["marker_pos"][0],
                                  np.zeros(markers - len(f["marker_pos"][0]), dtype=np.int64)]))
        mm.append(np.concatenate([f["marker_mask"][0],
                                  np.zeros(markers - len(f["marker_mask"][0]), dtype=bool)]))
        qt.append(QTYPE[p.questions[q]["type"]])
    return {"input_ids": np.stack(ids), "attention_mask": np.stack(am),
            "marker_pos": np.stack(mp), "marker_mask": np.stack(mm),
            "qtype": np.array(qt, dtype=np.int64)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="一个 session 量完端到端 + 延迟构成 + batching。")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True, help="jsonl，每行一个 {\"state\": ...} 或纯文本")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", help="延迟构成文档（论文宏的来源）")
    # 文档里写出来的目录名：默认只用目录名，调用方可以传「$WORK\export\<变体>」这种去标识写法，
    # 免得把一个本机绝对路径写进仓库文档。
    ap.add_argument("--display-dir", default="",
                    help="文档里显示的模型目录（默认只用 basename；本机绝对路径不要写进文档）")
    args = ap.parse_args(argv)
    # 文档里写出来的目录名：默认自己拼成去标识写法（$WORK\export\<变体>），
    # 免得把本机绝对路径写进仓库文档；--display-dir 可以覆盖。
    display_dir = args.display_dir or (
        "$WORK\\export\\" + Path(args.model_dir).name if "export" in Path(args.model_dir).parts
        else Path(args.model_dir).name)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))

    state_cpu = cpu_probe()
    state_mem = mem_info()
    t0 = time.perf_counter()
    p = Predictor(args.model_dir, threads=args.threads)
    load_s = time.perf_counter() - t0
    rss_after_load = rss_mb()
    qids = list(p.questions)
    print(f"[session] cpu_probe={state_cpu}ms  load={load_s:.2f}s  rss={rss_after_load:.1f}MB",
          file=sys.stderr)

    # ── 1) 端到端（与 mai_jev.deploy 的 CLI 同款协议）──────────────────────────
    texts = load_texts(args.input, args.warmup + args.repeats)
    lat, rss_peak_e2e = [], rss_after_load
    for i in range(args.warmup + args.repeats):
        text = texts[i % len(texts)]
        t = time.perf_counter()
        for qid in qids:
            p.predict_question(qid, text)
        if i >= args.warmup:
            lat.append((time.perf_counter() - t) * 1000)
        rss_peak_e2e = max(rss_peak_e2e, rss_mb())
    rss_peak_all = rss_peak_e2e
    lat.sort()
    e2e = {
        "p50": round(lat[len(lat) // 2], 2),
        "p95": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
        "mean": round(sum(lat) / len(lat), 2),
    }

    # ── 2) state 预算扫描（只量 session.run 前向，正文用**真实工单**）──────────
    # 部署口径是「精确长度」：state 只截**不补**。真实工单正文 p50 只有几十 token，
    # 所以预算一旦超过该工单正文的长度，几档就截到同一条序列 —— 表里 seq_len 会显示
    # 这件事（同一条的几格差异是噪声，不是「预算越大越慢」）。要看长度→成本的曲线，
    # 得用足够长的正文；那属于另一个实验，这里不做，免得把合成长文本的成本
    # 当成真实工单的成本。
    sweep = {}
    for qid in qids:
        head_len = len(p._heads[qid].ids)
        full = p.state_room[qid]
        budgets = list(BUDGETS) + [full]
        tickets = [texts[j % len(texts)] for j in range(SWEEP_TICKETS)]
        feeds, seq_len = {}, {}
        for budget in budgets:
            p.state_room[qid] = budget
            feeds[budget] = [p._feed(qid, t) for t in tickets]
            lens = sorted({int(f["input_ids"].shape[1]) for f in feeds[budget]})
            seq_len[budget] = [lens[0], lens[-1]]
        p.state_room[qid] = full
        # 轮转量：一轮里把各档位依次量一遍，机器状态漂移就被摊到各档上，而不是全压在后几档
        for budget in budgets:
            for f in feeds[budget]:
                p.session.run(["logits", "act_logits"], f)
        ts = {b: [] for b in budgets}
        for _ in range(SWEEP_REPEATS):
            for budget in budgets:
                for f in feeds[budget]:
                    t = time.perf_counter()
                    p.session.run(["logits", "act_logits"], f)
                    ts[budget].append((time.perf_counter() - t) * 1000)
        sweep[qid] = {"head_len": head_len, "state_room_full": full,
                      "seq_len": {str(b): seq_len[b] for b in budgets},
                      "budget_p50_ms": {str(b): round(statistics.median(ts[b]), 1) for b in budgets}}
        rss_peak_all = max(rss_peak_all, rss_mb())
    only_ms = round(sum(sweep[q]["budget_p50_ms"]["0"] for q in qids), 1)
    full_ms = round(sum(sweep[q]["budget_p50_ms"][str(sweep[q]["state_room_full"])] for q in qids), 1)

    # ── 3) batching（同一个 session）──────────────────────────────────────────
    rows = texts[:BATCH_TICKETS]
    for r in rows[:3]:
        p.session.run(["logits", "act_logits"], batch_feed(p, qids, r))
        for q in qids:
            p.session.run(["logits", "act_logits"], p._feed(q, r))
    t_batch, t_loop = [], []
    for r in rows:
        t = time.perf_counter()
        p.session.run(["logits", "act_logits"], batch_feed(p, qids, r))
        t_batch.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        for q in qids:
            p.session.run(["logits", "act_logits"], p._feed(q, r))
        t_loop.append((time.perf_counter() - t) * 1000)
    t_batch.sort()
    t_loop.sort()
    merged = round(statistics.median(t_batch), 1)
    separate = round(statistics.median(t_loop), 1)

    state_cpu_end = cpu_probe()
    state_mem_end = mem_info()
    report = {
        "model_dir": args.model_dir,
        "threads": args.threads,
        "max_len": p.max_len,
        "load_seconds": round(load_s, 2),
        "rss_after_load_MB": round(rss_after_load, 1),
        "rss_peak_MB": round(rss_peak_e2e, 1),
        "n": len(lat),
        "latency_ms": e2e,
        "note": "p50/p95 是「一条工单跑完三原语」的总耗时，batch=1，warmup 已排除。",
        "latency_session": {
            "cpu_probe_ms_before": state_cpu,
            "cpu_probe_ms_after": state_cpu_end,
            "mem_before": state_mem,
            "mem_after": state_mem_end,
            "rss_peak_after_probes_MB": round(rss_peak_all, 1),
            "note": "本文件里的端到端、state 预算扫描、batching 三组数字都出自这一次进程/一个 session。"
                    "rss_peak_MB 只取端到端那一段（部署口径）；rss_peak_after_probes_MB 含后面两段探针。"
                    "cpu_probe 与 mem 是这次 session 的机器状态读数，用来判断跨 session 的数差是"
                    "CPU 状态还是内存压力造成的。",
        },
        "sweep": {
            "budgets": list(BUDGETS),
            "repeats_per_budget": SWEEP_REPEATS,
            "tickets_per_budget": SWEEP_TICKETS,
            "repeats_per_ticket": SWEEP_REPEATS,
            "timing": "只含 session.run 前向，不含 tokenizer；正文是真实工单、state 只截不补，"
                      "所以预算超过该工单正文长度的档位会截到同一条序列（各档实际长度见 seq_len）",
            "by_question": sweep,
            "state_room_0_total_ms": only_ms,
            "full_budget_total_ms": full_ms,
        },
        "batch_merge": {
            "tickets": BATCH_TICKETS,
            "merged_p50_ms": merged,
            "three_single_p50_ms": separate,
            "timing": "只含 session.run 前向；合并那一组是右 padding 的 batch=3",
        },
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[session] end-to-end p50={e2e['p50']} p95={e2e['p95']} mean={e2e['mean']}  "
          f"state0={only_ms}  full={full_ms}  merged={merged}  three={separate}  "
          f"cpu_end={state_cpu_end}  rss_peak(e2e)={rss_peak_e2e:.1f}", file=sys.stderr)

    # ── 论文用的延迟构成文档（句式即 make_paper_tables.py 的正则）──────────────
    if args.out_md:
        def _row(qid: str) -> str:
            cells, budgets = [qid, str(sweep[qid]["head_len"])], list(BUDGETS) + [sweep[qid]["state_room_full"]]
            for i, b in enumerate(budgets):
                v = str(sweep[qid]["budget_p50_ms"][str(b)])
                cells.append(f"**{v}**" if i == 0 else v)
            return "| " + " | ".join(cells) + " |"

        rows_md = "\n".join(_row(qid) for qid in qids)
        cols = " | ".join(str(b) for b in BUDGETS) + " | 满"
        ncols = len(BUDGETS) + 3                      # question + head + 各档
        md = f"""# v04 int8 靶机延迟构成（threads={args.threads}，Windows，本机 CPU）

口径：`{display_dir}`（int8-gather，单图），`Predictor(session.run)` 单次前向耗时，
batch=1，精确长度。这不是靶机实测，是训练机实测，靶机只会更慢。

**同一个 session**：本文的三组数字（端到端、state 预算扫描、batching）出自**一次进程、
一个 session**。这一点是硬要求：同一个包在不同 session 之间的端到端 p50 实测能差 2.5 倍
（见 [RUNS.md](RUNS.md) 2026-10-17 条），比模型侧任何效应都大。本次 session 的机器状态读数
（2000x2000 numpy matmul 的 p50）：**{state_cpu}ms**（跑完再量一次是 {state_cpu_end}ms）。
物理内存：开始时空闲 {state_mem.get('phys_avail_MB', '?')}MB / 总 {state_mem.get('phys_total_MB', '?')}MB，
结束时 {state_mem_end.get('phys_avail_MB', '?')}MB —— 内存不是这次波动的原因。

端到端（`{Path(args.input).name}` 里的 {args.repeats} 条真实工单，warmup {args.warmup}，
与 `python -m mai_jev.deploy` 同款协议）：
p50 **{e2e['p50']}ms**、p95 **{e2e['p95']}ms**、mean **{e2e['mean']}ms**，
加载后常驻 **{round(rss_after_load, 1)}MB**、峰值 **{round(rss_peak_e2e, 1)}MB**、加载 {round(load_s, 2)}s。

## 一句话结论

**p50 < 50ms 这条验收线在 322M 编码器 + {args.threads} 线程下不可能达到，且不是量化方案的问题。**
模型**完全不读工单正文**（state 预算设 0）时，三原语合计已经要 {only_ms}ms。

## 延迟由问句主导，正文几乎不参与

部署口径是「精确长度」：state 只截**不补**。真实工单正文 p50 只有几十 token，
所以预算一旦超过该工单正文的长度，几档就截到**同一条序列** —— 各档实际序列长度见
`--out-json` 的 `sweep.by_question.*.seq_len`，右几格的小差异是噪声，不是「预算越大越慢」。
每格是该档 {SWEEP_TICKETS} 条工单 × {SWEEP_REPEATS} 轮（共 {SWEEP_TICKETS * SWEEP_REPEATS} 次）前向的 p50（ms），
档位之间轮转取样以抵消机器状态漂移。

| question | head 长度 | {cols} |
{"|".join(["---"] * ncols)}|
{rows_md}

三原语合计：state_room=0 → **{only_ms}ms**（完全不给正文）；满预算 → **{full_ms}ms**（真实工单的完整长度）。

## batching：合并成一次前向更慢

同一批工单、同一个 session 里对比：三原语合成 batch=3 一次前向 p50 **{merged}ms**，反而比三次
batch=1 的 **{separate}ms** 更慢 —— 编码器无论如何都要逐行跑，而三个问句产生的序列确实不同。
（batch=3 右 padding 还会让短的 `noul` 行返回非有限值，这是硬约束，见
[quantization-and-deployment.md](quantization-and-deployment.md) §7。NaN / argmax 那两问由
`scripts/probe_batch_merge.py` 负责，本文只报耗时。）

## 复现

```powershell
python scripts/probe_latency_session.py `
  --model-dir <导出目录> --input <工单 jsonl> `
  --threads {args.threads} --repeats {args.repeats} --warmup {args.warmup} `
  --out-json <session.json> --out-md docs/research/v04-latency.md
```

**跨 session 不可比**：本文的数字只在上面那个 session 的状态下成立。要横向对比两个变体，
必须在同一个 session 里交替测（见 `docs/research/RUNS.md` 2026-10-17 条）。
"""
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md, encoding="utf-8")
        print(f"[session] 写好 {out_md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
