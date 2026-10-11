"""mai-jev：把原始中文客服语料转成 laya 三原语训练/评测 jsonl。

输入（jsonl，每行一条，UTF-8）：
  {"id": "...", "state": "工单正文", "split": "train|calib|test",
   "labels": {"route_team": "退款处理", "urgency": "4",
              "needs_refund_or_escalation": "true"}}

输出（三件套，都在 --out 目录下）：
  <prefix>_train.jsonl / <prefix>_calib.jsonl / <prefix>_test.jsonl
  每行形如：
  {"id": "...", "state": "...",
   "questions": {...QUESTIONS...},
   "labels": {"route_team": "退款处理", "urgency": 4,
              "needs_refund_or_escalation": true}}

设计原则：
  - 标签在落盘前统一校验（route_team 属于 ROUTE_TEAMS、urgency 属于 1..5、
    noul 属于 {true,false}），非法行被拒并计数，不静默丢弃。
  - split 只允许 train/calib/test；calib 只用于温度标定，绝不进训练。
  - 支持 --dedup（按 state 文本精确去重）与 --min-len / --max-len 过滤。
  - 默认干跑只统计，加 --write 才落盘。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  G:\\mai-jev-work\\venv\\Scripts\\python.exe G:\\GitHub\\mai-jev\\scripts\\build_dataset.py ^
      --inputs G:\\mai-jev-work\\data\\raw\\*.jsonl --out G:\\mai-jev-work\\data ^
      --prefix v0.1 --write
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mai_jev.questions import (  # noqa: E402
    NOUL_LABELS,
    QUESTIONS,
    ROUTE_TEAMS,
    URGENCY_LEVELS,
)

SPLITS = ("train", "calib", "test")


def normalize_noul(value: object) -> str | None:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "是"}:
            return "true"
        if v in {"false", "0", "no", "否"}:
            return "false"
    if isinstance(value, (int, float)):
        return "true" if value else "false"
    return None


def normalize_urgency(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        n = int(value)
        return str(n) if 1 <= n <= 5 else None
    if isinstance(value, str):
        v = value.strip()
        if v in URGENCY_LEVELS:
            return v
        if v.isdigit() and 1 <= int(v) <= 5:
            return v
    return None


def validate(record: dict, idx: int, source: str) -> tuple[dict | None, str | None]:
    """返回 (规范化后的记录, 错误原因)。"""
    state = record.get("state")
    if not isinstance(state, str) or not state.strip():
        return None, "missing state"
    labels_in = record.get("labels")
    if not isinstance(labels_in, dict):
        return None, "missing labels"

    route = labels_in.get("route_team")
    if route not in ROUTE_TEAMS:
        return None, f"bad route_team={route!r}"
    urg = normalize_urgency(labels_in.get("urgency"))
    if urg is None:
        return None, f"bad urgency={labels_in.get('urgency')!r}"
    noul = normalize_noul(labels_in.get("needs_refund_or_escalation"))
    if noul is None:
        return None, f"bad needs_refund_or_escalation={labels_in.get('needs_refund_or_escalation')!r}"

    split = record.get("split", "train")
    if split not in SPLITS:
        return None, f"bad split={split!r}"

    rid = record.get("id") or f"{Path(source).stem}:{idx}"
    return (
        {
            "id": str(rid),
            "state": state.strip(),
            "questions": QUESTIONS,
            "labels": {
                "route_team": route,
                "urgency": int(urg),
                "needs_refund_or_escalation": noul == "true",
            },
            "split": split,
        },
        None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="原始 jsonl（支持 glob）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--prefix", default="v0.1")
    ap.add_argument("--write", action="store_true", help="缺省只统计不落盘")
    ap.add_argument("--dedup", action="store_true", help="按 state 精确去重")
    ap.add_argument("--min-len", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2000)
    args = ap.parse_args()

    files: list[str] = []
    for pattern in args.inputs:
        files.extend(glob.glob(pattern))
    if not files:
        print("no input files matched", file=sys.stderr)
        raise SystemExit(2)

    out_dir = Path(args.out)
    if args.write:
        out_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    buckets: dict[str, list[dict]] = {s: [] for s in SPLITS}
    route_counter: Counter[str] = Counter()
    urg_counter: Counter[str] = Counter()
    noul_counter: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    total = 0

    for path in files:
        with open(path, encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    rejected[f"json:{exc.msg}"] += 1
                    continue
                rec, err = validate(raw, idx, path)
                if rec is None:
                    rejected[str(err)] += 1
                    continue
                state = rec["state"]
                if len(state) < args.min_len or len(state) > args.max_len:
                    rejected["length"] += 1
                    continue
                if args.dedup:
                    if state in seen:
                        rejected["duplicate"] += 1
                        continue
                    seen.add(state)
                buckets[rec["split"]].append(rec)
                route_counter[rec["labels"]["route_team"]] += 1
                urg_counter[str(rec["labels"]["urgency"])] += 1
                noul_counter["true" if rec["labels"]["needs_refund_or_escalation"] else "false"] += 1

    print(f"files            : {len(files)}")
    print(f"lines read       : {total}")
    for split in SPLITS:
        print(f"  {split:<6}: {len(buckets[split])}")
    print(f"rejected         : {sum(rejected.values())}")
    for reason, n in rejected.most_common():
        print(f"    {reason}: {n}")
    print("route_team dist  :", dict(route_counter))
    print("urgency dist     :", dict(sorted(urg_counter.items())))
    print("noul dist        :", dict(noul_counter))

    if args.write:
        for split in SPLITS:
            target = out_dir / f"{args.prefix}_{split}.jsonl"
            with open(target, "w", encoding="utf-8") as fh:
                for rec in buckets[split]:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"wrote {target} ({len(buckets[split])} rows)")
    else:
        print("dry run: pass --write to emit files")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
