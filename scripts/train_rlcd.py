"""mai-jev：把三原语数据转成 laya 训练/标定 jsonl，并调用 laya-train。

链路：
  原始 jsonl --(scripts/build_dataset.py)--> v0.1_{train,calib,test}.jsonl
            --(本脚本 convert)-------> v0.1_{train,calib,test}.laya.jsonl
            --(laya-train)-----------> checkpoint

为什么要有这一层：
  build_dataset 的输出带 `labels` / `split` / `id`，是给评测和审计看的；
  laya-train 只认 `{state, questions, expected}`（docs/finetune.md:50-70）。
  这一层同时负责两件 laya 不会替我们做的事：
    1. `urgency` 从我们的 1..5 减成 0-based index（laya/train.py:265：
       `ValueError: expected score 5 out of bounds [0, 4]`）。
    2. state 里的 `[MASK]` 清成空格（laya/train.py:287-289 encode_state 也会
       替换，但那时只报 skip，我们提前清掉能少丢数据）。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'

  # 1) 转格式（干跑默认只统计）
  python scripts\\train_rlcd.py ^
      convert --input-dir $WORK\\data --prefix v0.1 --write

  # 2) 冒烟：3k 样本跑通管线（应先单独调小样本）
  python scripts\\train_rlcd.py ^
      run --data $WORK\\data\\v0.1_train.laya.jsonl ^
      --eval-data $WORK\\data\\v0.1_test.laya.jsonl ^
      --out $WORK\\runs\\smoke --epochs 1 --max-samples 3000 --dry-run

  # 3) 全量
  ... run --out $WORK\\ckpt\\v0.1 --epochs 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mai_jev.questions import QUESTIONS, SCORE_OFFSET  # noqa: E402

# `laya -m` 不可用（laya 包没有 __main__.py），`laya train` 才是等价入口；但 venv 的
# Scripts\laya.exe 不在 PATH 里，subprocess 会 FileNotFoundError。
# 方案：直接用 venv python 调 `-c "from laya.cli import main; sys.exit(main(['train', ...]))"`
# —— 和 laya-train.exe 走的是同一个 main()（laya/cli.py:394-396 转发到 train_cli.main）。
_LAYA_BOOT = (
    "import sys;"
    "from laya.cli import main;"
    "sys.exit(main(['train'] + sys.argv[1:]))"
)

# `python -m laya` 不可用（laya 包没有 __main__.py），`laya train` 才是等价入口；但 venv 的
# Scripts\laya.exe 不在 PATH 里，subprocess 直接调会 FileNotFoundError。
# 方案：用 venv python 执行上面的引导代码，把 'train' 之后的 argv 原样转发给
# laya.cli.main —— 与 laya-train.exe 走的是同一个 main()（laya/cli.py:394-396）。
LAYA_TRAIN = [sys.executable, "-c", _LAYA_BOOT]
SPLITS = ("train", "calib", "test")

def to_laya_row(rec: dict) -> dict:
    """我们的一行 -> laya 的 `{state, questions, expected}` 行。

    两种输入 schema 都支持：
      * v0.1（#7 之前的冒烟数据）：`{state, labels:{...}}`，本函数负责换算。
      * v0.2（#7 的 `scripts/build_dataset.py`）：已经是 laya 形状
        `{state, questions, gold:{qid:{probabilities:...}}}`，只清 `[MASK]` 后直通。
        它的 `gold.urgency.probabilities` 的 key 已经是 0-based（build_dataset.py
        内做掉 SCORE_OFFSET），所以不能再减一次。
    """
    state = rec["state"].replace("[MASK]", " ")
    if "gold" in rec and isinstance(rec["gold"], dict):
        # v0.2：已带 questions + gold，laya.items_from_rows 直接吃
        # （laya/train.py:320-339）。原样透传，只做 MASK 清理。
        return {**rec, "state": state}
    labels = rec["labels"]
    expected = {
        "route_team": labels["route_team"],
        # laya 的 score 是 0-based level index；我们对外用 1..5。
        "urgency": int(labels["urgency"]) - SCORE_OFFSET,
        "needs_refund_or_escalation": bool(labels["needs_refund_or_escalation"]),
    }
    return {"state": state, "questions": QUESTIONS, "expected": expected}


def cmd_convert(args: argparse.Namespace) -> int:
    in_dir = Path(args.input_dir)
    stats = {}
    for split in SPLITS:
        src = in_dir / f"{args.prefix}_{split}.jsonl"
        if not src.exists():
            print(f"missing {src}", file=sys.stderr)
            return 2
        rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
        out_rows = [to_laya_row(r) for r in rows]
        stats[split] = len(out_rows)
        if args.write:
            dst = in_dir / f"{args.prefix}_{split}.laya.jsonl"
            with open(dst, "w", encoding="utf-8") as fh:
                for row in out_rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"wrote {dst} ({len(out_rows)} rows)")
        else:
            print(f"[dry] would write {in_dir / f'{args.prefix}_{split}.laya.jsonl'} ({len(out_rows)} rows)")
    print("split sizes:", stats)
    return 0


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [
        *LAYA_TRAIN,
        "--data", str(args.data),
        "--out", str(args.out),
        "--base", args.base,
        "--loss", args.loss,
        "--epochs", str(args.epochs),
        "--micro-batch", str(args.micro_batch),
        "--grad-accum", str(args.grad_accum),
        "--encoder-lr", str(args.encoder_lr),
        "--head-lr", str(args.head_lr),
        "--device", args.device,
        "--seed", str(args.seed),
    ]
    if args.eval_data:
        cmd += ["--eval-data", str(args.eval_data)]
    if args.freeze_encoder:
        cmd += ["--freeze-encoder"]
    if args.label_smoothing:
        cmd += ["--label-smoothing", str(args.label_smoothing)]
    if args.max_len:
        cmd += ["--max-len", str(args.max_len)]
    if args.head_max_len:
        cmd += ["--head-max-len", str(args.head_max_len)]
    if args.dry_run:
        cmd += ["--dry-run"]
    return cmd


def cmd_run(args: argparse.Namespace) -> int:
    data = Path(args.data)
    if args.max_samples:
        # 截断成小样本 smoke：只取前 N 行，写到同目录的 _smoke 文件。
        lines = data.read_text(encoding="utf-8").splitlines()
        keep = [line for line in lines[: args.max_samples] if line.strip()]
        smoke = data.with_name(data.stem + f"_smoke{args.max_samples}.jsonl")
        smoke.write_text("\n".join(keep) + "\n", encoding="utf-8")
        data = smoke
        print(f"smoke subset: {len(keep)} rows -> {smoke}")
        if args.eval_data:
            eval_path = Path(args.eval_data)
            eval_lines = [line for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            eval_keep = eval_lines[: min(len(eval_lines), max(200, args.max_samples // 5))]
            eval_smoke = eval_path.with_name(eval_path.stem + "_smoke.jsonl")
            eval_smoke.write_text("\n".join(eval_keep) + "\n", encoding="utf-8")
            args.eval_data = eval_smoke
            print(f"smoke eval subset: {len(eval_keep)} rows -> {eval_smoke}")

    cmd = build_cmd(args)
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd)
    write_train_config(args)
    return proc.returncode


def write_train_config(args: argparse.Namespace) -> None:
    """把本轮的全部超参与输入路径写成 `runs/<id>/train_config.yaml`。

    票 #12 要求可复现：只靠命令历史不够，参数和输入的指纹必须落在 checkpoint 旁边。
    laya 自己会把训练后的超参写进 `rl_agent_config.json`，但那是它视角的；
    这里记的是**我们**怎么调的（含数据文件 sha256），两者互补。
    """
    out = Path(args.out)
    if not out.exists():
        return
    try:
        import torch

        peak_mb = int(torch.cuda.max_memory_allocated() / 1024 / 1024) if torch.cuda.is_available() else 0
    except Exception:  # 记不到显存不该让训练整体失败
        peak_mb = -1

    def digest(path: str) -> str:
        p = Path(path)
        if not p.exists():
            return "missing"
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    lines = [
        "# 由 scripts/train_rlcd.py run 自动生成，勿手改",
        f"out: {out}",
        f"peak_cuda_alloc_mb: {peak_mb}",
        "",
    ]
    for flag in (
        "data", "eval_data", "base", "loss", "epochs", "micro_batch", "grad_accum",
        "encoder_lr", "head_lr", "label_smoothing", "max_len", "head_max_len",
        "device", "seed", "freeze_encoder", "dry_run", "max_samples",
    ):
        value = getattr(args, flag, None)
        lines.append(f"{flag}: {value}")
        if flag in ("data", "eval_data") and value:
            lines.append(f"{flag}_sha256_16: {digest(value)}")
    (out / "train_config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'train_config.yaml'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="三件套 -> laya 训练 jsonl")
    c.add_argument("--input-dir", required=True)
    c.add_argument("--prefix", default="v0.1")
    c.add_argument("--write", action="store_true")
    c.set_defaults(func=cmd_convert)

    r = sub.add_parser("run", help="调用 laya-train")
    r.add_argument("--data", required=True, help="laya 格式 train jsonl")
    r.add_argument("--eval-data", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--base", default="multilingual", help="laya 内置名或本地 checkpoint 目录")
    r.add_argument("--loss", default="rlcd", choices=["rlcd", "soft-ce"])
    r.add_argument("--epochs", type=int, default=4)
    r.add_argument("--micro-batch", type=int, default=8)
    r.add_argument("--grad-accum", type=int, default=8)
    r.add_argument("--encoder-lr", type=float, default=2.5e-5)
    r.add_argument("--head-lr", type=float, default=1e-4)
    r.add_argument("--label-smoothing", type=float, default=0.0)
    r.add_argument("--max-len", type=int, default=0)
    r.add_argument("--head-max-len", type=int, default=0)
    r.add_argument("--device", default="auto")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--freeze-encoder", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--max-samples", type=int, default=0, help=">0 时截断成冒烟子集")
    r.set_defaults(func=cmd_run)

    args = ap.parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
