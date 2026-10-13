#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后冒烟测试：验证 mai-jev 模型已真正发布到 Hugging Face / ModelScope
且可下载、可加载、可推理。

用法
----
    python scripts/verify_published_model.py --source hf
    python scripts/verify_published_model.py --source modelscope
    python scripts/verify_published_model.py --source hf --repo Maicarons/mai-jev-zh-support \
        --revision v0.1.0 --skip-onnx

设计约束
--------
* `huggingface_hub` / `modelscope` / `onnxruntime` / `laya` 全部**惰性 import**。
  缺包时打印明确的安装提示并以非零码退出——**绝不自动安装任何东西**。
下载缓存强制落本地盘（`--cache-root` 默认 $WORK）：系统盘空间紧张，
  一个 1GB 权重落 C 盘两次就会出事。
* 只断言**结构合法性**（文件齐全、体积匹配、非 NaN、概率在 [0,1]、
  choice 落在候选集内），不断言数值质量——质量口径在 docs/research 下的评测里。

退出码：0 全部通过；1 校验失败；2 缺少依赖/参数错误。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# transformers 探测 TF 时可能死锁，官方建议。
os.environ.setdefault("USE_TF", "0")

DEFAULT_REPO = os.environ.get("MAI_JEV_REPO", "Maicarons/mai-jev-zh-support")
DEFAULT_CACHE_ROOT = os.environ.get("MAI_JEV_CACHE_ROOT", r"$WORK")

# 上游/本项目导出的 F16 权重实测 614 MB。给 5% 容差。
EXPECTED_SAFETENSORS_MB = 614.0
SAFETY_TOLERANCE = 0.05

REQUIRED_FILES = (
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "rl_agent_config.json",
)
# 编码器配置：上游把 ModernBERT 配置放在 encoder/config.json；容忍根级 config.json。
ENCODER_CONFIG_ANY = ("encoder/config.json", "config.json")

SAMPLE_STATE = (
    "您好，3 月的账单被重复扣款两次，请今天之内退还多收的费用，否则我们将取消订阅。"
)

SAMPLE_QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "该工单应转交哪个团队处理？",
        "criteria": {
            "after_sales": "售后",
            "refund": "退款处理",
            "tech": "技术支持",
            "billing": "账务",
            "complaint": "投诉升级",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "该工单的紧急程度如何？",
        "criteria": ["1", "2", "3", "4", "5"],
    },
    "refund": {
        "type": "noul",
        "instructions": "客户是否明确要求退款？",
    },
}


def log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"[verify] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def need(module: str, pip_name: str | None = None, py: str | None = None):
    """惰性 import；缺失时给出可照抄的安装命令并退出。"""
    try:
        return __import__(module)
    except ImportError:
        target = pip_name or module
        exe = py or "python"
        die(
            f"缺少依赖 `{module}`。请先安装（脚本不会自动安装）：\n"
            f"    {exe} -m pip install -U {target}"
        )


def cache_dir_for(source: str, cache_root: Path) -> Path:
    sub = "hf-cache" if source == "hf" else "ms-cache"
    path = cache_root / sub
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(source: str, repo: str, revision: str | None, cache_dir: Path) -> Path:
    """把仓库快照拉到本地缓存，返回快照目录。"""
    if source == "hf":
        need("huggingface_hub")
        from huggingface_hub import snapshot_download

        log(f"source=hf  repo={repo}  revision={revision or 'main'}")
        log(f"downloading snapshot -> {cache_dir}")
        return Path(
            snapshot_download(
                repo_id=repo,
                revision=revision or None,
                cache_dir=str(cache_dir),
            )
        )

    need("modelscope")
    from modelscope.hub.snapshot_download import snapshot_download

    log(f"source=modelscope  repo={repo}  revision={revision or 'master'}")
    log(f"downloading snapshot -> {cache_dir}")
    return Path(
        snapshot_download(
            model_id=repo,
            revision=revision or None,
            cache_dir=str(cache_dir),
        )
    )


def check_files(snapshot: Path) -> dict:
    log("--- file inventory ---")
    missing = [f for f in REQUIRED_FILES if not (snapshot / f).is_file()]
    if missing:
        die(f"快照缺少必需文件: {missing}\n  快照目录: {snapshot}")
    if not any((snapshot / p).is_file() for p in ENCODER_CONFIG_ANY):
        die(f"快照缺少编码器配置（任一即可）: {list(ENCODER_CONFIG_ANY)}\n  快照目录: {snapshot}")

    total = 0
    for p in sorted(snapshot.rglob("*")):
        if p.is_file():
            total += p.stat().st_size
            log(f"  {p.stat().st_size / 1024 / 1024:9.1f} MB  {p.relative_to(snapshot)}")
    log(f"files OK, total {total / 1024 / 1024 / 1024:.2f} GB")

    st_mb = (snapshot / "model.safetensors").stat().st_size / 1024 / 1024
    lo = EXPECTED_SAFETENSORS_MB * (1 - SAFETY_TOLERANCE)
    hi = EXPECTED_SAFETENSORS_MB * (1 + SAFETY_TOLERANCE)
    if not (lo <= st_mb <= hi):
        die(
            f"model.safetensors 大小 {st_mb:.1f} MB 超出预期区间 "
            f"[{lo:.1f}, {hi:.1f}] MB（基线 {EXPECTED_SAFETENSORS_MB} MB，F16，170 张量）"
        )
    log(f"safetensors size {st_mb:.1f} MB (expected ~{EXPECTED_SAFETENSORS_MB} MB)")

    if total > 2.5 * 1024**3:
        log(f"WARNING: 仓库 {total / 1024**3:.2f} GB 偏大，考虑把 fp32 ONNX 挪到 git tag")

    return {"snapshot": snapshot, "total_bytes": total, "safetensors_mb": st_mb}


def check_tensors(snapshot: Path) -> None:
    """直接读 safetensors header，确认张量数与 dtype。"""
    import struct

    with open(snapshot / "model.safetensors", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    dtypes: dict[str, int] = {}
    for v in header.values():
        dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + 1
    log(f"safetensors header: {len(header)} tensors, dtypes={dtypes}")
    if len(header) < 100:
        die(f"张量数 {len(header)} 明显偏少，权重可能被截断或上传中断")


def run_predict(snapshot: Path) -> None:
    """用 laya.Agent（laya.load 即该模型的 from_pretrained 路径）跑三个原语。"""
    need("laya", "laya")
    import laya

    log("--- transformers/laya load ---")
    agent = laya.load(str(snapshot))
    log(f"loaded {type(agent).__name__} from {snapshot}")

    log("--- sample ---")
    log(f"state: {SAMPLE_STATE}")
    result = agent.predict(SAMPLE_STATE, SAMPLE_QUESTIONS, max_len=512)
    answers = result["answers"]

    import math

    for name, q in SAMPLE_QUESTIONS.items():
        if name not in answers:
            die(f"原语 `{name}` 没有返回结果")
        a = answers[name]
        probs = a.get("probabilities") or {}
        for k, p in probs.items():
            if p is None or isinstance(p, bool) or not isinstance(p, (int, float)) or math.isnan(p):
                die(f"原语 `{name}` 的概率 {k}={p} 非法 (NaN/None/非数值)")
            if not 0.0 <= float(p) <= 1.0:
                die(f"原语 `{name}` 的概率 {k}={p} 超出 [0,1]")

        if q["type"] == "choice":
            picked = a.get("choice")
            if not probs:
                die(f"原语 `{name}` 没有返回概率分布")
            if picked not in q["criteria"]:
                die(f"choice 返回 `{picked}`，不在候选集 {list(q['criteria'])} 内")
            top = max(probs.items(), key=lambda kv: kv[1])
            log(f"choice  {name:<12} -> {picked}   p={probs[picked]:.3f}  (top={top[0]})")
        elif q["type"] == "score":
            if not probs:
                die(f"原语 `{name}` 没有返回等级概率分布")
            s = a.get("score")
            if s is None or math.isnan(float(s)):
                die(f"score 返回 {s} 非法")
            log(f"score   {name:<12} -> {s:.4f}   levels={len(probs)}")
        elif q["type"] == "noul":
            # noul 返回标量概率（不是分布）
            v = a.get("noul")
            if v is None or isinstance(v, bool):
                die(f"noul 返回 {v} 非法")
            v = float(v)
            if math.isnan(v) or not 0.0 <= v <= 1.0:
                die(f"noul 返回 {v} 不是 [0,1] 内的有限概率")
            log(f"noul    {name:<12} -> {v:.3f}")

    log("transformers path: all three primitives answered validly.")


def check_onnx(snapshot: Path) -> None:
    """能建 onnxruntime session 就过；仓库没放 ONNX 时只提示。"""
    onnx_dir = snapshot / "onnx"
    if not onnx_dir.is_dir():
        log("no onnx/ directory published -- skipping ONNX check")
        return
    candidates = sorted(onnx_dir.glob("*.onnx"))
    if not candidates:
        log("onnx/ is empty -- skipping ONNX check")
        return

    need("onnxruntime")
    import onnxruntime as ort

    for f in candidates:
        mb = f.stat().st_size / 1024 / 1024
        sess = ort.InferenceSession(str(f), providers=["CPUExecutionProvider"])
        ins = [i.name for i in sess.get_inputs()]
        log(f"onnx {f.name}: {mb:.1f} MB, session created, inputs={ins}")


def main() -> int:
    ap = argparse.ArgumentParser(description="验证 mai-jev 已发布且可加载")
    ap.add_argument("--source", choices=["hf", "modelscope"], required=True)
    ap.add_argument("--repo", default=DEFAULT_REPO, help="仓库 id（HF owner/name 或 魔搭 owner/name）")
    ap.add_argument("--revision", default=None, help="版本；HF 默认 main，ModelScope 默认 master")
    ap.add_argument(
        "--cache-root",
        default=DEFAULT_CACHE_ROOT,
        help="缓存根目录，默认 %(default)s（不要指向系统盘）",
    )
    ap.add_argument("--skip-onnx", action="store_true", help="跳过 ONNX 检查")
    ap.add_argument("--skip-predict", action="store_true", help="只校验文件，不跑推理（无需 torch）")
    args = ap.parse_args()

    cache_dir = cache_dir_for(args.source, Path(args.cache_root))
    try:
        snapshot = download(args.source, args.repo, args.revision, cache_dir)
    except Exception as exc:  # 下载失败通常是 401/404/网络
        die(f"下载失败：{type(exc).__name__}: {exc}")

    try:
        check_files(snapshot)
        check_tensors(snapshot)
        if args.skip_predict:
            log("--skip-predict given, not running inference.")
        else:
            run_predict(snapshot)
        if args.skip_onnx:
            log("--skip-onnx given, not checking ONNX.")
        else:
            check_onnx(snapshot)
    except SystemExit:
        raise
    except Exception as exc:
        die(f"校验失败：{type(exc).__name__}: {exc}", code=1)

    log(f"OK: {args.source}:{args.repo} is published, loads and answers all three primitives.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())