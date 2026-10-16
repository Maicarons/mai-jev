#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把已经组装好的发布目录推送到 Hugging Face / ModelScope（ticket #5，HITL）。

这个脚本**只做上传**：它不训练、不评测、不重新组装。上传什么由
`scripts/build_export.py` 决定，传完由 `scripts/verify_published_model.py` 验收。
三个阶段分开，是为了让「产物是什么」与「产物被传到哪」互不掩盖。

用法
----
    # 只看要传什么（不联网、不需要 token）
    python scripts/publish_model.py --source both --dry-run

    # 推到模型两地（token 从 --env-file 或进程环境读）
    python scripts/publish_model.py --source both --stage $WORK\\export\\publish

凭证（**绝不落盘、绝不回显**）
------------------------------
优先级：进程环境变量 > `--env-file`（默认 `<仓库根>/.env`，已在 .gitignore 里）。

    HF_TOKEN=hf_xxx            # HF **Write** 权限 token
    MODELSCOPE_API_TOKEN=ms_xxx # ModelScope 写权限访问令牌

`--env-file` 里的键名允许 `hf-key` / `ms-key`（历史写法）与 `HF_TOKEN` /
`MODELSCOPE_API_TOKEN` 两种拼法。**不要把 token 写进命令行** —— 命令行会进 shell 历史。

两个仓库名默认不同，这是实测事实不是笔误
----------------------------------------
HF 账号是 `Maicarons`，ModelScope 认证后的 owner 是 `Mai2026`（`whoami` 确认过，见 #5）。
所以默认为 `Maicarons/mai-jev-zh-support` 与 `Mai2026/mai-jev-zh-support`；
`--repo` 可覆盖，`--repo` 对 `--source both` 会同时作用于两地。

退出码：0 成功；1 上传失败；2 参数/依赖/凭证错误。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

DEFAULT_STAGE = os.environ.get("MAI_JEV_STAGE", r"$WORK\export\publish")
DEFAULT_HF_REPO = os.environ.get("MAI_JEV_HF_REPO", "Maicarons/mai-jev-zh-support")
DEFAULT_MS_REPO = os.environ.get("MAI_JEV_MS_REPO", "Mai2026/mai-jev-zh-support")
DEFAULT_MS_NAME = "mai-jev 中文客服工单决策模型"

# 缺任何一个都不算一个可用的发布目录（形状由 scripts/build_export.py 定义）。
REQUIRED = (
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "rl_agent_config.json",
    "questions.json",
    "calibration.json",
    "onnx/model.onnx",
    "onnx/calibration.json",
    "onnx/deploy_config.json",
    "onnx/MANIFEST.json",
)

# 上传时跳过的东西：`.ms_upload_cache` / `_upload_cache` 是 ModelScope 上传器留下的缓存，
# 不属于模型产物；它要是被推到 HF，读的人会以为那是一个模型文件（2026-10-17 已在两地见过，
# 见 STRAY_PATTERNS）。
IGNORE_PATTERNS = ["**/.ms_upload_cache", "**/_upload_cache/**",
                   "**/__pycache__/**", "**/*.pyc", ".git/**"]

# 被误传过、之后要从远端删掉的路径（不是模型产物）。
STRAY_PATTERNS = [".ms_upload_cache", ".ms_upload_cache/**", "_upload_cache", "_upload_cache/**"]

HF_TOKEN_KEYS = ("HF_TOKEN", "hf-key", "hf_token")
MS_TOKEN_KEYS = ("MODELSCOPE_API_TOKEN", "ms-key", "modelscope_api_token", "ms_token")


def log(msg: str) -> None:
    print(f"[publish] {msg}", flush=True)


def die(msg: str, code: int = 2) -> None:
    print(f"[publish] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def need(module: str, pip_name: str | None = None) -> None:
    try:
        __import__(module)
    except ImportError:
        die(
            f"缺少依赖 `{module}`。请先安装（脚本不会自动安装）：\n"
            f"    python -m pip install -U {pip_name or module}"
        )


def read_env_file(path: Path) -> dict[str, str]:
    """极简 KEY=VALUE 读取；不执行、不展开、不打印。"""
    if not path.is_file():
        return {}
    env: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def find_token(env: dict[str, str], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = os.environ.get(key) or env.get(key)
        if value:
            return value
    return None


def resolve_token(env: dict[str, str], keys: tuple[str, ...], source: str) -> str:
    value = find_token(env, keys)
    if value:
        return value
    die(
        f"没有 `{source}` 的写权限凭证。请二选一：\n"
        f"  ① 设进程环境变量 `{keys[0]}`；或\n"
        f"  ② 在 --env-file 里加一行（键名 {keys[0]} 或 {keys[-2]}）。\n"
        f"token 不会被打印，也不会被写进任何文件。"
    )
    raise AssertionError("unreachable")


def check_stage(stage: Path) -> None:
    if not stage.is_dir():
        die(f"staging 目录不存在: {stage}\n  先用 scripts/build_export.py 组装发布目录。")
    missing = [rel for rel in REQUIRED if not (stage / rel).is_file()]
    if missing:
        die(f"staging 目录不完整，缺: {missing}\n  目录: {stage}")
    total = sum(p.stat().st_size for p in stage.rglob("*") if p.is_file())
    log(f"staging OK: {stage}  ({total / 1024**3:.2f} GB)")
    for rel in REQUIRED:
        log(f"  {(stage / rel).stat().st_size / 1024 / 1024:9.1f} MB  {rel}")


def publish_hf(stage: Path, repo: str, token: str, dry_run: bool) -> None:
    if dry_run:
        log(f"[hf] dry-run: would create+upload {stage} -> {repo}")
        return
    need("huggingface_hub")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    # 先自证 token 有效：本地 .env 里放 ModelScope token 是踩过的坑（#5），
    # 那种情况下 create_repo 会 401，但报错淹没在 traceback 里，看不出是凭证问题。
    try:
        api.whoami()
    except Exception as exc:
        die(
            f"Hugging Face 凭证无效（{type(exc).__name__}: {exc}）。\n"
            f"  需要的是 huggingface.co 上对该账号有 **Write** 权限的 token（settings/tokens），\n"
            f"  ModelScope 的令牌在 HF 侧一律无效 —— 见 issue #5。"
        )
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    started = time.time()
    api.upload_folder(folder_path=str(stage), repo_id=repo, repo_type="model",
                      ignore_patterns=IGNORE_PATTERNS, delete_patterns=STRAY_PATTERNS)
    log(f"[hf] uploaded in {time.time() - started:.1f}s")
    log(f"[hf] url: https://huggingface.co/{repo}")


def publish_ms(stage: Path, repo: str, token: str, dry_run: bool) -> None:
    if dry_run:
        log(f"[ms] dry-run: would create+upload {stage} -> {repo}")
        return
    need("modelscope", "modelscope")
    from modelscope.hub.api import HubApi

    api = HubApi(token=token)
    try:
        api.create_model(model_id=repo, chinese_name=DEFAULT_MS_NAME)
        log("[ms] repo created")
    except Exception as exc:  # 已存在不是错误，这里刻意不吞其他异常的类型
        log(f"[ms] create_model note: {exc!r}")
    started = time.time()
    api.upload_folder(repo_id=repo, folder_path=str(stage), ignore_patterns=IGNORE_PATTERNS)
    log(f"[ms] uploaded in {time.time() - started:.1f}s")
    # ModelScope 的 upload_folder 没有 delete_patterns（只有 ignore_patterns），
    # 清掉误传过的非产物文件要单独调 delete_files；远端没有这些路径时会报错，所以吞掉。
    try:
        api.delete_files(repo_id=repo, repo_type="model", delete_patterns=STRAY_PATTERNS)
        log("[ms] stray cache files removed")
    except Exception as exc:
        log(f"[ms] delete_files note: {exc!r}")
    log(f"[ms] url: https://www.modelscope.cn/models/{repo}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="发布 mai-jev 到 HF / ModelScope（只上传，不组装）。")
    parser.add_argument("--source", choices=["hf", "modelscope", "both"], default="both")
    parser.add_argument("--stage", default=DEFAULT_STAGE, help="已组装好的发布目录")
    parser.add_argument("--repo", help="覆盖仓库 id（对 both 同时作用于两地）")
    parser.add_argument("--env-file", default=str(Path(__file__).resolve().parents[1] / ".env"))
    parser.add_argument("--dry-run", action="store_true", help="只校验 staging 与凭证是否存在，不上传")
    args = parser.parse_args(argv)

    stage = Path(args.stage)
    check_stage(stage)

    env = read_env_file(Path(args.env_file))
    if env:
        log(f"read {len(env)} key(s) from {args.env_file} (values not echoed)")
    else:
        log(f"no readable {args.env_file}; relying on process environment")

    if args.source in ("hf", "both"):
        repo = args.repo or DEFAULT_HF_REPO
        token = find_token(env, HF_TOKEN_KEYS)
        if token is None and not args.dry_run:
            token = resolve_token(env, HF_TOKEN_KEYS, "Hugging Face")
        if token is None:
            log(f"[hf] dry-run: 无凭证 —— 这块上传会在真正运行时失败（repo {repo}）")
        publish_hf(stage, repo, token or "", args.dry_run)
    if args.source in ("modelscope", "both"):
        repo = args.repo or DEFAULT_MS_REPO
        token = find_token(env, MS_TOKEN_KEYS)
        if token is None and not args.dry_run:
            token = resolve_token(env, MS_TOKEN_KEYS, "ModelScope")
        if token is None:
            log(f"[ms] dry-run: 无凭证 —— 这块上传会在真正运行时失败（repo {repo}）")
        publish_ms(stage, repo, token or "", args.dry_run)

    log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
