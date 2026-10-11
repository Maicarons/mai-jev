"""mai-jev：冒烟训练管线的数据前置——生成、build_dataset、转 laya 格式。

先跑通这条链路，再谈全量：
  make_smoke_data.py -> build_dataset.py -> train_rlcd.py convert -> laya-train --dry-run

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  G:\\mai-jev-work\\venv\\Scripts\\python.exe G:\\GitHub\\mai-jev\\scripts\\prepare_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# venv 在仓库外的 G:\mai-jev-work\venv（G 盘空间大，放得下权重与 checkpoint）。
WORK = Path("G:/mai-jev-work")
PY = WORK / "venv" / "Scripts" / "python.exe"
RAW = WORK / "data" / "raw"
DATA = WORK / "data"


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    steps = [
        [PY, REPO / "scripts" / "make_smoke_data.py", "--n", "300",
         "--out", RAW / "smoke.jsonl", "--write"],
        [PY, REPO / "scripts" / "build_dataset.py",
         "--inputs", str(RAW / "smoke.jsonl"),
         "--out", str(DATA), "--prefix", "v0.1", "--write"],
        [PY, REPO / "scripts" / "train_rlcd.py", "convert",
         "--input-dir", str(DATA), "--prefix", "v0.1", "--write"],
    ]
    for cmd in steps:
        code = run(cmd)
        if code != 0:
            print(f"step failed with exit code {code}", file=sys.stderr)
            return code
    print("smoke data ready:", DATA)
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    raise SystemExit(main())
