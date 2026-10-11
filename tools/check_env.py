"""mai-jev 环境自检：venv 关键版本 + GPU/CUDA 状态 + 三原语问题定义格式校验。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  G:\\mai-jev-work\\venv\\Scripts\\python.exe G:\\GitHub\\mai-jev\\tools\\check_env.py

Windows 控制台注意：直接跑会把中文按 GBK 打烂（乱码）。本脚本把 stdout 切成
UTF-8，并在 WriteConsoleW 场景下也能正常显示。
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import sys

REQUIRED = [
    "torch",
    "torchvision",
    "transformers",
    "tokenizers",
    "safetensors",
    "datasets",
    "numpy",
    "pyarrow",
    "pandas",
    "scikit-learn",
    "scipy",
    "psutil",
    "laya",
    "onnx",
    "onnxruntime",
]

# laya 的 questions 参数必须是 dict（question id -> definition），不是 list；
# TypeError: questions must be a dict of question id -> definition, got list
# 而且问题定义里不写 "name" 字段，id 就是 dict 的 key。
QUESTIONS = {
    "route_team": {
        "type": "choice",
        "criteria": {
            "售后": "一般售后咨询",
            "退款处理": "要求退款",
            "技术支持": "产品故障排查",
            "账务": "发票与账单",
            "投诉升级": "要求投诉或升级",
        },
        "instructions": "这个工单应该交给哪个团队处理？",
    },
    "urgency": {
        "type": "score",
        "criteria": [
            "完全不紧急，可以慢慢处理",
            "不太紧急，本周内处理即可",
            "一般，需要尽快处理",
            "比较紧急，影响客户正常工作",
            "非常紧急，客户明确要求立即处理",
        ],
        "instructions": "这个工单有多紧急？1=不紧急，5=非常紧急",
    },
    "needs_refund_or_escalation": {
        "type": "noul",
        "instructions": "客户是否要求退款或升级处理？",
    },
}

STATE = "客户：上周买的耳机一边没声音，申请退款，语气很急，说再拖就要投诉。"


def main() -> None:
    print(f"python  : {sys.version.split()[0]} ({platform.python_implementation()})")
    print(f"exe     : {sys.executable}")
    for name in REQUIRED:
        try:
            print(f"{name:<14}: {md.version(name)}")
        except md.PackageNotFoundError:
            print(f"{name:<14}: MISSING")

    import psutil
    import torch

    print(f"\ncuda available : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device         : {torch.cuda.get_device_name(0)}")
        free, total = torch.cuda.mem_get_info()
        print(f"vram           : free {free / 2**30:.2f} GB / total {total / 2**30:.2f} GB")
    print(f"cpu cores      : {psutil.cpu_count(logical=False)} physical / {psutil.cpu_count()} logical")
    print(f"ram total      : {psutil.virtual_memory().total / 2**30:.1f} GB")

    import laya

    print(f"\nlaya           : {laya.__version__}")
    print(f"laya QTYPES    : {laya.QTYPES}")
    print(f"laya default models: {laya.DEFAULT_MODELS}")

    agent = laya.load("convaiinnovations/laya-multilingual")
    out = agent.predict(STATE, QUESTIONS)
    answers = out.get("answers", out)
    for name, ans in answers.items():
        print(f"  {name}: {ans}")
    print("\nOK: 三原语问题定义格式均通过 laya 校验器。")


if __name__ == "__main__":
    # Windows 默认 GBK 控制台：切 UTF-8，避免中文乱码与 UnicodeEncodeError。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
