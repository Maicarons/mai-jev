"""mai-jev 冒烟：加载 laya-multilingual，跑中文 + 英文各一条三原语预测，记录延迟与置信度。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  G:\\mai-jev-work\\venv\\Scripts\\python.exe G:\\GitHub\\mai-jev\\tools\\smoke_laya.py

已确认的 laya 0.4.3 API 约束（写在注释里，别再踩）：
  - `questions` 必须是 dict（question id -> definition），传 list 会
    TypeError: questions must be a dict of question id -> definition, got list
  - 问题定义里不写 "name" 字段，id 就是 dict 的 key。
  - `choice` 用 `criteria`（dict label->描述 或 label 列表）；写 `options` 会被拒。
  - `score` 的 `criteria` 是等级描述列表，index 0 = 最低分（不是 min/max）。
  - `noul` 不写 criteria（若写只允许 {'true','false'}）。
  - Router 默认按语言自动路由；中文工单要显式 `model='multilingual'`。
  - 出厂 multilingual 检查点自带温度超界，会被 laya 钳到 0.5，置信度视为未标定：
    laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
    using choice:11+=0.1006 -> 0.5
"""

from __future__ import annotations

import json
import sys
import time

import laya

QUESTIONS = {
    "route_team": {
        "type": "choice",
        "criteria": {
            "技术支持": "网络无法连接、设备故障、报错排查",
            "账务": "发票、对账、付款与套餐变更",
            "物流查询": "配送进度、签收、改地址",
            "退款处理": "退货、退款、赔付",
            "产品咨询": "购买前咨询、功能与兼容性",
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

STATES = {
    "zh": (
        "客户：我上周购买的路由器一直无法连接网络，重启也没用，已经三天了。"
        "我要申请退款，请尽快处理，急！"
    ),
    "en": (
        "Customer: The printer I bought last week keeps jamming on every other page. "
        "This is really frustrating, I want my money back."
    ),
}


def main() -> None:
    t0 = time.perf_counter()
    router = laya.Router(default="multilingual", preload=True)
    print(f"[load] {time.perf_counter() - t0:.1f}s loaded={router.loaded}")

    results = {}
    for lang, state in STATES.items():
        lat = []
        out = None
        for _ in range(3):
            t = time.perf_counter()
            out = router.predict(state, QUESTIONS, model="multilingual", lang=lang)
            lat.append((time.perf_counter() - t) * 1000)
        results[lang] = {"latency_ms": [round(x, 1) for x in lat], "output": out}
        print(f"[{lang}] latency_ms={[round(x, 1) for x in lat]}")
        print(json.dumps(out.get("answers", out), ensure_ascii=False, indent=2, default=str))

    print("\n--- Agent-level API (multilingual) ---")
    agent = laya.load("convaiinnovations/laya-multilingual", device=None)
    print("agent model_id:", agent.model_id)
    out = agent.predict(STATES["zh"], QUESTIONS)
    results["agent_api"] = out
    print(json.dumps(out.get("answers", out), ensure_ascii=False, indent=2, default=str))

    with open(r"G:\mai-jev-work\logs\smoke_laya.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print("wrote G:\\mai-jev-work\\logs\\smoke_laya.json")


if __name__ == "__main__":
    # Windows 控制台默认 GBK：切 UTF-8，避免中文乱码。
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
