"""Laya 基线冒烟：加载 laya-multilingual，跑中文 + 英文各一条三原语预测，记录延迟与置信度。

用法：
  python tools\smoke_laya.py
"""

from __future__ import annotations

import json
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
        for i in range(3):
            t = time.perf_counter()
            out = router.predict(state, QUESTIONS, model="multilingual", lang=lang)
            lat.append((time.perf_counter() - t) * 1000)
        results[lang] = {"latency_ms": [round(x, 1) for x in lat], "output": out}
        print(f"[{lang}] latency_ms={[round(x,1) for x in lat]}")
        print(json.dumps(out.get("answers", out), ensure_ascii=False, indent=2, default=str))

    print("\n--- Agent-level API (multilingual) ---")
    agent = laya.load("convaiinnovations/laya-multilingual", device=None)
    print("agent model_id:", agent.model_id)
    out = agent.predict(STATES["zh"], QUESTIONS)
    results["agent_api"] = out
    print(json.dumps(out.get("answers", out), ensure_ascii=False, indent=2, default=str))

    with open(r"$WORK\logs\smoke_laya.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2, default=str)
    print("wrote $WORK\\logs\\smoke_laya.json")


if __name__ == "__main__":
    main()
