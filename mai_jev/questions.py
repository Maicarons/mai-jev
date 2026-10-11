"""mai-jev：三原语问题的唯一定义。

所有脚本（build_dataset / train_rlcd / calibrate_and_eval / export_onnx / bench_target）
都从这里 import QUESTIONS，保证训练、标定、评测、推理用的标签集完全一致。

laya 0.4.3 的硬约束（踩过的坑，别再犯）：
  - `questions` 必须是 dict（question id -> definition），传 list 会
    TypeError: questions must be a dict of question id -> definition, got list
  - 定义里不写 "name" 字段，id 就是 dict 的 key。
  - `choice` 用 `criteria`（dict label->描述，或 label 列表）；写 `options` 会被拒。
  - `score` 的 `criteria` 是等级描述列表，**index 0 = 最低分**（不是 min/max）。
  - `noul` 不写 criteria（若写只允许 {'true','false'} 两个键）。
  - 中文必须显式 `model="multilingual"`，否则 Router 按语言猜，会走错检查点。
"""

from __future__ import annotations

# --- choice: route_team -------------------------------------------------
# 五个路由团队。顺序即 label index，calibrate_and_eval 的混淆矩阵按这个顺序画。
ROUTE_TEAMS: tuple[str, ...] = (
    "售后",
    "退款处理",
    "技术支持",
    "账务",
    "投诉升级",
)

# --- score: urgency -----------------------------------------------------
# index 0 = 最低分。urgency 的标签就是 "1".."5" 的字符串（laya 的 score 输出
# 是连续值，落在 [0, 4] 区间，1-based 显示为 score+1；详见 calibrate_and_eval）。
URGENCY_LEVELS: tuple[str, ...] = ("1", "2", "3", "4", "5")

# --- noul: needs_refund_or_escalation -----------------------------------
NOUL_LABELS: tuple[str, ...] = ("false", "true")

# laya 的 score 头返回的是 [0, n_levels-1] 区间上的期望值；我们对外报 1-based。
SCORE_OFFSET = 1

QUESTIONS: dict[str, dict] = {
    "route_team": {
        "type": "choice",
        "criteria": {
            "售后": "一般售后咨询：保修、维修、换货、使用问题",
            "退款处理": "要求退货、退款、赔付、价格保护",
            "技术支持": "产品故障排查：无法联网、报错、兼容性、安装调试",
            "账务": "发票、对账、付款、套餐变更、自动续费",
            "投诉升级": "明确要求投诉、升级、赔偿或威胁曝光/监管",
        },
        "instructions": "这条中文客服工单应该交给哪个团队处理？",
    },
    "urgency": {
        "type": "score",
        "criteria": [
            "完全不紧急：一般咨询，没有时间要求",
            "不太紧急：希望尽快，但可接受本周内处理",
            "一般：影响使用，需要尽快处理",
            "比较紧急：影响客户正常工作或经营，有明确期限",
            "非常紧急：客户明确要求立即处理，或已威胁投诉/曝光/监管",
        ],
        "instructions": "这个工单有多紧急？1=不紧急，5=非常紧急",
    },
    "needs_refund_or_escalation": {
        "type": "noul",
        "instructions": "客户是否明确要求退款、赔付或升级处理？",
    },
}

# 每个问题一个简短的英文 key，用于结果文件列名。
QUESTION_ORDER: tuple[str, ...] = ("route_team", "urgency", "needs_refund_or_escalation")


def question_types() -> dict[str, str]:
    """question id -> laya 问题类型。"""
    return {qid: spec["type"] for qid, spec in QUESTIONS.items()}


def label_space(qid: str) -> tuple[str, ...]:
    """question id -> 合法标签集合（score 返回等级字符串）。"""
    if qid == "route_team":
        return ROUTE_TEAMS
    if qid == "urgency":
        return URGENCY_LEVELS
    if qid == "needs_refund_or_escalation":
        return NOUL_LABELS
    raise KeyError(qid)
