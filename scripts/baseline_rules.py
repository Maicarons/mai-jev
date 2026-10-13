"""mai-jev：规则基线。用关键词规则给三原语打标签，作为"必须超过的同规模基线"。

这是最朴素的可解释基线：不做任何学习，只按关键词与正则打分。它的作用有两个：
  1. 在真实数据集到位之前，先验证 build_dataset / 评测脚本的管线是通的。
  2. 作为 #6 验收里的"同规模基线"之一 —— 微调后的模型必须明显超过它，
     否则说明模型根本没学到东西。

规则（中文客服工单域，详见 docs/adr/0002-baseline-rules.md）：
  route_team：
    - 命中 投诉/曝光/监管/12315/消协/媒体/升级/赔偿 → 投诉升级
    - 命中 退款/退货/赔付/价格保护/退钱 → 退款处理
    - 命中 发票/账单/扣费/续费/付款/对账/套餐 → 账务
    - 命中 连不上/无法/报错/故障/坏了/没反应/安装/调试/兼容 → 技术支持
    - 命中 保修/维修/换货/多久/怎么用 → 售后
    - 否则 → 售后（兜底，取多数类）
  优先级从高到低，因为"要求退款并投诉"这类文本同时命中多个关键词，
  业务上投诉升级 > 退款处理 > 账务 > 技术支持 > 售后。
  urgency：按"时限压力 + 情绪强度 + 诉求强度"三项累加，映射到 1..5。
  noul：命中 退款/退货/退钱/赔付/赔偿/价格保护 → true，否则 false。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  python scripts\\baseline_rules.py ^
      --data $WORK\\data\\v0.1_test.jsonl --out $WORK\\runs\\baseline_rules.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mai_jev.questions import ROUTE_TEAMS, URGENCY_LEVELS  # noqa: E402

ROUTE_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("投诉升级", re.compile(r"投诉|曝光|监管|12315|消协|消费者协会|媒体|升级处理|赔偿|官司|起诉")),
    ("退款处理", re.compile(r"退款|退货|退钱|赔付|价格保护|退费|返还|不想要了")),
    ("账务", re.compile(r"发票|抬头|账单|扣费|续费|自动续|付款|对账|套餐|多收|乱扣|价格不对")),
    ("技术支持", re.compile(r"连不上|无法连接|没网|报错|故障|坏了|没反应|死机|安装|调试|兼容|连接网络|没声音|不开机|闪退|卡顿")),
    ("售后", re.compile(r"保修|维修|换货|多久|怎么用|说明书|保质期|售后")),
]

FALLBACK_ROUTE = "售后"

TIME_PRESSURE = re.compile(r"立即|马上|尽快|立刻|今天|当天|现在|急|催|限时|三天|两天|一周|已经.*天|等.*天|拖")
EMOTION = re.compile(r"生气|愤怒|无语|失望|垃圾|坑|骗|投诉|不能忍|受不了|烦|差劲|恶心|什么态度|太差")
DEMAND = re.compile(r"必须|一定|要求|否则|不然|再不|就要|立马|立刻|马上|坚决")


def predict_route(state: str) -> str:
    for team, pattern in ROUTE_RULES:
        if pattern.search(state):
            return team
    return FALLBACK_ROUTE


def predict_urgency(state: str) -> int:
    score = 1
    if TIME_PRESSURE.search(state):
        score += 2
    if EMOTION.search(state):
        score += 1
    if DEMAND.search(state):
        score += 1
    # 显式数字线索："已经三天了" / "等了一周"
    if re.search(r"[三四五]天|一周|半个月|一个月|很久", state):
        score += 1
    return max(1, min(5, score))


def predict_noul(state: str) -> bool:
    return bool(re.search(r"退款|退货|退钱|赔付|赔偿|价格保护|退费|返还", state))


def predict_all(state: str) -> dict:
    return {
        "route_team": predict_route(state),
        "urgency": predict_urgency(state),
        "needs_refund_or_escalation": predict_noul(state),
    }


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _labels_from_row(row: dict) -> dict:
    """v0.1 schema 的 gold 标签；v0.2 的 `gold` 结构由 eval_checkpoint 传入解析器。"""
    return row["labels"]


def evaluate(rows: list[dict], gold_labels=None) -> dict:
    """规则基线评测。

    `gold_labels` 可选：调用方（`scripts/eval_checkpoint.py`）传入自己的 schema 解析器，
    以便同一批行既能喂给 v0.2 的 `gold` 结构，也能喂给 v0.1 的 `labels` 结构。
    """
    if gold_labels is None:
        gold_labels = _labels_from_row

    route_total = route_hit = 0
    urg_total = urg_hit = urg_off_by_one = 0
    noul_total = noul_hit = 0
    confusions: dict[str, dict[str, int]] = {}
    for row in rows:
        gold = gold_labels(row)
        pred = predict_all(row["state"])

        route_total += 1
        if pred["route_team"] == gold["route_team"]:
            route_hit += 1
        confusions.setdefault(gold["route_team"], {}).setdefault(pred["route_team"], 0)
        confusions[gold["route_team"]][pred["route_team"]] += 1

        urg_total += 1
        if pred["urgency"] == gold["urgency"]:
            urg_hit += 1
        if abs(pred["urgency"] - gold["urgency"]) <= 1:
            urg_off_by_one += 1

        noul_total += 1
        if pred["needs_refund_or_escalation"] == gold["needs_refund_or_escalation"]:
            noul_hit += 1

    return {
        "n": len(rows),
        "route_team": {"accuracy": route_hit / max(1, route_total), "n": route_total},
        "urgency": {
            "accuracy": urg_hit / max(1, urg_total),
            "off_by_one": urg_off_by_one / max(1, urg_total),
            "n": urg_total,
        },
        "needs_refund_or_escalation": {"accuracy": noul_hit / max(1, noul_total), "n": noul_total},
        "route_confusion": confusions,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="build_dataset 产出的 jsonl")
    ap.add_argument("--out", help="可选：把指标写成 json")
    args = ap.parse_args()

    rows: list[dict] = []
    for path in args.data:
        rows.extend(load_jsonl(Path(path)))
    if not rows:
        print("no rows", file=sys.stderr)
        raise SystemExit(2)

    metrics = evaluate(rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    # 分布 sanity check
    print("\nlabel sanity: ROUTE_TEAMS =", ROUTE_TEAMS, " URGENCY =", URGENCY_LEVELS)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {out}")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
