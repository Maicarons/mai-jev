"""mai-jev 三原语训练数据集构建脚本（laya-train 输入格式）。

职责：把 #4 审计过的三个原始数据源（$WORK\\data\\raw\\）转成
laya `train` 直接吃的 jsonl 行，并产出统计报告。

    {"id": ..., "state": "中文工单正文", "split": "train|calib|test",
     "questions": {mai_jev.questions.QUESTIONS 全量},
     "gold": {"route_team": {"probabilities": {...}},
              "urgency": {"probabilities": {"0": ...}},
              "needs_refund_or_escalation": {"probabilities": {...}}},
     "provenance": {"source": ..., "license": ..., "proxy": ...}}

laya 0.4.3 硬约束（实测，scripts/train_rlcd.py 踩过的坑）：
  - choice 的 gold 键 = criteria 的键（5 个中文团队名）
  - score 的 gold 键 = **等级 index 字符串** "0".."4"（不是显示用的 "1".."5"）
  - noul 的 gold 键 = "false"/"true"
  - 缺 gold 的问题不会报错，只是不出 item（items_from_rows 的行为）

三个源与用法见 DATASETS.md；本文件只写「怎么转」。

用法（Windows 控制台务必 UTF-8）：
    $env:PYTHONIOENCODING = 'utf-8'
    python scripts\\build_dataset.py ^
        --raw $WORK\\data\\raw --out $WORK\\data\\adapted --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mai_jev.questions import (  # noqa: E402
    QUESTIONS,
    ROUTE_TEAMS,
    URGENCY_LEVELS,
)

SEED = 20261013

# 划分比例：train / calib / test。calib 只用于温度标定（#6），绝不进训练。
#
# calib 必须够大：RUNS.md 记录 smoke 轮 calib 只有 72 条时三个温度全被 laya 钳到
# 0.5（无法标定）。10% × 3.2k ≈ 320 条，每个原语 320 条，足够拟合温度。
# train 仍控制在 3k 以内，满足 map「先在 ≤3k 样本上 1 小时跑通管线」的约束。
TRAIN_FRAC = 0.80
CALIB_FRAC = 0.10
TEST_FRAC = 0.10

# 去重：精确重复按 state 的 sha1；近似重复按字符 3-gram Jaccard >= 0.85 判重。
NEAR_DUP_THRESHOLD = 0.85

# 状态长度预算：state + 问题指令拼接后不得超过 768 token。
STATE_TOKEN_BUDGET = 768

# 软标签强度：gold 用 0.9/0.025 而不是 one-hot，抗过拟合并给标定留信息量。
GOLD_STRONG = 0.9
GOLD_WEAK = 0.025

# --------------------------------------------------------------------------
# 源清单：只有 audit（#4）确认过 schema 的源才进来
# --------------------------------------------------------------------------
SOURCES = {
    "zh-tw": {
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/albertkingdom/custom-service-ticket-zh-tw",
    },
    "mri": {
        "license": "MIT",
        "url": "https://huggingface.co/datasets/MoriyaResearchInstitute/MRI-decision-data-100k",
    },
    "zh-bench": {
        "license": "CC-BY-4.0",
        "url": "https://huggingface.co/datasets/CodyQin/zh-decision-bench",
    },
}


# --------------------------------------------------------------------------
# 繁体 -> 简体 + 台湾国语 -> 大陆词汇（不引 OpenCC：AGENTS.md 要求先批准新依赖）
# --------------------------------------------------------------------------
TW2SP = {
    "臺灣": "台湾", "台灣": "台湾", "超商": "便利店", "攏": "都", "蝦皮": "虾皮",
    "購物": "购物", "帳號": "账号", "帳務": "账务", "客訴": "客诉", "投訴": "投诉",
    "帳單": "账单", "網路連線": "网络连接", "網路": "网络", "連不上": "连不上",
    "連線": "连线", "軟體": "软件", "硬體": "硬件", "資訊": "信息", "檔案": "文件",
    "郵局": "邮局", "快遞": "快递", "退換貨": "退换货", "換貨": "换货", "訂單": "订单",
    "包裹": "包裹", "帳戶": "账户", "餘額": "余额", "儲值": "储值", "點數": "点数",
    "序號": "序列号", "認證": "认证", "驗證碼": "验证码", "簡訊": "短信", "客服": "客服",
    "售後": "售后", "服務": "服务", "人員": "人员", "處理": "处理", "回覆": "回复",
    "回應": "回应", "盡快": "尽快", "儘快": "尽快", "馬上": "马上", "已經": "已经",
    "還沒": "还没", "沒有": "没有", "麼": "么", "個": "个", "這": "这", "們": "们",
    "問題": "问题", "詢問": "询问", "電": "电", "話": "话", "手機": "手机",
    "電腦": "电脑", "平板": "平板", "充電": "充电", "電池": "电池", "螢幕": "屏幕",
    "畫面": "画面", "操作": "操作", "功能": "功能", "品質": "品质", "瑕疵": "瑕疵",
    "壞掉": "坏掉", "故障": "故障", "退款": "退款", "退費": "退费", "賠付": "赔付",
    "賠償": "赔偿", "補償": "补偿", "發票": "发票", "統一編號": "纳税人识别号",
    "載具": "载体", "載明": "载明", "時效": "时效", "逾期": "逾期", "扣款": "扣款",
    "授權": "授权", "銀行": "银行", "信用卡": "信用卡", "行動支付": "移动支付",
    "轉帳": "转账", "電商": "电商", "賣場": "卖场", "商家": "商家", "商品": "商品",
    "出貨": "出货", "到貨": "到货", "簽收": "签收", "運費": "运费", "倉庫": "仓库",
    "門市": "门店", "到超商": "到便利店",
}


def tw2sp(text: str) -> str:
    """繁体台湾国语 -> 简体大陆词汇。按长度降序替换，避免短词吃掉长词。"""
    for src in sorted(TW2SP, key=len, reverse=True):
        text = text.replace(src, TW2SP[src])
    return text


# --------------------------------------------------------------------------
# zh-tw：6 类 -> 5 团队映射（docs/research/datasets-and-licenses.md 1.2 节）
# --------------------------------------------------------------------------
ZHTW_CATEGORY_TO_TEAM = {
    # 「退換貨申請」= 要求退貨/換貨，正对应 QUESTIONS 里「退款处理：要求退货、退款、
    # 赔付、价格保护」的定义；换货/维修语境由 needs_refund_or_escalation 区分。
    "退換貨申請": "退款处理",
    "帳務爭議": "账务",
    "帳號/系統問題": "技术支持",
    "客訴投訴": "投诉升级",
    "商品諮詢": "售后",
    "物流查詢": "售后",
}

# 情绪强度 -> 紧急度：1..5 同序（proxy，DATASETS.md 有说明）
ZHTW_EMOTION_TO_URGENCY = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}


# --------------------------------------------------------------------------
# MRI：英文选项键 -> 5 团队（1.1 节实测的选项集合）
# --------------------------------------------------------------------------
MRI_ROUTE_MAP = {
    "unknown": "售后",       # 信息不足 -> 兜底；gold 用软标签体现不确定性
    "account": "技术支持",
    "billing": "账务",
    "technical": "技术支持",
    "fulfillment": "售后",
    "security": "技术支持",
}


def flatten_state(state) -> str:
    """MRI 的 state 可能是自然语句或结构化 dict -> 统一成字符串。"""
    if isinstance(state, str):
        return state.strip()
    if isinstance(state, dict):
        parts = []
        for key, value in state.items():
            if isinstance(value, (list, tuple)):
                value = "，".join(str(v) for v in value)
            elif isinstance(value, dict):
                value = "，".join(f"{k}={v}" for k, v in value.items())
            parts.append(f"{key}：{value}")
        return "；".join(parts)
    if isinstance(state, (list, tuple)):
        return "；".join(flatten_state(item) for item in state)
    return str(state).strip()


def normalize_target(value) -> str | None:
    """任意真值表示 -> laya noul 的 "true"/"false"。"""
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


# --------------------------------------------------------------------------
# gold / row 构造
# --------------------------------------------------------------------------
def make_gold(route: str, urgency: str, noul: str) -> dict:
    """三原语的 teacher 概率。choice 键=团队名，score 键=等级 index 字符串。"""
    urg_index = int(urgency) - 1
    return {
        "route_team": {
            "probabilities": {
                team: (GOLD_STRONG if team == route else GOLD_WEAK)
                for team in ROUTE_TEAMS
            }
        },
        "urgency": {
            "probabilities": {
                str(i): (GOLD_STRONG if i == urg_index else GOLD_WEAK)
                for i in range(len(URGENCY_LEVELS))
            }
        },
        "needs_refund_or_escalation": {
            "probabilities": {"false": 0.1, "true": 0.9}
            if noul == "true"
            else {"false": 0.9, "true": 0.1}
        },
    }


def make_row(rid: str, state: str, route: str, urgency: str, noul: str,
             split: str, source: str, proxy: str = "") -> dict:
    return {
        "id": rid,
        "state": state,
        "split": split,
        "questions": QUESTIONS,
        "gold": make_gold(route, urgency, noul),
        "provenance": {
            "source": source,
            "license": SOURCES[source]["license"],
            "proxy": proxy,
        },
    }


# --------------------------------------------------------------------------
# 去重工具
# --------------------------------------------------------------------------
def state_key(state: str) -> str:
    return hashlib.sha1(state.strip().encode("utf-8")).hexdigest()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    text = "".join(text.split())
    if len(text) <= n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


# MRI 的工单路由 family 是模板化数据：去掉工单号后整个 family 只有 4~5 条不同
# 文本（300 行 -> 27 条）。带工单号的原文会让 near-dup 退化成「保留第一条」，
# 因此 MRI 侧先去工单号再判重，保留同一模板的全部变体（它们带不同的
# route/urgency 目标，是有效监督）。
ID_PATTERN = re.compile(r"[A-Z]{2,}-[A-Z0-9]+-\d+")


def normalize_for_dedup(state: str) -> str:
    return ID_PATTERN.sub("", state)


class Deduper:
    """精确去重 + 近似去重（字符 3-gram Jaccard）。"""

    def __init__(self, threshold: float = NEAR_DUP_THRESHOLD) -> None:
        self.threshold = threshold
        self._exact: set[str] = set()
        self._grams: list[tuple[str, set[str]]] = []

    def is_dup(self, state: str, normalized: str | None = None) -> bool:
        """精确 key 用原文，近似比对用规范化文本（MRI 的工单号已被抹掉）。"""
        key = state_key(state)
        if key in self._exact:
            return True
        grams = char_ngrams(normalize_for_dedup(state) if normalized is None else normalized)
        for _, other in self._grams:
            union = len(grams | other)
            if union and len(grams & other) / union >= self.threshold:
                return True
        self._exact.add(key)
        self._grams.append((key, grams))
        return False


# --------------------------------------------------------------------------
# 划分：按 id 的确定性划分（同 id 同 split，重跑可复现）
# --------------------------------------------------------------------------
def assign_split(rid: str, rng: random.Random) -> str:
    r = rng.random()
    if r < TRAIN_FRAC:
        return "train"
    if r < TRAIN_FRAC + CALIB_FRAC:
        return "calib"
    return "test"


# --------------------------------------------------------------------------
# 源适配器
# --------------------------------------------------------------------------
def adapt_zhtw(raw: Path, deduper: Deduper) -> tuple[list[dict], Counter]:
    """zh-tw：3148 行 -> 路由(类别) + 紧急度(情绪) + noul(needs_human)。

    needs_human 是「是否需要人工介入」，与「是否要求退款/升级」语义接近但不等价；
    作为 noul 的 proxy 使用，DATASETS.md 已注明该偏差。
    """
    rows: list[dict] = []
    rejected: Counter = Counter()
    rng = random.Random(SEED + 1)
    for split_src in ("train", "validation", "test"):
        path = raw / "zh-tw" / f"{split_src}.jsonl"
        if not path.exists():
            rejected[f"missing {path.name}"] += 1
            continue
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected[f"json:{exc.msg}"] += 1
                continue
            ticket = rec.get("ticket") or {}
            text = tw2sp(str(ticket.get("ticket_text", "")).strip())
            category = str(ticket.get("category", "")).strip()
            if not text or category not in ZHTW_CATEGORY_TO_TEAM:
                rejected[f"category={category!r}"] += 1
                continue
            route = ZHTW_CATEGORY_TO_TEAM[category]
            emotion = ticket.get("emotion_intensity")
            urgency = ZHTW_EMOTION_TO_URGENCY.get(int(emotion)) if emotion is not None else None
            needs_human = ticket.get("needs_human")
            noul = "true" if needs_human else "false"
            if urgency is None:
                rejected[f"emotion={emotion!r}"] += 1
                continue
            rid = f"zhtw-{split_src}-{idx:05d}"
            if deduper.is_dup(text):
                rejected["dedup"] += 1
                continue
            rows.append(
                make_row(
                    rid, text, route, urgency, noul,
                    assign_split(rid, rng), "zh-tw",
                    proxy="emotion_intensity->urgency, needs_human->noul",
                )
            )
    return rows, rejected


# 参与训练的 MRI family：只保留「中文客服工单路由」family。
#
# 排除的 family（docs/research/datasets-and-licenses.md 1.1 节已记录）：
#   - threshold-*   : state 是 INC/RISK + 数值指标（不可用分钟、风险分），不是工单正文，
#                     且 score 是「数值区间等级」，不是紧急度语义。
#   - dialogue-*    : state 是对话轮次/领域话题（客服、差旅、教育…），非工单。
#   - language-*    : 语言学任务（语义、语用、翻译），与业务路由无关。
#   - world-*/logic-*: 通用世界知识与逻辑推理，与业务路由无关。
#   - software-*    : state 是软件缺陷（work_item + 报告），criteria 是研发团队
#                     （frontend/backend/qa/…），不是我们的 5 个客服团队。
#   - 其它 insufficient-*: 只有 meta-noul（信息是否明确），无路由语义。
MRI_ROUTE_FAMILIES = (
    "ticket-routing-intent-v1",                 # 50 行，主诉 -> 团队
    "ticket-routing-language-mix-v1",           # 50 行，中英混排主诉
    "ticket-routing-negative-disambiguation-v1",  # 50 行，否定范围路由
    "ticket-routing-compound-priority-v1",      # 50 行，多请求按显式顺序
    "ticket-routing-severity-priority-v1",      # 50 行，安全标记优先
    "insufficient-choice-unknown-v1",           # 50 行，信息不足 -> unknown
)

# MRI 的 noul 全是「信息是否明确给出」这一 meta 语义，不是「客户是否要求退款/
# 升级」。因此这批行的 noul 统一置 false 并在 proxy 字段注明（DATASETS.md 已
# 说明该偏差会使 needs_refund_or_escalation 偏向 false）。
MRI_NOUL_DEFAULT = "false"

# MRI 的 urgency：路由 family 本身没有 score 目标。中性默认 3，并标注 proxy。
MRI_URGENCY_DEFAULT = "3"

# 参与 urgency 监督的 MRI score family：只取「影响单位数/不可用分钟/风险分」这类
# 真实严重度信号，它们的 0..3 等级与工单紧急度方向一致（区间越高越紧急）。
# dialogue-*-score / language-*-score 是对话复杂度，语义不符，排除。
MRI_SCORE_FAMILIES = (
    "threshold-sla-duration-v1",
    "threshold-risk-band-v1",
    "threshold-queue-load-v1",
    "threshold-temperature-v1",
)


def adapt_mri(raw: Path, deduper: Deduper) -> tuple[list[dict], Counter]:
    """MRI：只取中文客服工单路由 family，贡献 route_team 监督。

    这批行**只有 route 标签是可信的**：
    - urgency 无对应目标（路由 family 不带 score），置中性默认 3，proxy 已注明；
    - noul 是「信息是否明确给出」的 meta 语义，与「客户是否要求退款/升级」不同，
      统一置 false，proxy 已注明（详见 DATASETS.md 的偏差一节）。

    去重前先抹掉工单号（normalize_for_dedup），否则模板化文本会被近似去重吃掉。
    """
    rows: list[dict] = []
    rejected: Counter = Counter()
    rng = random.Random(SEED + 2)
    wanted = set(MRI_ROUTE_FAMILIES)
    for path in sorted((raw / "mri").glob("*.jsonl")):
        for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected[f"json:{exc.msg}"] += 1
                continue
            meta = rec.get("_meta") or {}
            family = str(meta.get("family", ""))
            if family not in wanted:
                continue
            state = flatten_state(rec.get("state"))
            if not state:
                rejected["missing state"] += 1
                continue
            questions = rec.get("questions") or {}
            targets = rec.get("targets") or {}
            route = None
            for qid, target in targets.items():
                spec = questions.get(qid) or {}
                if spec.get("type") == "choice" and qid == "route":
                    key = str(target)
                    route = MRI_ROUTE_MAP.get(key, key if key in ROUTE_TEAMS else None)
            if route not in ROUTE_TEAMS:
                rejected[f"mapping:{family}:{route}"] += 1
                continue
            rid = f"mri-{path.stem}-{idx:05d}"
            if deduper.is_dup(state, normalize_for_dedup(state)):
                rejected["dedup"] += 1
                continue
            rows.append(
                make_row(
                    rid, state, route, MRI_URGENCY_DEFAULT, MRI_NOUL_DEFAULT,
                    assign_split(rid, rng), "mri",
                    proxy=f"{family}; choice-key->team; urgency=neutral-default; "
                          f"noul=false (meta-noul 不映射)",
                )
            )
    return rows, rejected


def adapt_zhbench(raw: Path) -> tuple[list[dict], Counter]:
    """zh-bench：business_scenarios 的 ecommerce_cs 34 行，**只进 test**。

    标签空间本身不同（4 团队 / 3 级紧急度），按 1.1 节说明映射到我们的空间；
    这 34 行是唯一带人工 gold 的 held-out 集，用于 #6 的评测。
    """
    rows: list[dict] = []
    rejected: Counter = Counter()
    path = raw / "zh-decision-bench" / "business_scenarios.jsonl"
    if not path.exists():
        rejected[f"missing {path.name}"] += 1
        return rows, rejected
    ZH_ROUTE_MAP = {
        "物流配送": "售后",
        "退换售后": "售后",
        "支付问题": "账务",
        "账户安全": "技术支持",
    }
    ZH_URGENCY_MAP = {"不紧急": "1", "一般": "3", "紧急": "5"}
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            rejected[f"json:{exc.msg}"] += 1
            continue
        if rec.get("domain") != "ecommerce_cs":
            rejected[f"domain={rec.get('domain')!r}"] += 1
            continue
        state = flatten_state(rec.get("state") or rec.get("ticket") or rec.get("text"))
        gold = rec.get("gold") or rec.get("targets") or {}
        route = ZH_ROUTE_MAP.get(gold.get("route") or gold.get("route_team"))
        urgency = ZH_URGENCY_MAP.get(gold.get("urgency"))
        noul = normalize_target(gold.get("escalate"))
        if route not in ROUTE_TEAMS:
            rejected[f"route={gold.get('route')!r}"] += 1
            continue
        if urgency not in set(URGENCY_LEVELS):
            rejected[f"urgency={gold.get('urgency')!r}"] += 1
            continue
        if noul is None:
            rejected[f"escalate={gold.get('escalate')!r}"] += 1
            continue
        rows.append(
            make_row(
                f"zhbench-bs-{idx:05d}", state, route, urgency, noul,
                "test", "zh-bench",
                proxy="held-out eval only; 4-team/3-level -> 5-team/1..5",
            )
        )
    return rows, rejected


# --------------------------------------------------------------------------
# token 长度统计（用 laya-multilingual 的 tokenizer，本地缓存）
# --------------------------------------------------------------------------
TOKENIZER_SNAPSHOT = (
    r"$HF_HUB\models--convaiinnovations--laya-multilingual"
    r"\snapshots\1720e3e3357cfe1e281542e223f8273b0890ca34"
)


def token_lengths(states: list[str]) -> dict:
    """state + 问题指令的 token 长度分位数。tokenizer 缺失时降级为字符数。"""
    instructions = " ".join(
        spec.get("instructions", "") for spec in QUESTIONS.values()
    )
    try:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(
            str(Path(TOKENIZER_SNAPSHOT) / "tokenizer" / "tokenizer.json")
        )
        encode = lambda text: len(  # noqa: E731
            tok.encode(text, add_special_tokens=True).ids
        )
        mode = "laya-multilingual tokenizer"
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] tokenizer unavailable ({exc}); falling back to chars", file=sys.stderr)
        encode = lambda text: len(text)  # noqa: E731
        mode = "chars (tokenizer unavailable)"
    lengths = sorted(encode(state + " " + instructions) for state in states)
    if not lengths:
        return {"mode": mode, "n": 0}
    n = len(lengths)
    return {
        "mode": mode,
        "n": n,
        "p50": lengths[n // 2],
        "p90": lengths[int(n * 0.9)],
        "p99": lengths[min(n - 1, int(n * 0.99))],
        "max": lengths[-1],
        "over_budget": sum(1 for x in lengths if x > STATE_TOKEN_BUDGET),
        "budget": STATE_TOKEN_BUDGET,
    }


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="原始数据目录（含 mri/ zh-tw/ zh-decision-bench/）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--write", action="store_true", help="缺省只统计不落盘")
    args = ap.parse_args()

    raw = Path(args.raw)
    deduper = Deduper()

    all_rows: list[dict] = []
    per_source: dict[str, dict] = {}
    for name, fn in (
        ("zh-tw", lambda: adapt_zhtw(raw, deduper)),
        ("mri", lambda: adapt_mri(raw, deduper)),
        ("zh-bench", lambda: adapt_zhbench(raw)),
    ):
        rows, rejected = fn()
        all_rows.extend(rows)
        per_source[name] = {
            "rows": len(rows),
            "license": SOURCES[name]["license"],
            "rejected": dict(rejected),
        }

    route_counter: Counter = Counter()
    urg_counter: Counter = Counter()
    noul_counter: Counter = Counter()
    split_counter: Counter = Counter()
    source_counter: Counter = Counter()
    for row in all_rows:
        probs = row["gold"]["route_team"]["probabilities"]
        route_counter[max(probs, key=probs.get)] += 1
        uprobs = row["gold"]["urgency"]["probabilities"]
        urg_counter[str(int(max(uprobs, key=uprobs.get)) + 1)] += 1
        nprobs = row["gold"]["needs_refund_or_escalation"]["probabilities"]
        noul_counter[max(nprobs, key=nprobs.get)] += 1
        split_counter[row["split"]] += 1
        source_counter[row["provenance"]["source"]] += 1

    report = {
        "seed": SEED,
        "split_fractions": {"train": TRAIN_FRAC, "calib": CALIB_FRAC, "test": TEST_FRAC},
        "per_source": per_source,
        "total": len(all_rows),
        "splits": dict(split_counter),
        "route_team": dict(route_counter),
        "urgency": {k: urg_counter.get(k, 0) for k in URGENCY_LEVELS},
        "needs_refund_or_escalation": dict(noul_counter),
        "rows_by_source": dict(source_counter),
        "token_length": token_lengths([r["state"] for r in all_rows]),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.write:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for split in ("train", "calib", "test"):
            target = out / f"mai_jev_{split}.jsonl"
            n = 0
            with open(target, "w", encoding="utf-8") as fh:
                for row in all_rows:
                    if row["split"] == split:
                        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n += 1
            print(f"wrote {target} ({n} rows)")
        report_path = out / "build_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {report_path}")
    else:
        print("dry run: pass --write to emit files")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
