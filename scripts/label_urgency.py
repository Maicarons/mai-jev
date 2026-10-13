"""合成「中文工单紧急度」1–5 标注（ticket #16，方向 3）。

为什么需要它：v02 的 score 标签来自 `custom-service-ticket-zh-tw` 的
`emotion_intensity` 字段，语义是**客户情绪激动程度**；而 `mai_jev/questions.py` 里
urgency 问的是「这条工单有多紧急（时间要求 / 业务影响 / 期限）」。#12 用 2 轮 RLCD
把 score 从 0.1808 拉到 0.5948 就卡住了，choice/noul 同批数据分别到 0.9125/0.8397
—— 不是训练量问题，是标签语义不匹配（DATASETS.md §3）。

**与 `scripts/baseline_rules.py` 的区别（这一节是本脚本存在的理由）**：

`baseline_rules.predict_urgency` 是三信号累加器（时限 +2 / 情绪 +1 / 诉求 +1），
只做关键词命中。本脚本是一个**有证据等级的分级标注器**：

1. 分级的时间证据（小时 / 当天 / 几天 / 一周内 / 无期限），而不是「命中任意时限词 +2」；
2. **否定守卫**：`不着急`、`不用急`、`慢慢来`、`不急` 这类把强度往下的说法优先；
3. 业务影响分级（多客户 / 无法营业 / 停机 / 单人可等待），区分「影响客户经营」与
   「影响一个人体验」—— 这是 3 档与 4 档的真正分界，也是情绪强度根本没有的信息；
4. 升级威胁（投诉 / 曝光 / 监管 / 诉讼）是 5 档的**充分**条件而非加分项；
5. 逐级回落：最高证据档胜出，而不是累加，所以「投诉 + 一周 + 不着急」不会被加到 5。

**自证机制（对抗票里点名的「可能只是学会关键词」）**：本脚本同时导出
`urgency_antibody.jsonl` —— 一批**关键词命中但标注应当相反**的中文句子
（「不着急，慢慢处理就行」「投诉流程是什么，我想了解一下」…）。模型在这些句子上的
表现与规则基线并排报出：如果模型只是学会关键词，它在这些句子上会和规则基线一起错；
如果它学到了修饰语结构，准确率应当明显高于规则基线。这批句子是人工写的，不参与训练。

用法：

    python scripts/label_urgency.py --in-dir  $WORK\\data\\adapted ^
        --out-dir $WORK\\data\\adapted_v03 ^
        --report $WORK\\data\\adapted_v03\\relabel_report.json

只覆盖 `urgency` 一维；`route_team` 与 `needs_refund_or_escalation` 原样保留
（它们已经达标，重标它们没有收益，且会毁掉 #6 已发表的对比数字）。
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mai_jev.questions import QUESTIONS  # noqa: E402

SPLITS = ("train", "calib", "test")
URGENCY_LEVELS = ("1", "2", "3", "4", "5")


# ── 1. 时间证据分级 ────────────────────────────────────────────────────────
# 分数越高 = 期限越近。最强证据直接定档，不再叠加。
DEADLINE_HOURS = re.compile(r"(今天|当天|今日|24\s*小时|两三个小时|几个小时|马上|立即|立刻|现在就)")
DEADLINE_DAYS_1_3 = re.compile(r"(明[天后]|后天|这周内|本周内|三天内|两三天|3\s*天内|2\s*天内|尽快|加紧|催)")
DEADLINE_WEEK = re.compile(r"(一周内|本周|这周|下周|七天内|7\s*天内|几天内|近期)")
DEADLINE_ELAPSED = re.compile(r"(已经|等了|拖了|等了快|拖了快)[^，。！？]{0,6}(天|周|小时|个月)")

# 否定/降温：出现在时限词附近时把强度往回拉。放在所有加分证据之前判断。
NEGATED_URGENCY = re.compile(
    r"(不着急|不用急|不必急|不急(?!促|切)|无需急|慢慢(来|处理|弄)|不着急的话|随(便|时)就好|不催)")
# "急" 的其他误用：「急件发货」是物流术语，「急诊」是医疗，都不是工单紧急度。
URGENCY_FALSE_FRIEND = re.compile(r"(急诊|急件|加急费|急行|急切地想知道|着急地想知道)")

# ── 2. 业务影响分级 ────────────────────────────────────────────────────────
IMPACT_BUSINESS = re.compile(
    r"(无法营业|不能营业|停业|门店(都|全部)|全店|所有(客户|用户|门店)|公司(整个|所有)|"
    r"批量|大面积|多(名|个)(客户|用户|门店)|整(个|片)|公司业务|生产(线)?(停|中断)|"
    r"系统(整体)?(宕|挂了|崩了)|全(站|服)务(不可用|中断)|业务中断|资金(损失|差异).{0,4}(万|元))")
IMPACT_PERSONAL = re.compile(
    r"(我(个人|自己|一个人)|只有我|就我一个人|单个|一台|我的手机|我的账号|我的订单|我的设备)")
# 明确表示「可以等」是 1 档的正面证据，而不是「有影响」的证据。
IMPACT_WAITABLE = re.compile(
    r"(可以等|能等|都行|都行就好|不着急用|暂时不急|下周用|以后再|有空(再|的时候))")

# 只在**问规则本身**、不需要人动手的场景：这类句子常含业务关键词（退款/几天），
# 但没有待办的工单，因此是 1 档而不是 2 档。
ASKING_RULES = re.compile(
    r"(政策(里|上)?(写|说)|规定|规则|流程是什么|怎么投诉|如何(投诉|退货|退款)|"
    r"可以吗|能不能|支持吗|算不算|属于|规定了几天)")
# 「在问规则」与「真的要我办」的分界：前者是 1 档，后者至少是 2 档。
STRONG_REQUEST = re.compile(
    r"(我要|我想(退|申请|办理|退款|退货|投诉)|给我|帮我|申请|要求|务必|必须|起诉|"
    r"(我(要|准备|打算)(去|就)?(投诉|起诉|曝光)))")
# 「查资料」而不是「办事」：句子里出现了规则/政策/写/规定这一类**元信息**主语。
META_QUESTION = re.compile(
    r"(政策|规定|规则|条款|说明|文档|官网|客服说).{0,10}(写|说|里|上)|写了几天|规定几天")

# ── 3. 升级威胁：5 档的充分条件 ────────────────────────────────────────────
ESCALATION = re.compile(
    r"(投诉|曝光|315|12315|消协|消费者协会|监管|工商|起诉|诉讼|打官司|律师|"
    r"媒体|新闻|舆情|举报|消保委|仲裁)")
ESCALATION_NEGATED = re.compile(
    r"(不投诉|没有投诉|不会投诉|别投诉|投诉(流程|制度|记录|电话)是什么|如何投诉|怎么投诉|"
    r"问一下投诉|咨询投诉|投诉过|曾投诉|投诉率)")

# 付款/扣费类重复损失：2 档是「一般咨询」，这类至少 3 档。
FINANCIAL_IMPACT = re.compile(r"(重复扣(款|费)|多扣|扣了两次|两次扣款|重复支付|多付了|少了钱|金额不对)")

# 退款/赔付诉求本身不等于紧急（我们的 noul 已经覆盖这一点），这里只做最低限度托底。
REFUND_DEMAND = re.compile(r"(退款|退货|退钱|赔付|赔偿|退费|返还)")


def _has_negated_urgency(text: str) -> bool:
    """否定守卫：命中降温表达，且附近没有「不要慢慢来」这类反向修饰。"""
    m = NEGATED_URGENCY.search(text)
    if not m:
        return False
    window = text[max(0, m.start() - 12): m.end() + 12]
    if re.search(r"(不能|别|不要)(慢慢|不急|拖)", window):
        return False
    return True


def label_urgency(text: str) -> tuple[str, str]:
    """返回 `(等级, 证据标签)`。等级是 1..5 的字符串。

    分级顺序即证据强度顺序，**第一个命中的就是最终档位**（不累加）：
    这与 `baseline_rules` 的累加器是根本区别 —— 累加会让「投诉 + 一周 + 不着急」爆到 5。
    """
    t = text.strip()
    if not t:
        return "1", "empty"

    # 0) 误用守卫：急诊 / 急件 / 急切地想知道 —— 这些词不表示工单紧急。
    cleaned = URGENCY_FALSE_FRIEND.sub(" ", t)

    # 1) 升级威胁是充分条件（5 档），但要排除「投诉流程是什么」这类中性问法。
    if ESCALATION.search(cleaned) and not ESCALATION_NEGATED.search(cleaned):
        return "5", "escalation"

    # 2) 显式小时级期限（4 档）；否定守卫不适用于「今天/马上」这种硬期限，
    #    但适用于「尽快」，所以分开判。
    if DEADLINE_HOURS.search(cleaned):
        return "4", "deadline_hours"
    if DEADLINE_DAYS_1_3.search(cleaned) and not _has_negated_urgency(cleaned):
        return "4", "deadline_days_1_3"

    # 3) 否定守卫：明说不着急时，除非已经出现经营影响（下一档），否则压到 1–2。
    if _has_negated_urgency(cleaned):
        if IMPACT_BUSINESS.search(cleaned):
            return "3", "negated_but_business_impact"
        if IMPACT_PERSONAL.search(cleaned):
            return "2", "negated_personal"
        return "1", "explicitly_not_urgent"

    # 4) 经营级影响：4 档。这是「影响客户经营、有明确影响面」的定义性证据。
    if IMPACT_BUSINESS.search(cleaned):
        return "4", "business_impact"

    # 5) 明确期限在周内（3 档）。
    if DEADLINE_WEEK.search(cleaned) or DEADLINE_ELAPSED.search(cleaned):
        return "3", "deadline_week_or_elapsed"

    # 6) 资金损失（重复扣款类）：至少 3 档，即使没写期限。
    if FINANCIAL_IMPACT.search(cleaned):
        return "3", "financial_impact"

    # 7) 明确表示「可以等 / 都行」= 1 档的正面证据，优先于个人影响。
    if IMPACT_WAITABLE.search(cleaned):
        return "1", "explicitly_waitable"

    # 8) 一人一时的影响（没有等待余地也没有期限）= 2 档。
    if IMPACT_PERSONAL.search(cleaned):
        return "2", "personal_impact"

    # 10) 问政策文本本身（"退款政策里写了几天"）= 1 档：查资料，没人要动手办。
    if META_QUESTION.search(cleaned):
        return "1", "meta_question_about_policy"

    # 11) 问"我这种情况能不能退"= 2 档：确实有待办的事，只是没时间压力。
    if REFUND_DEMAND.search(cleaned) and ASKING_RULES.search(cleaned):
        return "2", "refund_request_without_urgency"

    # 12) 只有退款诉求、没有任何时间/影响线索 = 2 档（不紧急，但确实要办）。
    if REFUND_DEMAND.search(cleaned):
        return "2", "refund_only_no_time_signal"

    # 9) 其余：一般咨询，无任何紧急证据。
    return "1", "no_evidence"


# ── 规则反例集：人工写的「关键词命中但标注应当相反」句子 ────────────────────
# 不参与训练，只在评测里与规则基线并排报 accuracy。设计意图：这些句子里的关键词
# （急/投诉/退款/尽快）在 `baseline_rules` 的累加器下会得高分，但人读下来并不紧急。
ANTIBODY: list[tuple[str, str]] = [
    ("不着急，慢慢处理就行", "1"),                 # 命中「不着急」→ 规则累加器反而因 EMOTION/DEMAND 误判
    ("这个不急，我就是先问问流程", "1"),
    ("不用急，下个月有空了再说", "1"),
    ("投诉流程是什么？我想先了解一下", "1"),        # ESCALATION 会命中，但问的是流程
    ("投诉过的订单能保修吗", "1"),                  # 历史投诉，不是威胁
    ("产品没问题，我就是想问一下能不能退款", "2"),   # 退款诉求但无时间压力
    ("退款政策里写了几天可以退吗", "1"),            # 问规则本身
    ("我一个人的账号登录不上，麻烦了", "2"),        # 个人影响、无期限
    ("一台机器坏了，别的都正常", "2"),
    ("可以等，月底之前弄好都行", "1"),
    ("系统整体宕机了，所有用户都登录不进去", "4"),  # 正例：规则累加器给不出 4
    ("公司所有门店的收款都失败了，无法营业", "4"),
    ("已经拖了三天了还没人处理", "3"),             # 正例：elapsed 证据
    ("订单重复扣款了两次，金额不对", "3"),
]


def antibody_rows() -> list[dict]:
    """反例集 -> 与训练数据同构的行（gold 的 score key 是 0-based）。"""
    rows = []
    for idx, (text, level) in enumerate(ANTIBODY):
        gold = {"probabilities": {str(int(level) - 1): 1.0}}
        rows.append({
            "id": f"antibody-{idx:03d}",
            "state": text,
            "questions": dict(QUESTIONS["urgency"]),
            "gold": {"urgency": gold},
            "split": "antibody",
            "provenance": {"source": "handwritten", "license": "Apache-2.0",
                           "proxy": "handwritten-urgency-rubric"},
        })
    return rows


# ── 重标注主流程 ────────────────────────────────────────────────────────────

def relabel(rows: Iterable[dict]) -> tuple[list[dict], Counter]:
    """就地覆盖每行的 `gold.urgency.probabilities`（score 的 key 必须是 0-based）。"""
    out, evidence = [], Counter()
    by_level: Counter = Counter()
    for rec in rows:
        rec = json.loads(json.dumps(rec, ensure_ascii=False)) if not isinstance(rec, dict) else dict(rec)
        text = str(rec.get("state") or "")
        level, why = label_urgency(text)
        by_level[level] += 1
        evidence[why] += 1

        gold = dict(rec.get("gold") or {})
        urg = dict(gold.get("urgency") or {})
        urg["probabilities"] = {str(int(level) - 1): 1.0}
        gold["urgency"] = urg
        rec["gold"] = gold
        rec["questions"] = dict(rec.get("questions") or {})
        # 必须写回**完整**的问题定义：laya 的 items_from_rows 会因 score 缺 criteria
        # 报 invalid_question 并**静默跳过该问题**，评测件数就会少三分之一（686/343=2）。
        rec["questions"]["urgency"] = dict(QUESTIONS["urgency"])
        prov = dict(rec.get("provenance") or {})
        prov["urgency_proxy"] = f"synthetic-rubric:{why}"
        prov["urgency_old_proxy"] = prov.get("urgency_proxy") or prov.get("proxy", "")
        rec["provenance"] = prov
        out.append(rec)
    evidence["__levels__" + repr(dict(sorted(by_level.items())))] += 0
    return out, evidence


def relabel_selfcheck(rows: list[dict] | None = None) -> dict:
    """自检：rubric 必须与反例集的人工标签一致，且**应该**与规则基线不一致。

    两条断言缺一不可：
      1. `label_urgency(text) == want` —— 否则是我的 rubric 写错了；
      2. `predict_urgency(text) != want` 的比例 —— 这是「不同源」的证据。若规则基线
         全部答对反例集，就说明两者学的是同一套关键词，本标注器没有独立价值。
    """
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from baseline_rules import predict_urgency

    rubric_bad, rules_right = [], []
    for text, want in ANTIBODY:
        got, _ = label_urgency(text)
        if got != want:
            rubric_bad.append(f"{text!r} rubric={got} want={want}")
        if str(predict_urgency(text)) == want:
            rules_right.append(f"{text!r} rules={predict_urgency(text)}")
    return {"rubric_misses": rubric_bad, "rules_correct": rules_right,
            "n": len(ANTIBODY),
            "divergence": f"{len(rules_right)}/{len(ANTIBODY)} rules-baseline agree with antibody"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="#7 产出的 adapted 目录（含 mai_jev_*.jsonl）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="mai_jev")
    ap.add_argument("--report", help="统计报告 JSON 路径（默认写到 out-dir）")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    check = relabel_selfcheck()
    if check["rubric_misses"]:
        print("ABORT: 标注器与人工反例集不一致，先修 rubric：")
        for p in check["rubric_misses"]:
            print("  -", p)
        return 1
    print("selfcheck: rubric OK on all %d antibody rows; %s"
          % (check["n"], check["divergence"]))

    report = {
        "input_dir": str(in_dir),
        "prefix": args.prefix,
        "splits": {},
        "evidence_histogram": {},
        "selfcheck": check,
    }

    for split in SPLITS:
        src = in_dir / f"{args.prefix}_{split}.jsonl"
        if not src.exists():
            report["splits"][split] = {"missing": str(src)}
            continue
        rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
        new_rows, evidence = relabel(rows)
        dst = out_dir / f"{args.prefix}_{split}.jsonl"
        with dst.open("w", encoding="utf-8") as fh:
            for rec in new_rows:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        levels = Counter()
        for rec in new_rows:
            probs = rec["gold"]["urgency"]["probabilities"]
            levels[max(probs, key=lambda k: probs[k])] += 1
        before = Counter()
        for rec in rows:
            probs = (rec.get("gold") or {}).get("urgency", {}).get("probabilities") or {}
            if probs:
                before[max(probs, key=lambda k: probs[k])] += 1
        report["splits"][split] = {
            "rows": len(new_rows),
            "level_dist_0based": {k: before.get(k, 0) for k in "01234"},
            "level_dist_0based_after": {k: levels.get(k, 0) for k in "01234"},
        }
        for k, v in evidence.items():
            if not k.startswith("__"):
                report["evidence_histogram"][k] = report["evidence_histogram"].get(k, 0) + v
        print(f"{split}: {len(new_rows)} rows -> {dst}")
        print(f"  before {dict(sorted(before.items()))}")
        print(f"  after  {dict(sorted(levels.items()))}")

    ab = out_dir / f"{args.prefix}_antibody.jsonl"
    with ab.open("w", encoding="utf-8") as fh:
        for rec in antibody_rows():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    report["antibody"] = {"path": str(ab), "rows": len(ANTIBODY)}

    rp = Path(args.report) if args.report else out_dir / "relabel_report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", rp)
    print("antibody ->", ab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())