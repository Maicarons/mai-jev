"""合成「客户是否要求退款 / 赔付 / 升级」标注（ticket #17）。

为什么需要它：v0.2/v0.3 的 noul 标签来自 `custom-service-ticket-zh-tw` 的
`needs_human`（**是否需要人工介入**）与 zh-decision-bench 的 `gold.escalate`（同一问法），
而 `mai_jev/questions.py` 里 noul 问的是「客户是否**要求**退款、赔付或升级处理」。
两者语义不同：前者描述**运营动作**（这条要不要人接手），后者描述**客户诉求**。

代价是实测出来的（2026-10-17，v0.3 适配数据）：三个 split 的 noul 正例率都是 ~64%，
而正例里 **70% 连退款/赔付/升级的字眼都没有** —— 典型如「我訂的帽子卡在海關，急死我了，
請快幫我查清楚！」。它们共同的特征是「客户着急、要有人处理」，正是 `needs_human` 的定义。
⇒ 现在的 noul 头基本是**第二个 urgency 头**，0.8513 衡量的更像「客户是否在着急」。

**口径（2026-10-17 用户拍板，与本文件的分级顺序一一对应）**：

1. **明确提出诉求** → true：`我要退款`、`请把钱退回我的卡里`、`麻烦快点理賠`、`要求赔偿`；
2. **升级要求或威胁** → true：`我要找你们主管`、`再拖我就去消保协会投诉`、`发到网上曝光`。
   但**问投诉本身**（`投诉流程是什么`、`如何投诉`）不算 —— 那是问规则，不是威胁；
3. **资格 / 意向询问** → true（**宽松口径**：`我能不能退款？`、`可不可以退货？` 视为有退款意向）；
   这一类在残差探针（2026-10-17）后又扩了三形：`是不是能退货`、`有没有什么补偿方案`、
   `是要理賠還是補寄呢`（择一诉求）、`不然真的是要退货了`（条件式意愿）；
4. **纯问流程 / 规则** → false：`退貨流程怎么走`、`退貨運費是我出还是你们负担`、`贈品怎么处理`；
5. **只有情绪或催促** → false：`急死我了`、`快帮我查清楚`、`给我个说法`（没有任何退款/升级线索）。

第 4 与第 3 分界是本票最容易出错的地方：「问能不能退」（要结果）判 true，「问怎么退 / 谁出钱」
（要知识）判 false。分级顺序即证据强度顺序，**第一个命中的就是最终判定**，不累加。

宽松口径的**唯一例外**：客户明说「只是随便问问 / 好奇 / 东西没问题」时判 false ——
那不是「问资格」，而是「没有诉求」（`CURIOSITY` 与反例集里的对应行）。

**与 `scripts/baseline_rules.py` 的区别（本脚本存在的理由）**：

`baseline_rules.predict_noul` 是单条关键词表（命中 退款/退货/退钱/赔付/赔偿/价格保护 → true）。
它必然在两类句子上出错，而这两类恰好是语料里最多的：

- `不是要退款，我只是想了解退货流程` —— 关键词命中，实际是问流程；
- `再不解决我就找消费者保护协会` —— 没有退款词，实际是明确要求升级。

本脚本的守卫（**否定剥离**、**条件句豁免**、**中性问法排除**、**了解-守卫**、**好奇-豁免**、
**维权热线的字母数字边界**）就是为这些形态写的。其中「维权热线边界」与「了解-守卫」是
**眼验真实数据后**抓到的两个真 bug（`订单號是123456789` 里的 `12345`、`我要马上查明跟赔偿`）：
两者都已固化成反例集里的行，不会再退化。

**自证机制**：与 `label_urgency.py` 同构，本脚本导出 `noul_antibody.jsonl` ——
一批**关键词命中但判定应当相反**的中文句子（人工写，不参与训练），并在 `--report` 里
与 `baseline_rules.predict_noul` 并排报出。若本 rubric 与规则基线在反例集上答得一样，
就说明两者学的是同一套关键词，本脚本没有独立价值（`relabel_selfcheck` 会把这种情况报出来）。

用法：

    python scripts/label_noul.py --in-dir  $WORK\\data\\adapted_v03 ^
        --out-dir $WORK\\data\\adapted_v04 ^
        --report $WORK\\data\\adapted_v04\\relabel_report.json

只覆盖 `needs_refund_or_escalation` 一维；`route_team` 与 `urgency` 原样保留
（urgency 已由 #16 重标，route_team 本来就是真标签）。

标签写成 `{"false": 0.1, "true": 0.9}` 这类**软标签**，与 `scripts/build_dataset.py`
既有的 noul 目标格式一致（那里用 GOLD_STRONG=0.9 软化，因为 proxy 本身有噪声；
重标后噪声来源变了，但训练侧格式保持不变，避免两处不一致）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mai_jev.questions import QUESTIONS  # noqa: E402

SPLITS = ("train", "calib", "test")
GOLD_STRONG = 0.9  # 与 build_dataset.py 的 noul 软标签保持一致

NOUL_QID = "needs_refund_or_escalation"


# ── 1. 诉求对象：出现这些词说明句子在谈「退款 / 赔付 / 升级」这件事 ───────────
# 两个名词词表被下面多处复用（否定剥离也要用），所以单独拎出来拼正则。
REFUND_OBJ_ALT = (r"退款|退費|退费|退钱|退錢|退還|退还|退回|返还|返還|退貨|退货|赔付|賠付|"
                  r"赔偿|賠償|理赔|理賠|補償|补偿|价保|價保|价格保护|價格保護|退差價|退差价|差价|差價|"
                  r"退款金額|退款金额")
ESCALATE_OBJ_ALT = r"投诉|投訴"

REFUND_OBJECT = re.compile(r"(" + REFUND_OBJ_ALT + r")")

# 要求动词：出现在诉求对象**左侧 10 字窗口**里就是「明确提出」。
STRONG_CUE = re.compile(
    r"(要求|申請|申请|必須|必须|务必|務必|帮我|幫我|给我|給我|還我|还我|退我|"
    r"还要|還要|也要|只要求|不只要|"
    r"麻烦|麻煩|請|请|快点|快點|马上|馬上|立刻|立即|尽快|盡快|优先|優先|拜托|拜託)")

# 「了解 / 咨询 / 问一下 / 查一下」这类学习动词：它们夹在动词与对象之间时，
# 「我想了解一下退款流程」不能被当成「我想退款」。
# 注意 `查` / `看` / `问` **不能只写单字**：`我要马上查明跟赔偿` 是诉求，
# 单字 `查` 会让守卫误杀它（眼验抓到的第二类漏报）。
LEARN_CUE = re.compile(
    r"(了解|瞭解|咨询|諮詢|问一下|問一下|问问|問問|想问|想問|询问|詢問|知道|"
    r"查一下|查一查|查询|查詢|查看|看清楚|"
    r"流程|步驟|步骤|手续|手續|政策|規定|规定|条款|條款|规则|規則|制度|程序|"
    r"需要什么|要什么材料|材料)")

# 资格询问：口径（宽松）判定为 true —— 问「能不能退」也算有退款意向。
# 残差探针（2026-10-17）补进来的形态：`是不是能退货`、`是不是该直接退货算了`、
# `有没有什么补偿方案`、`有什么退款机制`。它们与「问流程」的差别是要的是**结果**而不是**知识**。
COMPENSATION_OBJ_ALT = (r"退款|退貨|退货|退費|退费|理赔|理賠|赔付|賠付|赔偿|賠償|差价|差價|价保|價保")

ELIGIBILITY_CUE = re.compile(
    r"(能不能|可不可以|可否|能否|是否可|可以嗎|可以吗|行不行|能退|可退|能申請|能申请|"
    r"是不是(能|可以|該|该|要|需要)|"
    r"有没有|有沒有|有什么|有什麼|有哪些|"
    # 残差探针第二批：`還能退貨嗎`、`可以辦理退貨退款嗎`、`有辦法退貨嗎`。
    # 用否定左界避免 `不能退貨` / `沒有辦法退貨`（那是**否定**，不是询问）。
    r"(?<![不沒没])(可以|可|能)[^。！？!?]{0,3}(" + COMPENSATION_OBJ_ALT + r")|"
    r"(?<![沒没])(有办法|有辦法|有无|有無))")

# 「A 还是 B」的择一诉求：`是要理賠還是補寄呢` / `不然就退貨退款`。
# 它在让客服给一个补救方案，属诉求而非问知识。
CHOICE_QUESTION = re.compile(
    r"(" + COMPENSATION_OBJ_ALT + r")"
    r"[^。！？!?]{0,6}(还是|還是|或|或是|不然|不然就)"
    r"[^。！？!?]{0,6}(退款|退貨|退货|退費|退费|理赔|理賠|赔付|賠付|赔偿|賠償|補寄|补寄|换货|換貨)")

# 直接意愿形态：`要退貨了` / `想退貨` / `要退款了`。允许 0–3 字间隔，但命中区间里
# 出现学习动词（`想了解一下退货`）就作废 —— 这条由阶梯里的 LEARN_CUE 守卫完成。
DIRECT_INTENT = re.compile(
    r"((要|想要|想|得|會|会)[^。！？!?]{0,3}(" + COMPENSATION_OBJ_ALT + r"))")

# 自陈意愿：`我要退款` / `我想退货` / `真心想退货算了`。与「我想了解」的区分靠 LEARN_CUE。
WILL_CUE = re.compile(
    r"(我要|我想要|我想|想要|想|打算|準備|准备|考慮|考虑|我要求|"
    r"还要|還要|也要|只要求|不只要)")

# 条件式意愿：`不然真的是要退货了` / `我觉得我根本要取消订单並退款`。
# 必须要求 `要` / `该` 紧跟其后，否则「不然的话，退货政策是什么」会被误判成诉求。
CONDITIONAL_INTENT = re.compile(r"(不然|否則|否则|根本|真的)[^。！？!?]{0,4}(要|該|该)")

# 「只是好奇 / 随便问问」是**没有诉求**，不算「问资格」：
# `东西没问题，我就是随便问问有没有退款这个选项`。这是宽松口径的唯一例外，写在这里备查。
CURIOSITY = re.compile(
    r"(随便问问|隨便問問|随便问|随便一問|就是问问|就是問問|纯粹问问|純粹問問|"
    r"顺便问问|順便問問|好奇|问一下而已|問一下而已)")

# 诉求对象**右侧 6 字窗口**里的诉求语气：`退款吧` / `退到我卡里` / `赔给我`。
POST_DEMAND = re.compile(r"^(的)?(給我|给我|到我|回來|回来|吧|就行|就好)")

# 独立的诉求形式（不带名词对象也能自证）：`把钱退我` / `還我錢` / `退款給我`。
STANDALONE_DEMAND = re.compile(
    r"(退(錢|钱|款|費|费)(给|給)我|退我(錢|钱|款)?|還我(錢|钱|款)|还我(錢|钱|款)|"
    r"把钱退|把錢退|退到我的|原路退回|退回來|退回来|赔付我|賠付我|赔偿我|賠償我)")

# 追讨**已申请 / 已承诺**的退款或补偿 —— 这是诉求的进行时态：
# `退款怎么还没到账，都五天了`、`说好的补偿到现在都没消息`。
# 它在 zh-bench 34 行（大陆简体）上暴露了成片漏报，因为那里的说法是「退款没到账」而不是「我要求退款」。
REFUND_PENDING = re.compile(
    r"(" + REFUND_OBJ_ALT + r")"
    r"[^。！？!?]{0,8}"
    r"(还没|還沒|还是没|還是沒|没收到|沒收到|没到账|沒到帳|没到帐|未收到|未到账|未到帳|"
    r"没消息|沒消息|没下文|沒下文|没处理|沒處理|没结果|沒結果|没动静|沒動靜|没进展|沒進展|"
    r"迟迟|遲遲|都[0-9一二三四五六七八九十]+天)")

# 不带名词对象的资格询问：`定金能退吗`、`这筆錢還可以退嗎`、`能不能退換貨`。
# 阶梯的对象循环进不来（句子里根本没有「退款/退货」这样的名词），所以要单独一步。
STANDALONE_ELIGIBILITY = re.compile(
    r"(?<![不沒没])(能不能|可不可以|能否|可否|可以|能|可)[^。！？!?]{0,3}退"
    r"[^。！？!?]{0,3}(?:[吗嗎吧么麼呢][^。！？!?]*|[？?。！!，,]|$)")

# ── 2. 升级：要求升级或威胁（口径判定 true），但「问投诉本身」不算 ─────────────
ESCALATION = re.compile(
    r"(投诉|投訴|举报|舉報|检举|檢舉|申诉|申訴|消保|消基會|消费者协会|消費者協會|"
    r"消费者保护协会|消費者保護協會|消费者保护|消費者保護|"
    r"监管|監管|工商|起诉|起訴|诉讼|訴訟|打官司|律师|律師|"
    r"媒体|媒體|记者|記者|新闻|新聞|曝光|舆情|輿情|上级|上級|主管|总部|總部|"
    r"经理|經理|提告|法院|差评|差評|负评|負評|发到网上|發到網上|爆料|"
    # 维权热线必须用**字母数字边界**：`订单編號是123456789` 含 `12345`、
    # `订单#A12345` 的 `A12345` 也含 `12345` —— 不加边界会把普通订单号判成「要找消协」。
    # 裸 `315` 不收了：它在中文里太常表示价格（`一共315元`），收益不抵误报。
    r"(?<![A-Za-z0-9])(?:12315|12345)(?![0-9]))")
# 中性问法：问投诉**本身**（流程 / 怎么投 / 以前投过）不是威胁。
ESCALATION_NEUTRAL = re.compile(
    r"((投诉|投訴|申訴|申诉)[^。！？!?]{0,4}(流程|程序|制度|电话|電話|记录|記錄|渠道|邮箱|郵箱|是什么)"
    r"|(如何|怎么|怎麼|怎样|怎樣|想問|想问|咨询|諮詢|了解|瞭解)[^。！？!?]{0,6}(投诉|投訴|申訴|申诉)"
    r"|(之前|以前|曾经|曾經|朋友|同事)[^。！？!?]{0,4}(投诉|投訴|申訴|申诉)过)")
# 威胁句：它压过中性问法 —— `我投诉过你们三次都没人理，再不解决我就投诉消协` 仍然是威胁。
THREAT = re.compile(
    r"((再不|不(解决|处理|给)|否则|否則|不然|我就|我就要|我就去|只能|只好|准备|準備|打算)"
    r"[^。！？!?]{0,8}(投诉|投訴|举报|舉報|曝光|消協|消协|消保|消基會|媒体|媒體|监管|監管|"
    r"起诉|起訴|法院|律师|律師|主管|上级|上級))")


def _is_escalation(text: str) -> bool:
    """升级要求 / 威胁 → True；问投诉本身（流程、怎么投、以前投过）→ False。"""
    if not ESCALATION.search(text):
        return False
    if THREAT.search(text):
        return True
    return not ESCALATION_NEUTRAL.search(text)


# ── 3. 否定剥离：把「不是要退款」整段挖掉，剩下的文本继续走阶梯 ───────────────
DEMAND_NEGATED = re.compile(
    r"(不是(要|想|要求)?(" + REFUND_OBJ_ALT + r"|" + ESCALATE_OBJ_ALT + r")"
    r"|(不|沒|没)(有)?(要|想|打算)(" + REFUND_OBJ_ALT + r"|" + ESCALATE_OBJ_ALT + r")"
    r"|不需要(" + REFUND_OBJ_ALT + r")"
    r"|不用(退款|退貨|退货|赔|賠|补|補)"
    r"|(別|别)(退款|退貨|退货))")
# 条件句里出现的「不退款」是**诉求**而不是否定：`如果不退款我就投诉`。
CONDITION_GUARD = re.compile(
    r"(如果|要是|若|萬一|万一|再不|再拖|否则|否則|不肯|不给|不給|拒绝|拒絕|拖着|一直不)")


def _strip_negated_demands(text: str) -> str:
    """挖掉「不是要退款」这类自我否定片段，但放过条件句里的「不退款」。"""
    spans = []
    for m in DEMAND_NEGATED.finditer(text):
        pre = text[max(0, m.start() - 12): m.start()]
        if CONDITION_GUARD.search(pre):
            continue  # 条件句：`如果不退款我就投诉`，这是诉求
        spans.append((m.start(), m.end()))
    if not spans:
        return text
    out, last = [], 0
    for start, end in spans:
        out.append(text[last:start])
        out.append(" ")
        last = end
    out.append(text[last:])
    return "".join(out)


# ── 4. 规则/流程询问：有诉求词，但问的是知识而不是意愿 ───────────────────────
RULE_QUESTION = re.compile(
    r"(流程|步驟|步骤|手续|手續|怎么走|怎麼走|怎么办|怎麼辦|怎么(处理|處理|办|辦|操作|申請|申请)|"
    r"如何(办理|辦理|申請|申请)|"
    r"運費|运费|郵費|邮费|运費|运费|谁负担|誰負擔|谁出|誰出|誰付|谁付|政策|規定|规定|"
    r"条款|條款|规则|規則|制度|写了几天|規定了|规定了)")


def label_noul(text: str) -> tuple[str, str]:
    """返回 `(标签, 证据标签)`。标签是 "true" / "false"。

    分级顺序即证据强度顺序，**第一个命中的就是最终判定**（不累加）。
    """
    t = (text or "").strip()
    if not t:
        return "false", "empty"

    # 0) 否定剥离：`不是要退款` → 挖掉，剩下的继续判。
    cleaned = _strip_negated_demands(t)

    # 1) 升级要求 / 威胁 —— 是充分条件，但要排除「问投诉本身」。
    if _is_escalation(cleaned):
        return "true", "escalation"

    # 2) 独立诉求形式：`把钱退我` / `原路退回` / `退款給我`。
    if STANDALONE_DEMAND.search(cleaned):
        return "true", "explicit_demand"

    # 2a) 追讨已申请 / 已承诺的退款或补偿：`退款怎么还没到账`、`说好的补偿没消息`。
    #     学习动词出现在命中区间里就作废（`退款政策还没搞清楚` 是问规则）。
    for m in REFUND_PENDING.finditer(cleaned):
        if not LEARN_CUE.search(m.group(0)):
            return "true", "refund_pending"

    # 2a2) 不带名词对象的资格询问：`定金能退吗`。看一眼 ±12 字窗口里的学习动词再判。
    for m in STANDALONE_ELIGIBILITY.finditer(cleaned):
        window = cleaned[max(0, m.start() - 12): m.end() + 6]
        if not LEARN_CUE.search(window):
            return "true", "standalone_eligibility"

    # 2b) 择一诉求：`是要理賠還是補寄呢` —— 让客服给一个补救方案，属诉求。
    if CHOICE_QUESTION.search(cleaned):
        return "true", "choice_question"

    # 2c) 直接意愿形态：`要退货了` / `想退貨`（区间里若有学习动词则跳过）。
    for m in DIRECT_INTENT.finditer(cleaned):
        if not LEARN_CUE.search(m.group(1)):
            return "true", "direct_intent"

    # 3) 逐个诉求对象看它的左右窗口，判定「要求 / 意愿 / 资格询问」。
    for m in REFUND_OBJECT.finditer(cleaned):
        pre = cleaned[max(0, m.start() - 10): m.start()]
        post = cleaned[m.end(): m.end() + 6]
        span = cleaned[max(0, m.start() - 10): m.end() + 6]
        curious = bool(CURIOSITY.search(pre))
        if STRONG_CUE.search(pre) and not LEARN_CUE.search(pre):
            return "true", "explicit_demand"
        if WILL_CUE.search(pre) and not LEARN_CUE.search(pre) and not curious:
            return "true", "declared_intent"
        if CONDITIONAL_INTENT.search(pre) and not LEARN_CUE.search(pre):
            return "true", "conditional_intent"
        # 资格询问要把**对象本身**算进窗口：`還能退貨嗎` 的 `能退` 跨在对象边界上。
        if ELIGIBILITY_CUE.search(span) and not curious:
            return "true", "eligibility_question"
        if POST_DEMAND.search(post):
            return "true", "soft_demand"

    # 4) 有诉求词，但只是在问规则 / 流程 → false（这是「问流程」与「问资格」的分界）。
    if RULE_QUESTION.search(cleaned) and REFUND_OBJECT.search(cleaned):
        return "false", "rule_or_process_question"

    # 5) 只有情绪 / 催促，没有任何退款-升级线索 → false（这是 v0.3 代理多报的主战场）。
    return "false", "no_demand_evidence"


# ── 反例集：人工写的「关键词命中但判定应当相反」句子 ──────────────────────────
# 不参与训练，只在自检与评测里与规则基线并排报。设计意图：下面 false 那一组的
# 关键词（退款/退货/投诉/索赔）会让 `baseline_rules.predict_noul` 判 true；
# true 那一组反过来是它**没有**关键词可命中的明确诉求。
ANTIBODY: list[tuple[str, str]] = [
    # —— 关键词命中，但实际是 false：否定 / 条件外 / 问规则 ——
    ("不是要退款，我只是想了解一下退货的流程", "false"),
    ("我没有要投诉，只是想知道大概什么时候能到", "false"),
    ("不用退款，帮我换个货就行", "false"),
    ("我没有要退款，我要的是换货", "false"),
    ("退货的流程要怎么走呢？我还没搞清楚步骤", "false"),
    ("退货运费是我自己出还是你们负担？", "false"),
    ("退款政策里写了几天可以退？", "false"),
    ("你们的退款手续费怎么算？", "false"),
    ("退货时赠品那台要怎么处理，是要一起退还还是直接送我？", "false"),
    ("投诉流程是什么？我想先了解一下", "false"),
    ("如何投诉你们？我在整理材料", "false"),
    ("我朋友之前投诉过你们，后来处理得挺好，我也想问问要怎么操作", "false"),
    ("别再打电话给我了，我自己会处理", "false"),
    ("东西没问题，我就是随便问问有没有退款这个选项", "false"),
    ("我的订单编号是1234567890，麻烦帮我查一下物流进度", "false"),
    ("我的订单是#A12345，请问什么时候发货？", "false"),
    ("这台机器一共315元，可以开发票吗？", "false"),
    ("不然的话，退货政策是什么？", "false"),
    ("退款政策里写了几天可以退？", "false"),
    ("退款流程还没走完，要等多久？", "false"),
    ("结账显示1500元，银行却扣了1600元，这是怎么回事啊？", "false"),
    # —— 没有关键词命中的明确诉求，实际是 true：升级要求 / 威胁 / 赔付 ——
    ("再不处理我就去找消费者保护协会了", "true"),
    ("我要找你们主管来谈这件事", "true"),
    ("你们要是不给个说法，我就把这事发到网上让大家看看", "true"),
    ("麻烦把钱原路退回到我的卡里", "true"),
    ("这单我已经不想要了，钱退我就行", "true"),
    ("请给我一个赔偿方案", "true"),
    ("能不能退我一半的钱作为补偿？", "true"),
    ("有没有什么补偿方案？", "true"),
    ("是要理賠還是補寄呢？", "true"),
    ("是不是能退貨？", "true"),
    ("不然真的是要退货了", "true"),
    ("退款怎么还没到账，都五天了", "true"),
    ("说好的补偿到现在都没消息", "true"),
    ("能补差价吗？", "true"),
    ("定金能退吗？", "true"),
    ("我要马上理賠或補寄", "true"),
]


def antibody_rows() -> list[dict]:
    """反例集 -> 与训练数据同构的行。"""
    rows = []
    for idx, (text, label) in enumerate(ANTIBODY):
        strong = {"true": GOLD_STRONG, "false": 1.0 - GOLD_STRONG}
        probs = {"false": round(1.0 - strong[label], 3), "true": round(strong[label], 3)}
        rows.append({
            "id": f"noul-antibody-{idx:03d}",
            "state": text,
            "questions": {NOUL_QID: dict(QUESTIONS[NOUL_QID])},
            "gold": {NOUL_QID: {"probabilities": probs}},
            "split": "antibody",
            "provenance": {"source": "handwritten", "license": "Apache-2.0",
                           "proxy": "handwritten-noul-rubric"},
        })
    return rows


# ── 重标注主流程 ────────────────────────────────────────────────────────────

def relabel(rows: Iterable[dict]) -> tuple[list[dict], Counter]:
    """就地覆盖每行的 `gold.needs_refund_or_escalation.probabilities`。"""
    out, evidence = [], Counter()
    by_label: Counter = Counter()
    for rec in rows:
        rec = dict(rec)
        text = str(rec.get("state") or "")
        label, why = label_noul(text)
        by_label[label] += 1
        evidence[why] += 1

        gold = dict(rec.get("gold") or {})
        noul = dict(gold.get(NOUL_QID) or {})
        strong = {"true": GOLD_STRONG, "false": 1.0 - GOLD_STRONG}[label]
        noul["probabilities"] = {"false": round(1.0 - strong, 3), "true": round(strong, 3)}
        gold[NOUL_QID] = noul
        rec["gold"] = gold
        rec["questions"] = dict(rec.get("questions") or {})
        # 必须写回**完整**的问题定义：laya 的 items_from_rows 缺定义时会静默跳过该问题
        # （label_urgency.py 已经踩过一次，见那里的注释）。
        rec["questions"][NOUL_QID] = dict(QUESTIONS[NOUL_QID])
        prov = dict(rec.get("provenance") or {})
        prov["noul_old_proxy"] = prov.get("noul_proxy") or prov.get("proxy", "")
        prov["noul_proxy"] = f"synthetic-rubric:{why}"
        rec["provenance"] = prov
        out.append(rec)
    return out, evidence


def noul_label_of(rec: dict) -> str:
    probs = (rec.get("gold") or {}).get(NOUL_QID, {}).get("probabilities") or {}
    if not probs:
        return "?"
    return max(probs, key=lambda k: probs[k])


def relabel_selfcheck() -> dict:
    """自检：rubric 必须与人写的反例集一致，且**应该**与规则基线不一致。

    1. `label_noul(text) == want` —— 否则是本 rubric 写错了；
    2. `baseline_rules.predict_noul(text) != want` 的比例 —— 若规则基线把反例集全答对，
       说明两者是同一套关键词，本标注器没有独立价值。
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from baseline_rules import predict_noul

    rubric_bad, rules_right = [], []
    for text, want in ANTIBODY:
        got, _ = label_noul(text)
        if got != want:
            rubric_bad.append(f"{text!r} rubric={got} want={want}")
        if ("true" if predict_noul(text) else "false") == want:
            rules_right.append(f"{text!r}")
    return {"rubric_misses": rubric_bad,
            "rules_baseline_correct": rules_right,
            "n": len(ANTIBODY),
            "divergence": f"{len(ANTIBODY) - len(rules_right)}/{len(ANTIBODY)} "
                          f"rows where rules baseline disagrees with the antibody set"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="待重标的 adapted 目录（含 mai_jev_*.jsonl）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="mai_jev")
    ap.add_argument("--report", help="统计报告 JSON 路径（默认写到 out-dir）")
    args = ap.parse_args()

    in_dir, out_dir = Path(args.in_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    check = relabel_selfcheck()
    if check["rubric_misses"]:
        print("ABORT: 标注器与人写反例集不一致，先修 rubric：")
        for p in check["rubric_misses"]:
            print("  -", p)
        return 1
    print(f"selfcheck: rubric OK on all {check['n']} antibody rows; {check['divergence']}")

    report = {"input_dir": str(in_dir), "prefix": args.prefix, "splits": {},
              "evidence_histogram": {}, "selfcheck": check}

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
        before = Counter(noul_label_of(r) for r in rows)
        after = Counter(noul_label_of(r) for r in new_rows)
        n = sum(after.values())
        report["splits"][split] = {
            "rows": len(new_rows),
            "true_rate_before": round(before.get("true", 0) / max(1, sum(before.values())), 4),
            "true_rate_after": round(after.get("true", 0) / max(1, n), 4),
            "label_dist_after": dict(sorted(after.items())),
            "majority_baseline_after": round(max(after.values()) / max(1, n), 4),
        }
        for k, v in evidence.items():
            report["evidence_histogram"][k] = report["evidence_histogram"].get(k, 0) + v
        print(f"{split}: {len(new_rows)} rows -> {dst}")
        print(f"  true rate {report['splits'][split]['true_rate_before']:.3f}"
              f" -> {report['splits'][split]['true_rate_after']:.3f}"
              f"  (majority baseline {report['splits'][split]['majority_baseline_after']:.3f})")

    ab = out_dir / f"{args.prefix}_noul_antibody.jsonl"
    with ab.open("w", encoding="utf-8") as fh:
        for rec in antibody_rows():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    report["antibody"] = {"path": str(ab), "rows": len(ANTIBODY)}

    rp = Path(args.report) if args.report else out_dir / "relabel_report_noul.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", rp)
    print("antibody ->", ab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
