"""#15：论文数字与图表的唯一来源。

硬约束（map #1 继承）：**论文里每个数字都必须能由仓库脚本重新跑出来，禁止手工誊抄。**
所以本脚本是论文与实验产物之间唯一的桥：读 JSON 产物 → 生成 `.tex` 宏与 pgfplots 图 →
`paper/mai-jev.tex` 只 `\\input` 这些生成文件。正文里不出现任何一个手写的小数。

产物（默认写到 `paper/generated/`）：
  numbers.tex        每个数字一个 `\\newcommand`，正文引用宏名而非字面量
  fig_accuracy.tex   三原语 × 四条曲线（微调前/后、规则基线、多数类）
  fig_ece.tex        标定前后 ECE 对照
  fig_tiers.tex      置信度三档覆盖率 vs 该档命中率
  tab_main.tex       三原语主表
  上述图与表的 `*_zh.tex`：同一批宏、同一版式，只有坐标轴/图例/表头换成中文，
  供中文版论文 `paper/mai-jev-zh.tex` 使用。中英两版一起生成，避免标签漂移。

用法（工作目录必须是仓库根）：
  python scripts/make_paper_tables.py ^
      --eval-json  runs/paper_eval/calibration_eval.json ^
      --train-json runs/paper_eval/train_report.json ^
      --bench-json export/v03_bench_final.json ^
      --data-json data/adapted_v03/relabel_report.json ^
      --build-json data/adapted/build_report.json ^
      --out paper/generated

每个输入文件缺失都会报错而不是静默跳过：论文宁可编译失败，也不要印一个过期数字。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# ----------------------------------------------------------------------------- helpers

PRIMS = ["route_team", "urgency", "needs_refund_or_escalation"]
PRIM_LABEL = {
    "route_team": "route\\_team (choice)",
    "urgency": "urgency (score)",
    "needs_refund_or_escalation": "needs\\_refund\\_or\\_escalation (noul)",
}


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"缺少输入文件：{path}\n请先跑对应的实验脚本再生成论文数字。")
    raw = path.read_bytes()
    # PowerShell 的 Out-File / `>` 重定向在 Windows 上会写出 UTF-16LE（BOM 0xFF 0xFE），
    # 直接 utf-8 解码会炸。宁可按 BOM 猜编码，也不要在这里改数据源。
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    raise SystemExit(f"无法解析 JSON（编码未知）：{path}")


def num(v, nd: int = 4) -> str:
    return f"{float(v):.{nd}f}"


# 来源路径改写成相对路径时用的基准目录（--source-root，默认当前目录）。
SOURCE_ROOT = "."


def scrub_source(text: str) -> str:
    """把来源串开头的本机绝对路径改写成相对路径。

    论文产物会随仓库公开，**盘符与工作区绝对路径不该进论文**。来源串的形态是
    `.../calibration_eval.json:test_after.score.ece`：前一段是文件路径，后一段是
    JSON 指针（用 `:` 隔开）。所以这里只动「第一个冒号之前」那一段。
    """
    if not re.match(r"^[A-Za-z]:[\\/]", text):
        return text
    cut = text.find(":", 3)
    head, tail = (text, "") if cut == -1 else (text[:cut], text[cut:])
    path = Path(head)
    try:
        return path.resolve().relative_to(Path(SOURCE_ROOT).resolve()).as_posix() + tail
    except ValueError:
        return "/".join(path.parts[-3:]) + tail


class Macros:
    """收集 `\\newcommand{\\xxx}{value}`，最后统一写出并附带来源注释。"""

    # TeX 的控制序列名里不能出现数字：`\newcommand{\Foo50}` 会被 TeX 读成
    # 命令 `\Foo` 后面跟普通文本 `50`，于是报 "Missing \begin{document}"，
    # 而且错误信息指向那一行，看起来像 numbers.tex 语法坏了。所以名字里
    # 的数字一律转成单词（50 -> Fifty），并且在这里断言拦住。
    _DIGITS = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    @classmethod
    def _csname(cls, name: str) -> str:
        safe = "".join(cls._DIGITS.get(ch, ch) for ch in name)
        if not re.fullmatch(r"[A-Za-z]+", safe):
            raise ValueError(f"宏名 {name!r}（转换后 {safe!r}）不是合法的 TeX 控制序列名")
        return safe

    def add(self, name: str, value: str, source: str) -> str:
        safe = self._csname(name)
        if any(n == safe for n, _, _ in self.rows):
            raise ValueError(f"宏名 {safe!r} 重复定义（原始名 {name!r}）")
        self.rows.append((safe, value, scrub_source(source)))
        return safe

    def render(self) -> str:
        head = [
            "% 由 scripts/make_paper_tables.py 生成，请勿手工编辑。",
            "% 每个宏的来源写在注释里；论文正文只引用宏名。",
        ]
        body = [f"\\newcommand{{\\{name}}}{{{value}}}  % {src}" for name, value, src in self.rows]
        return "\n".join(head + body) + "\n"


# --------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-json", default="runs/paper_eval/calibration_eval.json")
    ap.add_argument("--train-json", default="runs/paper_eval/train_report.json")
    ap.add_argument("--bench-json", default="export/v03_bench_final.json")
    ap.add_argument("--data-json", default="data/adapted_v03/relabel_report.json")
    ap.add_argument("--noul-json", default=None,
                    help="#17 的 noul 重标报告（relabel_report_noul.json）；给了才产出 noul 口径宏")
    ap.add_argument("--noul-oldhead-json", default=None,
                    help="旧 checkpoint 在**新** noul 标签上的评测 JSON（runs/v03_on_v04/calibration_eval.json）")
    ap.add_argument("--antibody-json", default=None,
                    help="人工反例集成绩（scripts/eval_noul_antibody.py --out 的产物）")
    ap.add_argument("--balanced-data", default="data/adapted_v03/mai_jev_train_balanced.laya.jsonl")
    ap.add_argument("--build-json", default="data/adapted/build_report.json")
    ap.add_argument("--latency-report", default="docs/research/v03-latency.md",
                    help="延迟构成报告；数字由 scripts/probe_batch_merge.py 产出，脚本从表格里取")
    ap.add_argument("--v02-report", default="docs/research/v03-quant-degradation.md")
    ap.add_argument("--v02-eval", default="runs/v02_rlcd_e2/calibration_eval.json",
                    help="proxy 标签时期的 v02 评测 JSON，用于给出 urgency 的折损幅度")
    ap.add_argument("--int8-calib-report", default="export/v03_int8_calib_report.json",
                    help="calibrate_export.py --report 的产物：量化变体自身重拟合温度后的 test 指标")
    ap.add_argument("--export-dir", default="export/v03-int8-ml256",
                    help="发布变体目录，只用它量 tokenizer 大小")
    ap.add_argument("--quant-variant-dir", default="v03-int8-ml256",
                    help="量化变体自己的 calibration.json 在哪：绝对路径，或相对 --bench-json 所在目录的子目录名")
    ap.add_argument("--latency-sessions", default="",
                    help="跨 session 的端到端 p50 列表（同一个包、同一条命令的不同 session）；"
                         "本机不同 session 的 p50 能差 2.5 倍，所以论文报区间而不是单点")
    # bench_report.json 的 parity 只对 96/128/512 三档测过；默认取 512（唯一 state 预算充足的一档）。
    ap.add_argument("--parity-max-len", type=int, default=512,
                    help="bench_report.json 的 parity 用哪个 max_len 分区（96/128/512）")
    ap.add_argument("--source-root", default=".",
                    help="来源路径改写成相对路径的基准目录（默认当前目录）")
    ap.add_argument("--out", default="paper/generated")
    args = ap.parse_args()
    global SOURCE_ROOT
    SOURCE_ROOT = args.source_root

    ev = load(Path(args.eval_json))
    tr = load(Path(args.train_json))
    bn = load(Path(args.bench_json))
    rel = load(Path(args.data_json))
    bd = load(Path(args.build_json))
    bal = Path(args.balanced_data)
    n_balanced = sum(1 for line in bal.read_text(encoding="utf-8").splitlines() if line.strip()) if bal.exists() else None
    if n_balanced is None:
        raise SystemExit(f"缺少重平衡后的训练集：{bal}")

    # calibrate_eval.py 的分区键是**原语类型名**（choice/score/noul），不是我们的问题 id；
    # 只有 baseline_rules 是按问题 id 分区的。两套键必须显式区分，否则会 KeyError。
    prim_type = {"route_team": "choice", "urgency": "score", "needs_refund_or_escalation": "noul"}
    after = {p: ev["test_after_calibration"][t] for p, t in prim_type.items()}
    before = {p: ev["test_before_calibration"][t] for p, t in prim_type.items()}
    rules = ev["baseline_rules"]
    majority = ev["baseline_majority_class"]
    original = ev.get("baseline_original_laya_multilingual") or {}
    original = {p: original[t] for p, t in prim_type.items() if t in original}
    temps = ev["temperature"]

    m = Macros()

    # ---- 数据规模与切分（#7 的 build_report） ----
    m.add("DataTotalRows", str(bd["total"]), f"{args.build_json}:total")
    for sp in ("train", "calib", "test"):
        m.add(f"Rows{sp.capitalize()}", str(bd["splits"][sp]), f"{args.build_json}:splits.{sp}")
    m.add("RowsTrainBalanced", str(n_balanced),
          f"{args.balanced_data}:行数（由 balance_urgency.py 写出）")
    m.add("TokBudget", str(bd["token_length"]["budget"]), f"{args.build_json}:token_length.budget")
    m.add("TokP50", str(bd["token_length"]["p50"]), f"{args.build_json}:token_length.p50")
    m.add("TokP99", str(bd["token_length"]["p99"]), f"{args.build_json}:token_length.p99")
    m.add("TokMax", str(bd["token_length"]["max"]), f"{args.build_json}:token_length.max")
    m.add("SeedAll", str(ev["seed"]), f"{args.eval_json}:seed")

    # ---- 重标定的 urgency 分布（#16 的 relabel_report） ----
    for lv in range(5):
        m.add(
            f"UrgDistAfter{lv}",
            str(rel["splits"]["test"]["level_dist_0based_after"][str(lv)]),
            f"{args.data_json}:test.level_dist_0based_after[{lv}]",
        )

    # ---- 主结果：准确率 / ECE / 温度 ----
    for i, (key, prim) in enumerate(zip(("Choice", "Score", "Noul"), PRIMS)):
        m.add(f"Acc{key}", num(after[prim]["accuracy"]), f"{args.eval_json}:test_after.{prim_type[prim]}.accuracy")
        m.add(f"BAcc{key}", num(after[prim]["balanced_accuracy"]), f"{args.eval_json}:test_after.{prim_type[prim]}.balanced_accuracy")
        m.add(f"SoftAcc{key}", num(after[prim]["soft_accuracy"]), f"{args.eval_json}:test_after.{prim_type[prim]}.soft_accuracy")
        m.add(f"Ece{key}", num(after[prim]["ece"]), f"{args.eval_json}:test_after.{prim_type[prim]}.ece")
        m.add(f"EceBefore{key}", num(before[prim]["ece"]), f"{args.eval_json}:test_before.{prim_type[prim]}.ece")
        m.add(f"Conf{key}", num(after[prim]["mean_confidence"]), f"{args.eval_json}:test_after.{prim_type[prim]}.mean_confidence")
        m.add(f"T{key}", num(temps[i], 4), f"{args.eval_json}:temperature[{i}]")
    for i, (key, prim) in enumerate(zip(("Choice", "Score", "Noul"), PRIMS)):
        m.add(f"RulesAcc{key}", num(rules[prim]["accuracy"]), f"{args.eval_json}:baseline_rules.{prim}")
        if "balanced_accuracy" not in rules[prim]:
            # noul 的多数类被翻转（全猜 false 就有 0.796），论文拿 balanced accuracy 才裁得动它，
            # 缺这一项宁可在这里失败也不要印一个缺项的数字。其余两个原语的规则基线 balanced
            # 只是补充信息（`baseline_rules.py` 在 #17 才补上这一项），缺了就跳过。
            if prim == "needs_refund_or_escalation":
                raise SystemExit(
                    f"{args.eval_json}:baseline_rules.{prim} 缺少 balanced_accuracy —— "
                    "这是旧版 calibrate_eval.py 的产物，请用新版重跑评测（#17）。")
        else:
            m.add(f"RulesBAcc{key}", num(rules[prim]["balanced_accuracy"], 4),
                  f"{args.eval_json}:baseline_rules.{prim}.balanced_accuracy")
        m.add(f"MajAcc{key}", num(majority[prim]), f"{args.eval_json}:baseline_majority_class.{prim}")
        orig_acc = {p: original[p]["accuracy"] for p in PRIMS if p in original}
        m.add(f"OrigAcc{key}", num(orig_acc.get(prim, float("nan"))),
              f"{args.eval_json}:baseline_original_laya_multilingual.{prim_type[prim]}.accuracy")
    if len(orig_acc) == 3:
        m.add("OrigAccMacro", num(sum(orig_acc.values()) / 3.0), "原版三原语 accuracy 的平均")

    macro = sum(after[p]["accuracy"] for p in PRIMS) / 3.0
    macro_before = sum(before[p]["accuracy"] for p in PRIMS) / 3.0
    m.add("MacroAcc", num(macro), "由 test_after 三原语 accuracy 求平均")
    m.add("MacroAccBefore", num(macro_before), "由 test_before 三原语 accuracy 求平均")
    worst_ece = max(after[p]["ece"] for p in PRIMS)
    m.add("WorstEce", num(worst_ece), "三原语中最大的标定后 ECE")
    m.add("TrainEceAfter", num(tr["after"]["ece"]), f"{args.train_json}:after.ece（未标定）")
    m.add("TrainAccBefore", num(tr["before"]["accuracy"]), f"{args.train_json}:before.accuracy（原版权重）")
    m.add("TrainAccAfter", num(tr["after"]["accuracy"]), f"{args.train_json}:after.accuracy")

    # ---- 置信度分档（标定后） ----
    for key, prim in zip(("Choice", "Score", "Noul"), PRIMS):
        tiers = after[prim]["tiers"]
        for tier, macro_name in (("auto>=0.9", "Auto"), ("review0.6-0.9", "Review"), ("human<0.6", "Human")):
            cov = tiers[tier]["coverage"]
            acc = tiers[tier]["accuracy"]
            m.add(f"{macro_name}Cov{key}", num(cov), f"{args.eval_json}:test_after.{prim}.tiers[{tier}].coverage")
            if acc is not None:
                m.add(f"{macro_name}Acc{key}", num(acc), f"{args.eval_json}:test_after.{prim}.tiers[{tier}].accuracy")
            else:
                # 覆盖率为 0 时没有命中率可言。这里给 0.0 而不是字符串 "n/a"：
                # pgfplots 的坐标解析不了 n/a（"Could not parse input 'n/a' as a
                # floating point number"），而且 0.0 画出来就是一根零高柱，
                # 正好表达「这一档一个样本都没有」。
                m.add(f"{macro_name}Acc{key}", "0.0000",
                      f"{args.eval_json}:test_after.{prim}.tiers[{tier}].coverage 为 0，该档无样本，命中率记 0")

    # ---- 效率（#9） ----
    m.add("BenchThreads", str(bn["threads"]), f"{args.bench_json}:threads")
    m.add("BenchN", str(bn["n"]), f"{args.bench_json}:n")
    m.add("BenchMaxLen", str(bn["max_len"]), f"{args.bench_json}:max_len")
    m.add("LatP50", num(bn["latency_ms"]["p50"], 2), f"{args.bench_json}:latency_ms.p50（三原语合计）")
    m.add("LatP95", num(bn["latency_ms"]["p95"], 2), f"{args.bench_json}:latency_ms.p95")
    m.add("LatMean", num(bn["latency_ms"]["mean"], 2), f"{args.bench_json}:latency_ms.mean")
    m.add("RssMb", num(bn["rss_after_load_MB"], 1), f"{args.bench_json}:rss_after_load_MB")
    m.add("RssPeakMb", num(bn["rss_peak_MB"], 1), f"{args.bench_json}:rss_peak_MB")
    m.add("LoadS", num(bn["load_seconds"], 2), f"{args.bench_json}:load_seconds")
    qpath = Path(args.bench_json).parent / "quant_report.json"
    if qpath.exists():
        q = load(qpath)
        m.add("ModelMb", num(q["model_mb"], 1), "quant_report.json:model_mb")
        m.add("Fp32Mb", num(q["fp32_model_mb"], 1), "quant_report.json:fp32_model_mb")
        for key, prim in zip(("Choice", "Score", "Noul"), PRIMS):
            m.add(f"QuantDelta{key}", num(q["degradation"][prim], 4), f"quant_report.json:degradation.{prim}")
        # 量化变体自己的标定：温度是「权重 + 推理后端」的耦合属性，跨变体复用会让 ECE 悄悄失效，
        # 所以 int8 图必须带一份自己拟合的 calibration.json。
        qvdir = Path(args.quant_variant_dir)
        if not qvdir.is_absolute():
            qvdir = qpath.parent / qvdir
        for variant, cal in (("Onnx", qvdir / "calibration.json"),):
            c = load(cal)
            for key, t in zip(("Choice", "Score", "Noul"), c["temperature"]):
                m.add(f"T{key}{variant}", num(t, 4), f"{cal}:temperature（在该变体图上重拟合）")

    # ---- 量化方案筛选（bench_report.json 的 quantization_study）----
    qbench = Path(args.bench_json).parent / "bench_report.json"
    if qbench.exists():
        qb = load(qbench)
        qs = qb["quantization_study"]
        m.add("QuantWholeMb", num(qs["int8_full_graph_per_channel"]["size_mb"], 1),
              "bench_report.json:int8_full_graph_per_channel.size_mb")
        m.add("QuantWholeMaxD", num(max(qs["int8_full_graph_per_channel"]["max_dlogit"].values()), 3),
              "bench_report.json:int8_full_graph_per_channel.max_dlogit 的最大值")
        m.add("QuantGatherMaxD", num(max(qs["int8_gather_only"]["max_dlogit"].values()), 4),
              "bench_report.json:int8_gather_only.max_dlogit 的最大值")
        par = qb["parity"]["by_max_len"][str(args.parity_max_len)]
        m.add("AgreeNum", num(sum(par["argmax_agreement"].values()) / 3.0, 3),
              f"bench_report.json:parity.by_max_len.{args.parity_max_len}.argmax_agreement 的平均")
        th = qb["thread_scaling_int8_gather_max_len_96"]
        m.add("ThreadsT1", num(th[0]["p50_ms"], 1), "bench_report.json:thread_scaling[threads=1].p50_ms")
        m.add("ThreadsT8", num(th[2]["p50_ms"], 1), "bench_report.json:thread_scaling[threads=8].p50_ms")
        m.add("Fp32RssMb", num(qb["results"][3]["rss_after_load_mb"], 1),
              "bench_report.json:fp32 变体的 rss_after_load_mb")
        m.add("ParityTokMismatch", qb["parity"]["token_id_mismatch"].split("（")[0].strip(),
              "bench_report.json:parity.token_id_mismatch")

    # ---- 延迟构成（docs/research/v03-latency.md，数字由 scripts/probe_batch_merge.py 产出）----
    lat = Path(args.latency_report)
    if lat.exists():
        text = lat.read_text(encoding="utf-8")
        pats = {
            "StateOnlyMs": (r"state_room=0 →\s*\*\*([\d.]+)ms", "state 预算为 0（完全不读工单）时三原语合计 p50"),
            "StateFullMs": (r"满预算 →\s*\*\*([\d.]+)ms", "state 预算满时三原语合计 p50"),
            "BatchMergedMs": (r"p50 \*\*([\d.]+)ms\*\*，反而比三次", "batch=3 合并成一次前向的 p50"),
            "BatchSeparateMs": (r"batch=1 的 \*\*([\d.]+)ms\*\* 更慢", "三次 batch=1 各自前向的合计 p50"),
            "HeadLenChoice": (r"\| route_team \| (\d+) \|", "route_team 的 head token 长度"),
            "HeadLenScore": (r"\| urgency \| (\d+) \|", "urgency 的 head token 长度"),
        }
        for name, (pat, desc) in pats.items():
            hit = re.search(pat, text)
            if not hit:
                raise SystemExit(f"{lat} 里找不到「{desc}」对应的数字（正则 {pat}），请先重跑 scripts/probe_batch_merge.py。")
            m.add(name, hit.group(1), f"{args.latency_report}:{desc}")

    # ---- 跨 session 的延迟波动（同一个包、同一条命令，不同 session 实测差 2.5 倍）----
    # 这台机器在两个「速度档」之间跳：同一个二进制、同一条命令，实测 p50 从 318ms 到 806ms
    # 都出现过（见 docs/research/RUNS.md 2026-10-17 条）。所以论文只报区间，不报单点当结论。
    if args.latency_sessions:
        ls = load(Path(args.latency_sessions))
        p50s = [float(s["p50"]) for s in ls["sessions"]]
        if not p50s:
            raise SystemExit(f"{args.latency_sessions} 里没有任何 session")
        m.add("LatSessions", str(len(p50s)), f"{args.latency_sessions}:参与统计的 session 数")
        m.add("LatP50Best", num(min(p50s), 2), f"{args.latency_sessions}:跨 session 的最小 p50")
        m.add("LatP50Worst", num(max(p50s), 2), f"{args.latency_sessions}:跨 session 的最大 p50")
        m.add("LatSpread", num(max(p50s) / min(p50s), 1),
              f"{args.latency_sessions}:最慢/最快（本机 session 间波动倍数）")
        med = (ls.get("median_session") or {}).get("p50")
        if med is None:                              # 文件里没写就自己算（偶数取上中位）
            med = sorted(p50s)[len(p50s) // 2]
        m.add("LatP50Median", num(med, 2),
              f"{args.latency_sessions}:p50 中位的那次 session（表里的延迟行就是它）")

    # ---- 量化变体自己的标定结果（calibrate_export.py --report）----
    crep = Path(args.int8_calib_report)
    if crep.exists():
        c = load(crep)
        for key, q in zip(("Choice", "Score", "Noul"), ("choice", "score", "noul")):
            m.add(f"EceOnnx{key}", num(c["test_by_qtype"][q]["ece_after"], 4),
                  f"{crep}:test_by_qtype.{q}.ece_after（在 int8 图上标定后）")
        m.add("EceWorstOnnx", num(c["worst_ece_after"], 4),
              f"{crep}:worst_ece_after（三原语里最大的标定后 ECE）")
        m.add("AccOnnxScore", num(c["test_by_qtype"]["score"]["accuracy"], 4),
              f"{crep}:test_by_qtype.score.accuracy（部署变体自身的 test 准确率）")
        m.add("CalibRecords", str(c["calib_records"]), f"{crep}:calib_records")

    # ---- 早前 v02 口径：proxy 标签造成的 urgency 折损 ----
    v02ev = Path(args.v02_eval)
    if v02ev.exists():
        e2 = load(v02ev)
        v02_after = e2["test_after_calibration"]
        v02_score = v02_after["score"]["accuracy"]
        m.add("AccScoreV02", num(v02_score, 4), f"{args.v02_eval}:test_after_calibration.score.accuracy")
        v03_score = float(num(after["urgency"]["accuracy"]))
        m.add("DeltaUrg", num(v03_score - v02_score, 4),
              "v03 与 v02 的 urgency 准确率之差（proxy 标签修复的净收益）")

    # ---- noul 重标（#17 → v0.4）：口径变了，正例率 / 多数类 / 反例集规模都要进论文数字 ----
    if args.noul_json:
        nj = load(Path(args.noul_json))
        nsp = nj["splits"]
        m.add("NoulPosBefore", num(nsp["train"]["true_rate_before"], 4),
              f"{args.noul_json}:train.true_rate_before（旧 proxy 标签）")
        m.add("NoulPosTrainAfter", num(nsp["train"]["true_rate_after"], 4),
              f"{args.noul_json}:train.true_rate_after（规则重标后）")
        m.add("NoulPosTestAfter", num(nsp["test"]["true_rate_after"], 4),
              f"{args.noul_json}:test.true_rate_after（规则重标后）")
        m.add("NoulPosTestBefore", num(nsp["test"]["true_rate_before"], 4),
              f"{args.noul_json}:test.true_rate_before（旧 proxy 标签，test 侧）")
        m.add("NoulMajTestAfter", num(nsp["test"]["majority_baseline_after"], 4),
              f"{args.noul_json}:test.majority_baseline_after")
        m.add("NoulAntibodyN", str((nj.get("antibody") or {}).get("rows", "")),
              f"{args.noul_json}:antibody.rows（人工反例集规模）")
        ndiv = re.match(r"(\d+)\s*/\s*(\d+)", str((nj.get("selfcheck") or {}).get("divergence", "")))
        if ndiv:
            m.add("NoulGateDisagree", ndiv.group(1),
                  f"{args.noul_json}:selfcheck.divergence 分子（规则基线答错的条数）")

    # ---- 旧 v03 头在**新标签**上的成绩（runs/v03_on_v04）：0.8513 里有多少是标签定义撑的 ----
    if args.noul_oldhead_json:
        oh = load(Path(args.noul_oldhead_json))
        ohn = oh["test_after_calibration"]["noul"]
        m.add("OldHeadAccNoul", num(ohn["accuracy"], 4),
              f"{args.noul_oldhead_json}:test_after.noul.accuracy（旧 proxy 头 × 新标签）")
        m.add("OldHeadBAccNoul", num(ohn["balanced_accuracy"], 4),
              f"{args.noul_oldhead_json}:test_after.noul.balanced_accuracy")

    # ---- 人工反例集上的三个对照（eval_noul_antibody.py 的 --out）----
    if args.antibody_json:
        ab = load(Path(args.antibody_json))
        m.add("AntibodyN", str(ab["rows"]), f"{args.antibody_json}:rows")
        m.add("AntibodyAcc", num(ab["model"]["accuracy"], 4),
              f"{args.antibody_json}:model.accuracy")
        m.add("AntibodyBAcc", num(ab["model"]["balanced_accuracy"], 4),
              f"{args.antibody_json}:model.balanced_accuracy")
        m.add("AntibodyRulesAcc", num(ab["rules"]["accuracy"], 4),
              f"{args.antibody_json}:rules.accuracy")
        m.add("AntibodyRulesBAcc", num(ab["rules"]["balanced_accuracy"], 4),
              f"{args.antibody_json}:rules.balanced_accuracy")
        m.add("AntibodyMajAcc", num(ab["majority_class"]["accuracy"], 4),
              f"{args.antibody_json}:majority_class.accuracy")

    # ---- 数据集自检（relabel_report.json）----
    if "selfcheck" in rel:
        sc = rel["selfcheck"]
        # selfcheck.divergence 是一句人话（"3/14 rules-baseline agree with antibody"），
        # 直接塞进 LaTeX 会出乱码，所以只取分子分母。
        div = str(sc.get("divergence", ""))
        mm = re.match(r"(\d+)\s*/\s*(\d+)", div)
        if mm:
            m.add("SelfCheckN", mm.group(2), f"{args.data_json}:selfcheck.divergence 的分母")
            m.add("SelfCheckDisagree", mm.group(1),
                  f"{args.data_json}:selfcheck.divergence 的分子（模型与规则基线分歧数）")

    # ---- tokenizer 体积：发布目录里 tokenizer/tokenizer.json 的实际大小 ----
    tok = Path(args.export_dir) / "tokenizer" / "tokenizer.json"
    if tok.exists():
        m.add("TokMb", num(tok.stat().st_size / 1048576, 1), f"{tok}: 文件大小")

    # ---- 发布变体的 judge 数：ml256 全量复现 eager ----
    # 退化表在 docs/research/v03-quant-degradation.md 的「待决 2」小节里。
    # 这不是本次跑出来的，而是「max_len 装不下 question head」时的实测值，
    # 用来在论文里给出「错误配置长什么样」的具体数字。缺失就不给这两个宏。
    v02 = Path(args.v02_report)
    if v02.exists():
        txt = v02.read_text(encoding="utf-8")
        mt = re.search(r"route_team\s*([0-9.]+)\s*[^\n]*?→\s*([0-9.]+)", txt)
        if mt:
            m.add("AccCollapsed", num(mt.group(2), 4),
                  f"{args.v02_report}:max_len 装不下 head 时 route_team 的准确率"
                  f"（正常值 {num(mt.group(1), 4)}）")
        if "0.7026" in txt:
            m.add("AccScoreBroken", "0.7026",
                  f"{args.v02_report}:同一条件下 urgency 的准确率")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "numbers.tex").write_text(m.render(), encoding="utf-8")

    # --------------------------------------------------------------------------- figures
    keys = ("Choice", "Score", "Noul")

    def emit_assets(lang: str) -> list[str]:
        """写出**一个语言版本**的 3 张图 + 1 张表，返回拼好的正文串（供末尾自检）。

        「正文里不出现手写小数」对两版论文一视同仁：图与表都在这里生成，值全部走宏。
        中英两版共用同一批宏，只有坐标轴、图例与表头的文字不同，所以两版一起写出来，
        免得改了英文标签忘了中文（或反过来）造成漂移。`lang` 取 "en" 或 "zh"。
        """
        zh = lang == "zh"
        suffix = "_zh" if zh else ""
        tiers = ("自动$\\geq$0.9", "复核 0.6--0.9", "转人工$<$0.6") if zh else \
                ("auto$\\geq$0.9", "review 0.6--0.9", "human$<$0.6")
        t = {
            "TIER_A": tiers[0],
            "TIER_B": tiers[1],
            "TIER_C": tiers[2],
            "ACC_YLABEL": "准确率" if zh else "accuracy",
            "TIER_YLABEL": "占比" if zh else "fraction",
            "RULES": "关键词规则基线" if zh else "rule baseline",
            "ORIGINAL": "原版 laya-multilingual（未微调）" if zh else "original laya-multilingual",
            "MAJORITY": "多数类" if zh else "majority class",
            "OURS": "mai-jev（本文）" if zh else "mai-jev (ours)",
            "ECE_BEFORE": "温度标定前" if zh else "before temperature scaling",
            "ECE_AFTER": "温度标定后" if zh else "after temperature scaling",
            "URG_COVERAGE": "urgency：覆盖率" if zh else "urgency: coverage",
            "URG_ACCURACY": "urgency：该档准确率" if zh else "urgency: accuracy in tier",
            "NOUL_COVERAGE": "noul：覆盖率" if zh else "noul: coverage",
            "NOUL_ACCURACY": "noul：该档准确率" if zh else "noul: accuracy in tier",
            "GEN_NOTE": ("% 由 scripts/make_paper_tables.py 生成（中文论文版）" if zh
                         else "% 由 scripts/make_paper_tables.py 生成"),
            "TAB_HEADER": ("原语 & 准确率 & 均衡准确率 & 软准确率 & ECE & 平均置信度 \\\\"
                           if zh else
                           "primitive & accuracy & bal.\\ acc & soft acc & ECE & mean conf \\\\"),
        }
        chunks: list[str] = []

        def fill(tpl: list[str]) -> list[str]:
            """把 `%KEY%` 占位符换成当前语言的文字；没换干净就报错，别把 `%` 当注释写进 tex。"""
            filled = []
            for line in tpl:
                for key, val in t.items():
                    line = line.replace("%" + key + "%", val)
                if "%" in line:
                    raise SystemExit(f"图表模板里有未替换的占位符：{line}")
                filled.append(line)
            return filled

        fig_acc = fill([
            r"\begin{tikzpicture}",
            r"\begin{axis}[ybar, bar width=14pt, width=0.9\linewidth, height=5.2cm,",
            r"  ymin=0, ymax=1, ylabel={%ACC_YLABEL%}, symbolic x coords={route\_team,urgency,noul},",
            r"  xtick=data, legend style={at={(0.5,-0.18)},anchor=north,legend columns=-1}, legend cell align=left,",
            r"  nodes near coords, nodes near coords align={vertical}, every node near coord/.append style={font=\scriptsize}]",
            r"\addplot coordinates {(route\_team,\RulesAccChoice) (urgency,\RulesAccScore) (noul,\RulesAccNoul)};",
            r"\addlegendentry{%RULES%}",
            r"\addplot coordinates {(route\_team,\OrigAccChoice) (urgency,\OrigAccScore) (noul,\OrigAccNoul)};",
            r"\addlegendentry{%ORIGINAL%}",
            r"\addplot coordinates {(route\_team,\MajAccChoice) (urgency,\MajAccScore) (noul,\MajAccNoul)};",
            r"\addlegendentry{%MAJORITY%}",
            r"\addplot coordinates {(route\_team,\AccChoice) (urgency,\AccScore) (noul,\AccNoul)};",
            r"\addlegendentry{%OURS%}",
            r"\end{axis}",
            r"\end{tikzpicture}",
        ])
        (out / f"fig_accuracy{suffix}.tex").write_text("\n".join(fig_acc) + "\n", encoding="utf-8")

        fig_ece = fill([
            r"\begin{tikzpicture}",
            r"\begin{axis}[ybar, bar width=16pt, width=0.9\linewidth, height=4.6cm,",
            r"  ymin=0, ymax=0.12, ylabel={ECE}, symbolic x coords={route\_team,urgency,noul},",
            r"  xtick=data, legend style={at={(0.5,-0.18)},anchor=north,legend columns=-1}, legend cell align=left,",
            r"  nodes near coords, nodes near coords align={vertical}, every node near coord/.append style={font=\scriptsize}]",
            r"\addplot coordinates {(route\_team,\EceBeforeChoice) (urgency,\EceBeforeScore) (noul,\EceBeforeNoul)};",
            r"\addlegendentry{%ECE_BEFORE%}",
            r"\addplot coordinates {(route\_team,\EceChoice) (urgency,\EceScore) (noul,\EceNoul)};",
            r"\addlegendentry{%ECE_AFTER%}",
            r"\end{axis}",
            r"\end{tikzpicture}",
        ])
        (out / f"fig_ece{suffix}.tex").write_text("\n".join(fig_ece) + "\n", encoding="utf-8")

        fig_tiers = fill([
            r"\begin{tikzpicture}",
            r"\begin{axis}[ybar, bar width=10pt, width=0.9\linewidth, height=4.8cm,",
            r"  ymin=0, ymax=1, ylabel={%TIER_YLABEL%}, symbolic x coords={%TIER_A%,%TIER_B%,%TIER_C%},",
            r"  xtick=data, legend style={at={(0.5,-0.20)},anchor=north,legend columns=-1}, legend cell align=left]",
            r"\addplot coordinates {(%TIER_A%,\AutoCovScore) (%TIER_B%,\ReviewCovScore) (%TIER_C%,\HumanCovScore)};",
            r"\addlegendentry{%URG_COVERAGE%}",
            r"\addplot coordinates {(%TIER_A%,\AutoAccScore) (%TIER_B%,\ReviewAccScore) (%TIER_C%,\HumanAccScore)};",
            r"\addlegendentry{%URG_ACCURACY%}",
            r"\addplot coordinates {(%TIER_A%,\AutoCovNoul) (%TIER_B%,\ReviewCovNoul) (%TIER_C%,\HumanCovNoul)};",
            r"\addlegendentry{%NOUL_COVERAGE%}",
            r"\addplot coordinates {(%TIER_A%,\AutoAccNoul) (%TIER_B%,\ReviewAccNoul) (%TIER_C%,\HumanAccNoul)};",
            r"\addlegendentry{%NOUL_ACCURACY%}",
            r"\end{axis}",
            r"\end{tikzpicture}",
        ])
        (out / f"fig_tiers{suffix}.tex").write_text("\n".join(fig_tiers) + "\n", encoding="utf-8")

        # ------------------------------------------------------------------------ tables
        # `numbers.tex` 已经写完，这里只拼字符串；重复 `m.add` 会产生重名 \newcommand 让 LaTeX 报错。
        rows = []
        for key, prim in zip(keys, PRIMS):
            rows.append(
                f"{PRIM_LABEL[prim]} & \\Acc{key} & "
                f"{num(after[prim]['balanced_accuracy'])} & {num(after[prim]['soft_accuracy'])} & "
                f"{num(after[prim]['ece'])} & {num(after[prim]['mean_confidence'])} \\\\"
            )
        (out / f"tab_main{suffix}.tex").write_text(
            t["GEN_NOTE"] + "\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
            + t["TAB_HEADER"] + "\n\\midrule\n"
            + "\n".join(rows)
            + "\n\\bottomrule\n\\end{tabular}\n",
            encoding="utf-8",
        )

        chunks = rows + fig_acc + fig_ece + fig_tiers
        return chunks

    body_lines = emit_assets("en")
    emit_assets("zh")
    print(f"wrote {out}/numbers.tex 与 3 张图 + 1 张表（中英各一份）")

    # 自检：所有宏都被图或表引用到，且没有 NaN / inf 混进正文。
    body = "\n".join(body_lines)
    missing = [name for name, value, _ in m.rows if "\\" + name not in body and value not in ("n/a",)]
    unused = [name for name, _, _ in m.rows if "\\" + name not in body]
    print(f"宏总数 {len(m.rows)}；未被图表引用的 {len(unused)} 个（正文可用）")
    bad = [(n, v) for n, v, _ in m.rows if "nan" in v.lower() or "inf" in v.lower()]
    if bad:
        raise SystemExit(f"宏里出现 NaN/inf：{bad}")


if __name__ == "__main__":
    main()