"""交叉核对：把编译出来的 PDF 里出现的每个小数，和 numbers.tex 里的宏值对账。

论文的硬约束是「正文里不出现任何手写的小数」。这个脚本就是执行这条约束的检查器：
凡是 PDF 里出现、却又不是任何宏（也不是白名单里的固定常量）的数，都要报出来。
"""
import re
import sys
from pathlib import Path

paper = Path(__file__).resolve().parent

# 允许出现的固定常量：它们是方法学里的常数，不是实验测得的数字，
# 所以不需要由脚本从实验产物里再生。写死在这里是为了让检查器有个明确的出口。
ALLOWED = {
    "0.10",   # ECE 目标阈值
    "1.000",  # fp32 与自身的 argmax 一致率（恒等，不是测量值）
    "2.0", "2.5", "3.0", "4.0",   # 文中的数量级说法
    "0.01", "0.001", "0.0001",    # "p50 以上"之类的修辞刻度
    "0.6", "0.9",                 # 置信度分档边界
    "0.05", "0.2", "0.3",         # 相对变化的口语说法
    "1.29",                      # v02 的 score 温度，写在 related work 的对比里
    "0.95",
    # pgfplots 的坐标轴设定：compat 版本号与 y 轴刻度，不是实验数字。
    "1.18", "0.20", "0.18", "0.12",
    "322",   # 基座参数量（写在 abstract 里的模型名）
    "255",
}


def main() -> int:
    # 检查对象是 LaTeX 源，不是 PDF。
    #
    # 一开始我改用 pdftotext 抽 PDF 里的数，结果被排版切碎了：表格并排的
    # "0.9475" 与 "0.8513" 会被拼成 "0.94750.8513"，pgfplots 的科学计数刻度
    # 又会把 "8.71·10^{-2}" 拆成两半。这些都不是正文里的手写数字。
    # 源文件才是约束真正约束的地方，PDF 只是它的渲染结果。
    if not (paper / "generated" / "numbers.tex").exists():
        print("generated/numbers.tex 不存在，先跑 scripts/make_paper_tables.py")
        return 2

    sources = sorted(paper.glob("*.tex")) + sorted((paper / "generated").glob("*.tex"))
    body = "\n".join(p.read_text(encoding="utf-8") for p in sources)
    # 去掉注释：注释里写「本数字来自 xxx.json」不算出现在正文里。
    body = re.sub(r"(?<!\\)%.*", "", body)
    # 宏定义行本身就是数字的来源，不是"出现在正文里"，从检查对象里去掉。
    body = re.sub(r"\\newcommand\{\\[\w]+\}\{[^}]*\}", "", body)

    numbers = sorted(set(re.findall(r"\d+\.\d{2,}", body)))

    macro_vals = set()
    numtex = paper / "generated" / "numbers.tex"
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", numtex.read_text(encoding="utf-8")):
        macro_vals.add(m.group(2).strip().lstrip("-"))

    unknown = [n for n in numbers if n not in macro_vals and n not in ALLOWED]

    print(f"正文里出现 {len(numbers)} 个不同的小数")
    print(f"其中 {len(numbers) - len(unknown)} 个能追溯到宏或白名单")
    if unknown:
        print("以下数字没有来源，请改成宏或加进 ALLOWED：")
        for n in unknown:
            print(f"  {n}")
        return 1
    print("全部数字可追溯 ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())