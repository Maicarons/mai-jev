"""mai-jev 部署侧推理：不依赖 torch / transformers / laya。

只依赖 `onnxruntime` + `tokenizers` + `numpy`。整个包的体积比训练栈小一个数量级，
而「常驻 ≤1GB / p50 < 50ms」这两条验收线只有在这种冷启动后的进程里测才有意义。

为什么要自己重建序列：laya 的 `build_head` / `build_sequence`（`laya/common.py:225` 与 `:156`）
决定了 `[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]` 这个布局，
训练与 ONNX 图都以它为准。laya 包能提供它，但部署端不该为了排一个序列去拖进 torch，
所以这里按源码逐行等价地重建，并用 `scripts/check_tokenizer_alignment.py` 式的对拍验证。

两条硬约束（见 docs/research/quantization-and-deployment.md §1.5 与 §7）：

1. **batch=1，且每条按自己的真实长度推理。** 导出的图在右 padding 的混合 batch 里
   会让短的 `qtype=2`（noul）行返回 NaN（340/1029 行），根因是 ModernBERT 的
   sliding-window attention 与 padding 在 torch.onnx 下的交互，未修。
2. **温度是「模型 + 推理后端」的耦合属性。** 换量化变体就必须重新拟合温度，
   所以 `calibration.json` 与 `.onnx` 同名同目录，不跨变体复用。

用法：

    from mai_jev.deploy import Predictor
    p = Predictor(r"export\\mai-jev-int8-ml96")
    print(p.predict("我的订单 TK-1234 付款后重复扣款，请尽快处理"))
    # -> {"route_team": {...}, "urgency": {...}, "needs_refund_or_escalation": {...}}

命令行：

    python -m mai_jev.deploy --model-dir export\\mai-jev-int8-ml96 --text "退款"
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List

import numpy as np

# 与 laya.train.QTYPES 一致（laya/train.py），也是 `mai_jev.questions` 里三种 type 的取值。
QTYPE = {"choice": 0, "score": 1, "noul": 2}

# laya 的 `Agent._to_internal` 把 noul 的渲染固定成 [false, true]（laya/common.py:127）。
_NOUL_FALSE = "false: no, the statement does not hold"
_NOUL_TRUE = "true: yes, the statement holds"

# 特殊 token 的字符串 -> id。`tokenizers.Tokenizer` 本身不暴露 `mask_token` 这类属性，
# 只能按字符串查表；候选名按「训练侧 transformers 实际用的那个」排前面（下面三个就是
# 这个 checkpoint 的真实取值：cls=<bos>=2, sep=<eos>=1, mask=<mask>=4）。
SPECIAL_TOKENS = {
    "cls": ("<bos>", "<s>", "[CLS]"),
    "sep": ("<eos>", "</s>", "[SEP]"),
    "mask": ("<mask>", "[MASK]"),
    "pad": ("<pad>", "[PAD]"),
}


def _special_id(tokenizer, name: str) -> int:
    for tok in SPECIAL_TOKENS[name]:
        tid = tokenizer.token_to_id(tok)
        if tid is not None:
            return int(tid)
    return -1

# 单个选项在 laya 的 build_head 里被截到 48 token（laya/common.py:248）。
MAX_TOKENS_PER_OPTION = 48
MAX_HEAD_INSTRUCTION = 8

# laya.train.QTYPE_NAMES（laya/common.py 的 temp_bucket 拼档名要用它）。
QTYPE_NAMES = ("choice", "score", "noul")


def _render_criterion(value: Any) -> str:
    """与 `laya.common.render_criterion` 等价：非字符串结构体渲染成单行 JSON。"""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict[str, Any]) -> List[str]:
    """与 `laya.common.render_options` 等价，选项顺序即 label-index 顺序。

    noul 的语义顺序恒为 [false, true]；score 渲染成 `level i: ...`。
    """
    qtype = q["type"]
    crit = q.get("criteria")
    if qtype == "choice":
        return [
            str(k) if v in (None, "") else "%s: %s" % (k, _render_criterion(v))
            for k, v in crit.items()
        ]
    if qtype == "score":
        return ["level %d: %s" % (i, _render_criterion(c)) for i, c in enumerate(crit)]
    crit = crit or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (_render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (_render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


class _Head:
    """问题半段：`[CLS] <type> instructions [SEP] [MASK] opt0 ... [SEP]`，以及每个选项的 marker 位置。

    逐行对应 `laya.common.build_head`（laya/common.py:225-276）：选项按 48 token 截断；
    预算不够时按 `max(4, (head_max_len - 16) // n)` 均分；instruction 再被截到剩余预算或至少 8 token。
    """

    def __init__(self, tokenizer, q: Dict[str, Any], head_max_len: int,
                 mask_id: int, cls_id: int, sep_id: int):
        self.q = q
        opts = render_options(q)
        ins = str(q.get("instructions") or "").replace(SPECIAL_TOKENS["mask"][0], " ")

        head_ids = tokenizer.encode("%s question: %s" % (q["type"], ins), add_special_tokens=False).ids
        opt_ids = [
            [mask_id] + tokenizer.encode(
                " " + opt.replace(SPECIAL_TOKENS["mask"][0], " "),
                add_special_tokens=False,
            ).ids[:MAX_TOKENS_PER_OPTION]
            for opt in opts
        ]

        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        if opt_budget < 16:
            per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, opt_budget)]

        ids = [cls_id] + head_ids + [sep_id]
        markers = []
        for opt in opt_ids:
            markers.append(len(ids))
            ids.extend(opt)
        ids.append(sep_id)
        self.ids = ids
        self.markers = markers
        self.options = opts


class Predictor:
    """一个导出目录 = 一个 `.onnx` + 一份同名 `calibration.json` + `questions.json` + tokenizer。

    目录约定（由 `scripts/build_export.py` 生成，不要手工拼）：

        mai-jev-int8-ml96/
            model.onnx              int8 动态 per-channel 单图
            calibration.json        该变体专属温度（`laya.calibrate.calibration_payload` 形状）
            questions.json          三原语定义（训练时冻结的那一份）
            tokenizer/tokenizer.json  XLM-R BPE，部署侧只需这一个文件
    """

    def __init__(self, model_dir: str, threads: int = 4):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.model_dir = model_dir
        onnx_path = os.path.join(model_dir, "model.onnx")
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                "%s not found at %r; run scripts/build_export.py first." % (onnx_path, model_dir))

        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir, "tokenizer", "tokenizer.json"))
        # tokenizer.json 自身带 truncation{max_length:48}，那是训练时截「单个选项」的残留。
        # HF 的 PreTrainedTokenizerFast 包装层不读它，而裸 `tokenizers.Tokenizer` 读，
        # 于是状态被静默砍到 48 token（49/60 条测试状态都受影响），序列跟训练分布对不上。
        # 这里显式关掉截断，长度只由 max_len 决定（与 `laya.common.encode_text` 一致）。
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        # 部署侧只用 token id，不还原字符串；但 head 的 `[MASK]`/`[CLS]`/`[SEP]` 必须是同一个值。
        tok = self.tokenizer
        self.mask_id = _special_id(tok, "mask")
        self.cls_id = _special_id(tok, "cls")
        self.sep_id = _special_id(tok, "sep")
        if min(self.mask_id, self.cls_id, self.sep_id) < 0:
            raise ValueError(
                "tokenizer.json is missing a mask/cls/sep token (looked for %s); see report 5.1"
                % {k: list(v) for k, v in SPECIAL_TOKENS.items() if k != "pad"})

        with open(os.path.join(model_dir, "questions.json"), encoding="utf-8") as fh:
            self.questions = json.load(fh)
        with open(os.path.join(model_dir, "calibration.json"), encoding="utf-8") as fh:
            calib = json.load(fh)
        self.temperature = [float(t) for t in calib.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {str(k): float(v) for k, v in (calib.get("temperature_by_options") or {}).items()}

        cfg_path = os.path.join(model_dir, "deploy_config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        self.max_len = int(cfg.get("max_len", 96))
        self.head_max_len = int(cfg.get("head_max_len", 256))

        # 变体的 max_len 就是它的产品身份：head 放不下时宁可截状态，也不截选项。
        self._heads = {
            qid: _Head(tok, q, self.head_max_len, self.mask_id, self.cls_id, self.sep_id)
            for qid, q in self.questions.items()
        }

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])

    # ── 单条推理 ──────────────────────────────────────────────────────────────

    def _feed(self, qid: str, text: str) -> dict:
        """batch=1、精确长度：这条是硬约束，不是优化。右 padding 的混合 batch 会让短 noul 行 NaN。"""
        head = self._heads[qid]
        text = str(text).replace(SPECIAL_TOKENS["mask"][0], " ")
        state_ids = self.tokenizer.encode(text, add_special_tokens=False).ids[: self.max_len]
        room = max(0, self.max_len - len(head.ids) - 1)
        ids = head.ids + state_ids[:room] + [self.sep_id]
        ids = ids[: self.max_len]
        markers = [m for m in head.markers if m < self.max_len]
        n = len(markers)
        return {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "marker_pos": np.array([markers], dtype=np.int64),
            "marker_mask": np.ones((1, n), dtype=bool),
            "qtype": np.array([QTYPE[self.questions[qid]["type"]]], dtype=np.int64),
        }

    def predict_question(self, qid: str, text: str) -> Dict[str, Any]:
        """单原语结果：选项概率 + 温度标定后的概率 + argmax 标签。"""
        feed = self._feed(qid, text)
        logits, _ = self.session.run(["logits", "act_logits"], feed)
        qtype = QTYPE[self.questions[qid]["type"]]
        raw = logits[0][: len(self._heads[qid].markers)].astype(float)
        scale = self.temperature_by_options.get(
            self._bucket(qtype, len(raw)), self.temperature[qtype])
        probs = _softmax(raw / max(scale, 1e-4))
        options = self._heads[qid].options
        best = int(np.argmax(probs))
        return {
            "question_id": qid,
            "type": self.questions[qid]["type"],
            "probs": probs.tolist(),
            "labels": [o.split(":", 1)[0] for o in options],
            "label": options[best].split(":", 1)[0],
            "confidence": float(probs[best]),
            "temperature": float(scale),
        }

    def predict(self, text: str) -> Dict[str, Any]:
        """一条工单跑完三原语。顺序与 `mai_jev.questions.QUESTION_ORDER` 一致。"""
        return {qid: self.predict_question(qid, text) for qid in self.questions}

    @staticmethod
    def _bucket(qtype: int, k: int) -> str:
        """`laya.common.temp_bucket`（laya/common.py:815-817）逐行等价。

        口径是**按候选数分档**，不是按 qtype 分支：k<=2 → "2"，<=5 → "3-5"，
        <=10 → "6-10"，否则 "11+"。档名拼在 qtype 名后面，所以 choice:3-5 / score:3-5 / noul:2。
        """
        size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
        return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def rss_mb() -> float:
    """当前进程 RSS（MB）。Windows 走 psapi，其他平台走 /proc；不额外引依赖。

    `GetProcessMemoryInfo` 的三个参数必须显式声明 argtypes：默认的 ctypes 转换会把
    `GetCurrentProcess()` 返回的伪句柄 -1 当成指针截断，调用静默失败（GetLastError=6）
    并把 WorkingSetSize 留在 0 上，于是 RSS 一律报 0.0。
    """
    import ctypes

    if os.name == "nt":
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        c = _PMC()
        c.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        fn.restype = wintypes.BOOL
        if not fn(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return c.WorkingSetSize / 1048576
        return c.WorkingSetSize / 1048576
    with open("/proc/self/statm", encoding="utf-8") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576


def main(argv: List[str] | None = None) -> int:
    import argparse
    import json as _json
    import time

    ap = argparse.ArgumentParser(description="mai-jev CLI：一次启动，一个进程，测常驻与 p50。")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--text", help="单条文本")
    ap.add_argument("--input", help="jsonl，每行一个 {\"state\": ...} 或纯文本")
    ap.add_argument("--max-len", type=int, help="覆盖导出目录里的 max_len")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="只输出 JSON")
    args = ap.parse_args(argv)

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(args.threads))

    t0 = time.perf_counter()
    predictor = Predictor(args.model_dir, threads=args.threads)
    if args.max_len:
        predictor.max_len = args.max_len
    load_s = time.perf_counter() - t0
    rss_after_load = rss_mb()

    if args.text:
        out = predictor.predict(args.text)
        if args.json:
            print(_json.dumps(out, ensure_ascii=False, indent=2))
        else:
            print("load %.2fs  RSS %.1f MB" % (load_s, rss_after_load))
            for qid, r in out.items():
                print("  %-26s %-14s conf=%.3f  %s"
                      % (qid, r["label"], r["confidence"],
                         " ".join("%s=%.2f" % (l, p) for l, p in zip(r["labels"], r["probs"]))))
        return 0

    if args.input:
        texts = []
        with open(args.input, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    texts.append(rec.get("state") if isinstance(rec, dict) else str(rec))
                except ValueError:
                    texts.append(line)
    else:
        texts = ["退款", "订单 TK-1 重复扣款，请尽快处理", "想咨询一下保修范围"] * max(1, args.warmup)

    lat = []
    rss_peak = rss_after_load
    for i in range(args.warmup + args.repeats):
        text = texts[i % len(texts)]
        t = time.perf_counter()
        for qid in predictor.questions:
            predictor.predict_question(qid, text)
        if i >= args.warmup:
            lat.append((time.perf_counter() - t) * 1000)
        rss_peak = max(rss_peak, rss_mb())
    lat.sort()
    report = {
        "model_dir": args.model_dir,
        "threads": args.threads,
        "max_len": predictor.max_len,
        "load_seconds": round(load_s, 2),
        "rss_after_load_MB": round(rss_after_load, 1),
        "rss_peak_MB": round(rss_peak, 1),
        "n": len(lat),
        "latency_ms": {
            "p50": round(lat[len(lat) // 2], 2),
            "p95": round(lat[min(len(lat) - 1, int(len(lat) * 0.95))], 2),
            "mean": round(sum(lat) / len(lat), 2),
        },
        "note": "p50/p95 是「一条工单跑完三原语」的总耗时，batch=1，warmup 已排除。",
    }
    print(_json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
