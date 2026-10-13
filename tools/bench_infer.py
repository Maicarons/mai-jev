"""mai-jev #8：推理侧成本实测（靶机口径：CPU-only、batch=1、单次三原语）。

测三件事，每件事都可独立复现：
  1. wall-clock  p50 / p95（warmup 之后，N 次重复，batch=1，一条 State 跑完三原语）
  2. 常驻内存   psutil RSS：加载后基线 + 峰值
  3. 瓶颈归属   --stages：分段计时，把一次推理拆成 tokenizer / encoder / heads / 后处理

三种精度模式：
  fp32  基线原样
  fp16  torch CPU 半精度（nn.Module.half()，注意 CPU fp16 很多算子没有高效 kernel）
  int8  torch.ao.quantization.quantize_dynamic(qint8)，覆盖 {Linear}

用法（每个配置一个独立进程，线程数必须用环境变量在 import torch 之前定死）：
  $env:PYTHONIOENCODING='utf-8'
  $env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
  python tools\\bench_infer.py ^
      --checkpoint $WORK\\runs\\smoke ^
      --data $WORK\\data\\v0.1_test.jsonl ^
      --mode fp32 --threads 4 --repeats 30 --warmup 5 --stages --out out.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 线程数必须在 torch 被 import 之前设好，否则 intra-op 池已经建好，改 env 无效。
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _set_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--mode", choices=["fp32", "fp16", "int8"], default="fp32")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--stages", action="store_true", help="额外做分段计时（tokenizer/encoder/head/post）")
    ap.add_argument(
        "--max-len",
        type=int,
        help="覆盖 agent cfg 的 max_len（序列长度是 encoder 的唯一可调杠杆，见 docs/research/quant-and-deploy.md）",
    )
    ap.add_argument("--out", help="把结果写成 json")
    args = ap.parse_args()

    _set_threads(args.threads)

    import psutil
    import torch

    torch.set_num_threads(args.threads)
    torch.set_grad_enabled(False)

    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    peak = {"rss": rss_before}

    def sample_peak() -> None:
        peak["rss"] = max(peak["rss"], proc.memory_info().rss)

    import laya

    from mai_jev.questions import QUESTIONS

    t0 = time.perf_counter()
    agent = laya.load(args.checkpoint, device="cpu")
    model = agent.model
    if args.mode == "fp16":
        # CPU fp16 只是为了量出「半精度在 x86 上到底帮不帮得上忙」，不是候选部署格式。
        model.half()
        agent.dtype = torch.float16
    elif args.mode == "int8":
        torch.backends.quantized.engine = "x86"
        # 只量化 encoder，不量化决策头。两个原因：
        #   1. 头只占 ~14M/322M 参数（见 param_bytes 对比），量化收益≈0；
        #   2. laya 的 head 是 nn.TransformerEncoder，量化后它的 `linear1.weight` 变成
        #      打包权重，torch.nn.modules.transformer 的 fast-path 检查
        #      （`x.device.type in _supported_device_type`，transformer.py:870-871）
        #      会拿到非 Tensor 对象而抛
        #      `AttributeError: 'function' object has no attribute 'device'`。
        model.encoder = torch.ao.quantization.quantize_dynamic(
            model.encoder, {torch.nn.Linear}, dtype=torch.qint8
        )
    load_s = time.perf_counter() - t0
    sample_peak()
    rss_loaded = proc.memory_info().rss
    if args.max_len:
        agent.cfg["max_len"] = args.max_len

    # 参数量与权重字节（量化后应该掉到 1/4 左右）
    n_params = sum(p.numel() for p in model.parameters())
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_qparams = sum(
        1 for mod in model.modules() if mod.__class__.__name__.startswith("Quantized")
    )

    rows: list[dict] = []
    for path in args.data:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    states = [r["state"] for r in rows]

    def one(state: str) -> None:
        payload = agent.predict(state, QUESTIONS)
        answers = payload.get("answers", payload)
        assert set(answers) == set(QUESTIONS), sorted(answers)

    for i in range(args.warmup):
        one(states[i % len(states)])
    sample_peak()

    lat_ms: list[float] = []
    for i in range(args.repeats):
        s = states[i % len(states)]
        t = time.perf_counter()
        one(s)
        lat_ms.append((time.perf_counter() - t) * 1000.0)
        sample_peak()

    lat_ms_sorted = sorted(lat_ms)
    p = lambda q: lat_ms_sorted[max(0, min(len(lat_ms_sorted) - 1, int(round(q * (len(lat_ms_sorted) - 1)))))]  # noqa: E731

    result = {
        "mode": args.mode,
        "threads": args.threads,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "load_seconds": round(load_s, 2),
        "params_M": round(n_params / 1e6, 2),
        "param_bytes_MB": round(n_bytes / 1024 / 1024, 1),
        "quantized_module_count": n_qparams,
        "rss_before_load_MB": round(rss_before / 1024 / 1024, 1),
        "rss_after_load_MB": round(rss_loaded / 1024 / 1024, 1),
        "rss_peak_MB": round(peak["rss"] / 1024 / 1024, 1),
        "latency_ms": {
            "p50": round(p(0.50), 2),
            "p95": round(p(0.95), 2),
            "mean": round(statistics.fmean(lat_ms), 2),
            "min": round(min(lat_ms), 2),
            "max": round(max(lat_ms), 2),
        },
    }

    if args.stages:
        result["stages"] = _stage_timing(agent, model, states, args)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")


def _stage_timing(agent, model, states, args) -> dict:
    """分段计时：把一次三原语推理拆成 tokenizer / encoder / heads / 后处理四段。

    tokenizer 单独调 `laya.common.encode_text` 计时；
    encoder / head 用 forward hook 计时（各自独立计时器，不用共享 start，避免互相覆盖）；
    后处理 = 总耗时 - tokenizer - encoder - head。
    """
    import laya.common as lc

    from mai_jev.questions import QUESTIONS

    acc = {"tokenizer": 0.0, "encoder": 0.0, "head": 0.0, "total": 0.0}
    handles = []

    def make_timer(key):
        box = {}

        def pre(mod, inp):
            box["t"] = time.perf_counter()

        def post(mod, inp, out):
            acc[key] += time.perf_counter() - box["t"]

        return pre, post

    # 注意：`DecisionModel.forward` 是 `for layer in self.head.layers: layer(...)`
    # —— 它**从不调用 `self.head(...)` 本身**，所以 hook 挂在 head 上永远不会触发。
    # 要计时必须挂到 head.layers 的每一层上。
    head = getattr(model, "head", None)
    if head is not None:
        for layer in head.layers:
            pre, post = make_timer("head")
            handles.append(layer.register_forward_pre_hook(pre))
            handles.append(layer.register_forward_hook(post))

    seq_lens: list[int] = []
    fwd_calls = {"encoder": 0}
    enc_pre, enc_post = make_timer("encoder")
    handles.append(model.encoder.register_forward_pre_hook(enc_pre))
    handles.append(model.encoder.register_forward_hook(enc_post))
    orig_enc_fwd = model.encoder.forward

    def counting_fwd(*a, **kw):
        fwd_calls["encoder"] += 1
        ids = kw.get("input_ids")
        if ids is None and a:
            ids = a[0]
        if ids is not None and hasattr(ids, "shape"):
            seq_lens.append(int(ids.shape[-1]))
        return orig_enc_fwd(*a, **kw)

    model.encoder.forward = counting_fwd

    n = max(args.warmup, 3)
    for i in range(n):
        s = states[i % len(states)]
        t = time.perf_counter()
        t1 = time.perf_counter()
        lc.encode_text(agent.tok, s, add_special_tokens=False)
        acc["tokenizer"] += time.perf_counter() - t1
        agent.predict(s, QUESTIONS)
        acc["total"] += time.perf_counter() - t

    for h in handles:
        h.remove()
    model.encoder.forward = orig_enc_fwd

    out = {k: round(v * 1000 / n, 2) for k, v in acc.items()}
    out["post_ms"] = round(
        max(0.0, out["total"] - out["tokenizer"] - out["encoder"] - out["head"]), 2
    )
    out["encoder_forward_calls_per_infer"] = round(fwd_calls["encoder"] / n, 2)
    out["seq_len_p50"] = sorted(seq_lens)[len(seq_lens) // 2] if seq_lens else None
    out["seq_len_max"] = max(seq_lens) if seq_lens else None
    out["n"] = n
    return out


if __name__ == "__main__":
    main()