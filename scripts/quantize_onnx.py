"""Quantize the exported laya ONNX graph and measure CPU latency + RSS.

Recommended recipe (see docs/research/quantization-and-deployment.md):
    quantize_dynamic(fp32, out, weight_type=QuantType.QInt8, per_channel=True)
i.e. per-tensor dynamic activations + per-channel int8 weights. `per_channel=True`
is what makes it usable: it lifts argmax agreement with eager fp32 from 0.44 to
0.67 overall and to 1.00 on rows with a clear fp32 margin. Static quantization is
rejected: QOperator fails outright on torch's fused Gather/Clip/Squeeze, and QDQ
collapses macro accuracy to 0.2536 on the #6 test split (vs 0.7823 fp32).

Measured on a desktop-class x86 CPU, 4 threads, batch=1, #6 test split (1029 records):
    fp32  1228.6 MB  p50 65.7 ms (max_len 512) / 264.0 ms (cap sweep)  RSS 1327 MB
    int8   308.6 MB  p50 65.7 ms (max_len 512) /  44.0 ms (max_len 96)  RSS  408 MB

Usage:
    python scripts/quantize_onnx.py --fp32 $WORK\\onnx\\laya_fp32.onnx \\
        --out-dir $WORK\\onnx

Dependencies: onnx, onnxruntime, numpy, ctypes (RSS, no extra package).
No optimum required.

Notes on this graph:
  * `quant_pre_process(..., skip_symbolic_shape=True)` is required -- ORT's
    symbolic Reshape handler asserts on the dynamic-shape graph torch 2.6 emits.
  * Static quantization is offered for comparison only. QOperator rewriter fails
    on torch's fused Gather/Clip/Squeeze with
    "Invalid model with unknown initializers/tensors.{'/Squeeze_1_output_0'}";
    QDQ succeeds but degrades accuracy further (measured, see the report).
"""
import argparse
import ctypes
import os
import time
from ctypes import wintypes

import numpy as np


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


_psapi = ctypes.WinDLL("psapi")
_k32 = ctypes.WinDLL("kernel32")


def rss_mb():
    c = _PMC()
    c.cb = ctypes.sizeof(_PMC)
    _psapi.GetProcessMemoryInfo(_k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize / 1048576


def make_feed(seq_len, markers=8, seed=1):
    rng = np.random.default_rng(seed)
    return {
        "input_ids": rng.integers(5, 1000, (1, seq_len)).astype(np.int64),
        "attention_mask": np.ones((1, seq_len), dtype=np.int64),
        "marker_pos": np.tile(np.arange(1, markers + 1), (1, 1)).astype(np.int64),
        "marker_mask": np.ones((1, markers), dtype=bool),
        "qtype": np.zeros((1,), dtype=np.int64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", default=os.path.join("onnx", "laya_fp32.onnx"))
    ap.add_argument("--out-dir", default="onnx")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--with-static", action="store_true",
                    help="also try static QDQ/QOperator, for the report's comparison table")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from onnxruntime.quantization import (
        CalibrationDataReader, QuantFormat, QuantType,
        quantize_dynamic, quantize_static,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prep = os.path.join(args.out_dir, "laya_prep.onnx")
    if not os.path.exists(prep):
        quant_pre_process(args.fp32, prep, skip_symbolic_shape=True)
    print("preprocessed: %.2f MB" % (os.path.getsize(prep) / 1048576))

    outs = {}
    recommended = os.path.join(args.out_dir, "laya_int8_perchannel.onnx")
    quantize_dynamic(prep, recommended, weight_type=QuantType.QInt8, per_channel=True)
    outs["int8_perchannel"] = recommended

    per_tensor = os.path.join(args.out_dir, "laya_int8_dynamic.onnx")
    if not os.path.exists(per_tensor):
        quantize_dynamic(prep, per_tensor, weight_type=QuantType.QInt8)
    outs["int8_per_tensor"] = per_tensor

    if args.with_static:
        class Reader(CalibrationDataReader):
            def __init__(self):
                self._it = iter([make_feed(s, seed=0) for s in (64, 128, 256, 512, 384, 192)])

            def get_next(self):
                return next(self._it, None)

        for fmt, name in ((QuantFormat.QDQ, "int8_static_qdq"), (QuantFormat.QOperator, "int8_static_qop")):
            path = os.path.join(args.out_dir, f"laya_{name}.onnx")
            try:
                quantize_static(prep, path, Reader(), quant_format=fmt, per_channel=True,
                                activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8)
                outs[name] = path
            except Exception as e:  # noqa: BLE001
                print("QUANT FAILED %s: %s" % (name, str(e)[:160]))

    import onnxruntime as ort
    print("\n=== batch=1, threads=%d, seq_len=%d, %d iters ===" % (args.threads, args.seq_len, args.iters))
    print(f"{'variant':<20}{'MB':>9}{'p50 ms':>9}{'p90 ms':>9}{'RSS MB':>9}")
    feed = make_feed(args.seq_len)
    for name, path in [("fp32", args.fp32)] + sorted(outs.items()):
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        for _ in range(5):
            sess.run(["logits", "act_logits"], feed)
        ts = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            sess.run(["logits", "act_logits"], feed)
            ts.append((time.perf_counter() - t0) * 1000)
        ts.sort()
        print(f"{name:<20}{os.path.getsize(path)/1048576:9.1f}{ts[len(ts)//2]:9.1f}"
              f"{ts[int(len(ts)*0.9)]:9.1f}{rss_mb():9.1f}")
        del sess


if __name__ == "__main__":
    main()