"""合成一小批中文客服工单，用来打通 build_dataset 的输入格式与端到端冒烟。

这是 **临时冒烟数据**，不是训练数据。真实数据来源见 #4 / docs/research/datasets-and-licenses.md。
合成规则是确定性的关键词规则（无随机），方便回归；只要 key 不变，输出就不变。

用法（Windows 控制台务必带 UTF-8，否则中文乱码）：
  $env:PYTHONIOENCODING='utf-8'
  python scripts\\make_smoke_data.py ^
      --n 300 --out $WORK\\data\\raw\\smoke.jsonl --write
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mai_jev.questions import NOUL_LABELS, ROUTE_TEAMS, URGENCY_LEVELS  # noqa: E402

PRODUCTS = ["路由器", "耳机", "扫地机器人", "电饭煲", "台灯", "机械键盘", "充电宝", "空气净化器"]
ISSUES = [
    ("无法连接网络，重启也没用", "技术支持", "4"),
    ("一边没声音，怀疑坏了", "技术支持", "3"),
    ("申请退货退款，东西不想要了", "退款处理", "2"),
    ("要赔付，等了三天没人处理", "投诉升级", "5"),
    ("发票抬头开错了，需要重开", "账务", "1"),
    ("自动续费扣了钱，要求退", "退款处理", "3"),
    ("投诉你们客服态度，要求升级处理", "投诉升级", "4"),
    ("想问问保修期多久", "售后", "1"),
    ("快递一直没到，改个地址", "售后", "2"),
    ("套餐变更后账单不对", "账务", "3"),
]
FILLERS = ["麻烦尽快", "谢谢", "已经等很久了", "再拖我要投诉了", "急！", "请处理一下", ""]


def synth(n: int, seed: int = 20261012) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        issue, team, urg = ISSUES[i % len(ISSUES)]
        product = PRODUCTS[rng.randrange(len(PRODUCTS))]
        filler = FILLERS[rng.randrange(len(FILLERS))]
        state = f"客户：我上周买的{product}{issue}。{filler}"
        split = "test" if i % 10 == 0 else ("calib" if i % 10 == 1 else "train")
        noul = "true" if team in {"退款处理", "投诉升级"} else "false"
        rows.append(
            {
                "id": f"smoke-{i:05d}",
                "state": state,
                "split": split,
                "labels": {
                    "route_team": team,
                    "urgency": urg,
                    "needs_refund_or_escalation": noul,
                },
                "source": "synthetic-smoke",
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    assert set(ROUTE_TEAMS) >= {t for _, t, _ in ISSUES}
    assert set(URGENCY_LEVELS) >= {u for _, _, u in ISSUES}
    assert set(NOUL_LABELS) == {"false", "true"}

    rows = synth(args.n)
    print(f"generated {len(rows)} rows")
    if args.write:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"wrote {target}")
    else:
        print("dry run: pass --write to emit the file")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    main()
