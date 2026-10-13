# Data

本目录存放 `scripts/build_dataset.py` 的输入/输出，全部数据文件被 `.gitignore` 忽略，只提交脚本与 `DATASETS.md`。

## 布局

```
data/
  raw/                      # 从上游下载的原始数据（build_dataset.py 的输入）
    mri/                    # MRI-decision-data-100k（MIT）104 个 jsonl
    zh-tw/                  # custom-service-ticket-zh-tw（CC BY 4.0）
    zh-decision-bench/      # zh-decision-bench（CC BY 4.0）仅评测
  adapted/                  # build_dataset.py --write 的输出
    mai_jev_train.jsonl     # laya 训练输入（含 questions + gold）
    mai_jev_calib.jsonl     # 温度标定专用（laya-train 不再使用；本文件保留）
    mai_jev_test.jsonl      # held-out 评测集
    build_report.json       # 统计报告（样本数/分布/token 长度/去重）
```

## 重建

```powershell
$env:PYTHONIOENCODING = 'utf-8'
python scripts\build_dataset.py `
    --raw $WORK\data\raw --out $WORK\data\adapted --write
```

原始数据如需重新下载，见 [DATASETS.md](../DATASETS.md) 的上游地址与许可。
