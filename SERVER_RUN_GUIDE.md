# Linux GPU 正式实验迁移与运行

本方案在服务器**重新开始**正式实验：3/5/7 目标各 3 个独立训练种子，每种子训练 `graph` 与原 22 特征 `stats`，共 18 套模型。每套训练阈值 405000 环境步、N=210、回放批量 32、容量 5000；每尺度全部模型完成后，八函数各用 10001–10030 这 30 个保留种子评估。主方法为两层 `action_attention`。结果留在 `runs/`。

## 迁移文件

只需要上传 `runs/GNN-Seq-server-source.tar`。其中包含完整 `GNN-Seq` 源码、冻结的 Seq 副本、`server_full_experiment.py`、`train.py`、`evaluate.py`、`compare.py` 和 `requirements-server.txt`；不包含本机 `runs/`，也不需要上传 Windows 虚拟环境。旁边的 `.sha256` 文件可用于传输校验。

本机 `formal_m3_graph_s2022` 是 CPU 运行，迁移到 GPU 时不作为正式配对实验的一部分。它在 7200 步时尚无 SADN 更新；本方案固定从头在服务器训练，避免两种设备/代码环境混用。

## 服务器准备

以下命令在服务器 Shell 中运行，将 `<user@host>` 和服务器目录替换为自己的值。先确认 `nvidia-smi`、Python 3.10、可用内存与存储空间；18 套模型的回放会占用大量磁盘。建议服务器目录位于容量充足的本地磁盘。

```bash
mkdir -p ~/seq-work
cd ~/seq-work
sha256sum -c GNN-Seq-server-source.tar.sha256
tar -xf GNN-Seq-server-source.tar
cd GNN-Seq
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -r requirements-server.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO CUDA")'
```

若最后一行显示 `False`，根据 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)选择 **Linux / Pip / Python / 服务器驱动适用的 CUDA** 安装命令，直到 `torch.cuda.is_available()` 为 `True`。不要直接安装冻结 Seq 仓库的旧版 `requirements.txt` 覆盖当前环境。

## 启动与进度

先检查固定计划，再运行正式脚本。脚本先各跑一次图模型和原特征模型的 GPU 小型训练；若失败立即停止。通过后顺序训练 18 套模型，最后评估和汇总。默认要求每套新模型开跑前至少有 30 GB 可用空间；它不会自动降低批量或改变正式协议。

```bash
python -B server_full_experiment.py --stage plan
mkdir -p runs
nohup .venv/bin/python -u server_full_experiment.py --stage all --device cuda > runs/server_full.log 2>&1 &
echo $! > runs/server_full.pid
```

查看总控日志和当前模型日志：

```bash
tail -f runs/server_full.log
tail -f runs/server_logs/formal_server_m3_graph_s2022.train.log
```

查看最近**已提交批次**的环境步数和 SADN 更新数：

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('runs/formal_server_m3_graph_s2022/metrics.jsonl')
if p.exists():
    x = json.loads(p.read_text().splitlines()[-1])
    print('steps', x['t_env'], '/ 405000; updates', x['training_steps'],
          '; episodes', x['episodes'])
else:
    print('No committed rollout yet')
PY
```

一轮采集和后续批量更新可能持续很久，`metrics.jsonl` 只在整轮提交后更新。每个运行保留 `resume.pt`；如果进程中断，在确认旧进程已停止后，用**相同虚拟环境、相同源码和相同完整命令**再启动 `server_full_experiment.py --stage all --device cuda`，它会跳过已完成模型并对未完成模型加 `--resume`。GPU 续训须保持可见 GPU 数量一致；不同 GPU/驱动上的逐位一致性未验证。

最终输出：`runs/server_formal_plan.json`、每模型的 `model/` 与 `formal_all_30.json`、每对模型的 `formal_server_m{m}_s{seed}_comparison.json`、总汇 `runs/server_formal_summary.json`。显著性解释应以独立**训练种子**为单位；30 个测试种子不等于 30 次独立训练。
