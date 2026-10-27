# 本机原规模训练运行记录

小规模筛查已按 [规则](SMALL_SCALE_GATE.md) 完成并[通过门槛](SMALL_SCALE_GATE_RESULT.md)。用户指定只用当前本机。正式协议为三种目标尺度、三个独立训练种子、原22维特征与图模型成对比较，共18个完整模型；当前仅启动首个模型，不能把其进度当作完整对照结果。

2026-10-22 已启动 `runs/formal_m3_graph_s2022`：`M_2_46_3`、N=210、每回合300代、总训练阈值405000环境步、收集12回合、SADN批量32、回放容量5000、两层 `action_attention`、`--fast-igd --fast-hv3`、CPU。首批3600步完成后曾自动暂停并核对原子快照：12回合、0更新、三训练函数均有覆盖、快照约13MB。随后从该快照以隐藏后台进程续跑；本次启动进程ID为77372，标准输出/错误日志在 `runs/formal_m3_graph_s2022.continue.stdout.log` 与 `.stderr.log`。当时已观察到第二批7200步、0更新。

读取最新进度：

```powershell
Get-Content runs\formal_m3_graph_s2022\metrics.jsonl -Tail 1
Get-Process -Id 77372 -ErrorAction SilentlyContinue
```

若进程因关机或故障停止，可在本目录使用**原训练配置加 `--resume`**，从最后一个已提交回合边界继续：

```powershell
& '..\Graph-Seq-MADAC\.conda-env\python.exe' -B train.py --name formal_m3_graph_s2022 --mode graph --seed 2022 --key M_2_46_3 --population-size 210 --budget-ratio 100 --total-steps 405000 --batch-episodes 32 --collect-episodes 12 --replay-episodes 5000 --graph-hidden-dim 64 --parallel-envs --fast-igd --fast-hv3 --threads 1 --resume
```

每批完成后 `resume.pt` 原子更新，`metrics.jsonl` 记录真实步数、更新数、任务覆盖和耗时。不要同时启动同一运行的第二个进程；如需续跑，先确认旧进程已经停止。当前尚无完成的原规模模型，也没有正式八函数对比结论。
