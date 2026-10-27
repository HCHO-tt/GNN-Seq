# GNN-Seq 与原特征 Seq 的验证方案

目标是回答：固定原 Seq 顺序决策、SADN 与 MOEA/D 后，把 22 维手工状态换成种群图状态，是否改善同规模问题的最终 IGD。训练协议依据 [原仓库审计](SEQ_TRAINING_PROTOCOL_AUDIT.md)，所有改动和运行产物只在 `GNN-Seq` 中。

## 先验证训练与评估链路

以下命令在 `GNN-Seq` 目录运行。`--smoke` 让种群 N=35、每回合 5 代、回放容量 8，`--key` 则明确改成原 Seq 的三函数联合任务；两组都只训练 30 环境步，不能用于性能结论。

```powershell
$py = "..\Graph-Seq-MADAC\.conda-env\python.exe"
& $py -B train.py --name check_graph --mode graph --smoke --key M_2_46_5 --total-steps 30 --checkpoint-interval 10
& $py -B train.py --name check_stats --mode stats --smoke --key M_2_46_5 --total-steps 30 --checkpoint-interval 10
& $py -B evaluate.py --model runs/check_graph/model --suite all --seeds 9001 --output runs/check_graph/suite_all_9001.json
& $py -B evaluate.py --model runs/check_stats/model --suite all --seeds 9001 --output runs/check_stats/suite_all_9001.json
& $py -B compare.py --graph runs/check_graph/suite_all_9001.json --stats runs/check_stats/suite_all_9001.json --output runs/check_comparison.json
```

每组 `metrics.jsonl` 和检查点应明确显示 DTLZ2、WFG4、WFG6 三类的回合数、`t_env` 与 `training_steps`。`evaluate.py` 的八类任务为已见 DTLZ2/WFG4/WFG6，未见 DTLZ4/WFG5/WFG7/WFG8/WFG9；同一组评估种子在每个函数上重复使用。`compare.py` 逐个函数、逐个种子配对，拒绝训练规模、预算、更新次数和评估种子不一致的两份结果。输出中的负差值表示图状态的最终 IGD 更低。

## 正式代码协议实验

不必等完整训练结束才诊断。新训练运行在 `initial/` 保存同一次运行的零更新策略；训练后用 `--checkpoint-interval 10800`（三目标每批3600步，第一次 SADN 更新约在10800步）保存后续检查点。先用固定的少量评估种子比较 `initial`、10800步、若干中间步：逐函数最终 IGD、动作改变比例、奖励非零比例、梯度/参数是否改变。零更新的八函数评估需明确传 `evaluate.py --allow-untrained`；再用 `diagnose_learning.py --before ... --after ... --output ...` 检查同种子下的动作和 IGD 变化。若参数和梯度没有变化，检查更新与回放链路；若参数变但动作不变，检查动作 Q 值差距、输入敏感性和探索/奖励；若动作变而 IGD 不改善，再检查奖励目标与策略质量。监测中途检查点只用于诊断，不按测试集 IGD 选最佳模型。达到足够训练覆盖后，才做下述正式八函数对照。

首先固定 `M_2_46_3`：m=3、D=6、N=210、每回合 300 代、训练阈值 405000 环境步。两种状态用相同训练种子和训练超参数分别训练，并记录 `config.json`、`metrics.jsonl`、`checkpoints/*/metadata.json` 中的任务覆盖、实际环境步与 SADN 更新次数。检查点必须已产生更新；比较时预先确定同一实际步数的检查点（例如各自最终检查点），不要按测试 IGD 挑最优。训练命令如下：

```powershell
& $py -B train.py --name graph_m3_s2022 --mode graph --key M_2_46_3 --seed 2022 --checkpoint-interval 10800
& $py -B train.py --name stats_m3_s2022 --mode stats --key M_2_46_3 --seed 2022 --checkpoint-interval 10800
& $py -B evaluate.py --model runs/graph_m3_s2022/model --suite all --runs 30 --seed-start 9001 --output runs/graph_m3_s2022/all_30.json
& $py -B evaluate.py --model runs/stats_m3_s2022/model --suite all --runs 30 --seed-start 9001 --output runs/stats_m3_s2022/all_30.json
& $py -B compare.py --graph runs/graph_m3_s2022/all_30.json --stats runs/stats_m3_s2022/all_30.json --output runs/compare_m3_s2022.json
```

这对模型各用一个训练种子，`--runs 30` 是每个测试函数的 30 个环境随机种子，**不等于 30 次独立训练**。先观察 N=210 的运行时间、磁盘和显存/内存需求，再以预先固定的多个训练种子重复同一对照。之后分别训练 `M_2_46_5` 与 `M_2_46_7` 两套尺度，测试仍在各自目标数上运行八函数；不要把 3/5/7 目标混为同一模型。当前按用户选择固定每尺度 3 个训练种子，即 9 对、18 套模型；具体种子、评估和结论边界以[三种子配对协议](FORMAL_STABILITY_PROTOCOL.md)为准。三对方向一致只能支持本次三种子范围内的重复性观察，不能据此宣称总体稳定优势。

主指标是**最终种群 IGD**：先对每函数的 30 次测试取均值，再分别列出已见三函数与未见五函数的平均值。每个函数同时保留各次结果、标准差、配对差值及图状态胜出的次数。`best_igd` 只作诊断，不能替代最终种群 IGD。跨训练种子再汇总每对模型在同函数、同测试种子上的差；置信区间或显著性分析的独立单位应是训练种子，不能把 30 次测试当作 30 套独立策略。

## 解释边界

- 原仓库代码默认 D=m+3，所以三个目标数对应 D=6/8/10；论文表中的 6/9/12 不能直接作为这里的规模标签。未恢复论文原始启动参数时，称为**公开代码协议对照**。
- 当前图状态不计算原 22 特征中的高维随机 HV。两种模式在相同初始种子下并非逐位相同的演化随机轨迹。配对测试控制问题与种子，但不能单独证明差异只由图拓扑造成；如需拆分，后续应补充无历史、单边、无消息传递及参数量匹配等消融。
- 当前检查点支持重载评估，没有精确中断续训。完整 N=210、回放 5000 回合可能需要大量资源；小规模链路通过不表示完整预算可在当前设备完成。
