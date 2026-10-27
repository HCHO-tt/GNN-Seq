# GNN-Seq：图状态替换实验

本目录独立实现“MOEA/D 种群图 → 图编码 → 原 Seq 顺序决策与 SADN → 原 MOEA/D”的实验。现有项目没有被编辑。`vendor/Seq-MADAC` 是开始工作时本机 `Seq-MADAC-main` 的冻结副本，保留原许可证；它不是重新下载、核验过提交号的 GitHub 原始版本。

三训练种子的[小规模筛查结果](SMALL_SCALE_GATE_RESULT.md)达到预先设定的进入原规模实验门槛；[原规模验证协议](FORMAL_STABILITY_PROTOCOL.md)与[N=210 资源试跑](FORMAL_PREFLIGHT.md)已记录。当前没有完成原规模完整预算的训练，不应据此声称在原规模稳定优于原 Seq。

目前默认图策略加入[动作条件节点读取](ACTION_ATTENTION_EXPERIMENT.md)；已有实验检查点仍按旧纯池化图策略加载。初测证明节点信息能影响动作，但尚未证明最终 IGD 改善。

新增的[局部效果归因实验](LOCAL_EFFECT_STUDY.md)可通过 `--graph-policy effect_attention` 显式启用；它记录完整动作配置与源节点收益，在不改原 SADN 损失的前提下加一个辅助预测更新。该版本在固定回放上学到了结果相关性，但尚未在闭环八函数评估中超过默认 `action_attention`，因此保持为实验选项。

## 运行

在本目录打开 PowerShell。本机策略禁止执行 `.ps1` 脚本，因此直接调用已验证的 Python 入口，无需修改执行策略：

```powershell
# 使用现有环境，只读取它，不安装或修改包
$py = "..\Graph-Seq-MADAC\.conda-env\python.exe"
& $py -B -m pytest --basetemp=runs/test_manual
& $py -B verify_originals.py
& $py -B train.py --name my_smoke_graph --smoke
& $py -B train.py --name my_pooled_control --smoke --graph-policy pooled
& $py -B train.py --name my_effect_probe --smoke --graph-policy effect_attention --graph-layers 1
& $py -B probe_counterfactual.py --model runs/my_effect_probe/model --states 3 --output runs/my_effect_probe/counterfactual.json
& $py -B diagnose_effect.py --model runs/my_effect_probe/model --replay runs/my_effect_probe/replay --output runs/my_effect_probe/factual.json
& $py -B train.py --name my_smoke_stats --mode stats --smoke
& $py -B train.py --name my_pilot_graph --smoke --budget-ratio 10 --total-steps 400 --graph-hidden-dim 64
& $py -B train.py --name my_pilot_stats --mode stats --smoke --budget-ratio 10 --total-steps 400 --graph-hidden-dim 64
```

实测 Python 为 `..\Graph-Seq-MADAC\.conda-env\python.exe`，也可将 `$py` 指向其他已准备好的环境。实测环境版本见 `environment_versions.json`，上游依赖要求保留于 `vendor/Seq-MADAC/requirements.txt`。不依赖 torch-geometric。`run.ps1` 另提供批量操作封装，但本机被脚本执行策略阻止；上面的直接 Python 命令已实际运行通过。

直接运行：

```powershell
$py = "..\Graph-Seq-MADAC\.conda-env\python.exe"
& $py -B train.py --name my_graph --smoke
& $py -B train.py --name my_stats --mode stats --smoke
& $py -B evaluate.py --model runs/my_graph/model --output runs/my_graph/reevaluation.json --seeds 9001 9002
```

`--name` 已存在时拒绝覆盖。所有输出位于本目录的 `runs/`。评估输出也限制在本目录。脚本禁用 Python 字节码写入。

完整规模入口：

```powershell
& $py -B train.py --name graph_full --key M_2_46_3
& $py -B train.py --name stats_full --mode stats --key M_2_46_3
```

原 Seq 的三函数联合训练、八函数测试及配对比较命令见 [验证方案](VALIDATION_PROTOCOL.md)。`M_2_46_3`、`M_2_46_5`、`M_2_46_7` 是三套分别训练的固定规模策略；联合训练结束会生成 `evaluation_plan.json`，需要明确执行 `evaluate.py --suite all`，不会将混合环境的随机一次回合作为最终测试。`--checkpoint-interval` 可调整检查点间隔，每个检查点记录实际训练步数和各任务回合数。

默认 N=210、budget_ratio=100、405000 环境步、每批收集 12 回合、采样 32 回合、回放容量 5000 回合。每个环境步对应一代，不是一次目标函数评估。默认 CPU；可使用 `--device cuda`。本次未执行完整规模训练，完整规模需要另外评估计算时间与磁盘/内存预算。

## 保留与变化

| 部分 | 实现 |
|---|---|
| 四步顺序决策 | 原 `SeqMAC`；后续智能体继续接收本代前序动作的原始编号 |
| 智能体网络 | 原 `RNNAgent`，保持 `use_rnn=False`，两层 64 维 MLP |
| SADN | 直接加载原 `sadn_learner.py` 类；未重写损失、TD 目标、奖励标准化、VDN 或目标网络更新 |
| 价值网络 | 四个独立分支，各自有 GNN 和原 MLP；目标网络复制完整四分支 |
| MOEA/D | 原子代算子、变异、标量化、替换规则、外部档案、权重调整、IGD 奖励 |
| 图编码 | 策略共享一个 GNN；两层关系分开的均值消息传递，mean+max 池化，结合全局上下文输出 64 维状态 |
| 动作条件节点读取 | 默认 `action_attention`：每个智能体的候选动作分别查询种群节点，查询包含本代前序动作；读取结果加到原 Seq 智能体 Q 值。图层使用残差保留局部差异；`--graph-policy pooled` 保留旧图策略用于消融 |
| 局部效果实验 | `effect_attention` 额外保留每个 UID 过去 20 代的动作频率与真实源节点收益；仅对已执行的完整动作配置训练收益/成功率预测，并以可学习门控将预测收益加入动作 Q 值。`--effect-loss-weight 0` 关闭辅助损失用于消融 |
| 新增插桩 | 记录真实替换事件，按子问题 UID 管理历史；不改变奖励 |
| 回放 | 保存原始图张量到磁盘，训练时重新编码；保留容量与采样回合语义 |
| 采集器 | 本地同步逐环境执行，每个环境独立 Python/NumPy RNG；保留每轮收集结束后更新 `batch_size_run` 次的节奏 |

采集器不是原多进程 runner 的逐位复现。原 `SeqMAC` 对提前结束后的部分 batch 选择有形状限制，这里先取活动回合子批次再调用它。由于 `use_rnn=False`，每次初始化 hidden 不丢失循环记忆。

加载适配仅在新进程内补上原 MOEA/D 缺少的 `math`、`functools`，并绕开引用缺失 HAPPO 类的 `learners/__init__.py`。权重初始化提前到 reset 末尾，保证第一张图可见真实权重；原首步重复初始化受 guard 保护。

## 节点特征

一个节点表示一个**持续存在的权重子问题及其当前解**。解被替换时沿用子问题历史；权重调整中被保留的子问题沿用 UID，新权重生成新 UID。UID 只用于历史对齐与距离相等时的排序，不输入网络。

记实际目标数为 m、最大目标数为 M（默认 7）、初始种群为 P0：

- `b = min(F0)`；`s = max(max(F0)-min(F0), 1e-8*max(max(abs(F0)),1))`，按目标计算。
- 整个回合固定 b、s。目标 `f_norm=(f-b)/s`，理想点差距 `d=(f-z*)/s`；允许超出 [0,1]，不截断真实进展。
- 初始标量尺度 `s_g=max(median(positive g(P0)),1e-12)`，没有正值时取 1。

每节点 `4M+9` 维，M=7 时 37 维：

| 切片/顺序 | 含义 |
|---|---|
| `[0:M]` | 子问题权重；多余目标补零 |
| `[M:2M]` | 固定 P0 坐标下的目标值 |
| `[2M:3M]` | 到当前理想点的逐目标差距 |
| `[3M:4M]` | 有效目标 mask |
| `4M+0` | `log1p(g/s_g)`，g 使用后端真实标量化函数 |
| `4M+1` | 目标空间 kNN 平均距离 / sqrt(m) |
| `4M+2` | 权重邻居的决策空间平均距离 / sqrt(D)，变量按边界归一化 |
| `4M+3,4` | 接收改进 U 最近 5、20 代均值 |
| `4M+5` | 连续没有被成功替换的代数 / 本回合代数预算 |
| `4M+6` | 自己产生的子代成功率 S 最近 5 代均值 |
| `4M+7` | 自己产生的子代贡献 C 最近 5 代均值 |
| `4M+8` | 历史覆盖度 min(age,20)/20 |

本版限定无约束实数问题；不将不完整的约束处理特征直接推广到约束优化。

## 两种边与全局信息

- 权重空间欧氏 kNN，k=10，权重发生调整时重建。
- 归一化目标空间欧氏 kNN，k=5，每代重建。
- 排除自身，两种关系分别取双向并集并去重；两种关系有独立线性映射。自身信息通过独立映射进入。
- k 最多 N-1。网络采用稀疏边聚合，回放保存 kNN 索引；构建 kNN 的距离计算仍为 O(N²)。
- 图邻域与原动作选择的 15/20/25/30 个 MOEA/D 邻居是不同用途，彼此没有替换。

全局上下文 20 维：预算进度、1/m、1/D、权重调整剩余冷却比例、EP 占用比例（5 维）；上一代四个动作 one-hot（4+4+4+2）；上一代动作有效标记（1）。初始动作向量及标记全零。

## 历史如何计算

在原后端**实际接受替换时**，对源子问题 i 产生、替换目标子问题 j 的子代，使用同一时刻的理想点及 j 的权重计算：

```text
e_ij = clip((g_old - g_new)/(abs(g_old)+1e-12), 0, 1)
U_j  = 1 - exp(-sum_i e_ij)
C_i  = 1 - exp(-sum_j e_ij)
S_i  = 成功替换至少一个目标的子代数 / 自己产生的子代数
```

每代结束后更新长度 20 的环形历史。成功率每个子代最多记一次，贡献可累加其多个替换。无历史时填零，均值只按实际已有代数计算。新权重在生成当代没有参与产生子代，coverage 仍为 0。

权重更新按原列表对象身份对齐，并持有旧对象引用防止 Python 重用 id；不按近似坐标猜测身份。当前代的事件先产生，完成权重映射后再更新幸存子问题的历史。历史仅来自已发生的进化事件，不读取未来种群、参考 Pareto 集或 IGD 序列。

## 对照、资源和解释边界

`--mode stats` 使用原 22 特征、原 Seq 控制器和原四个 CentralV MLP，采用同一训练入口和更新次数。原高维 HV 估计使用随机采样，且原特征含参考集参与的 HV 计算；图状态不调用这些特征。因此即使种子相同，两模式也不应被声称具有完全相同的演化随机轨迹或完全相同的状态信息。插桩一致性测试使用相同 `wo_obs=True` 和固定动作，覆盖自适应权重开/关。

SADN 原有末步 bootstrap 与奖励标准化对 padding 的处理原样保留。本实现不借机更改这些算法细节；长短混合回合的统计行为应在正式对比中单独审视。

图编码器使用 gradient checkpoint，价值网络按时间分块编码。磁盘回放避免一次预分配 5000 个图回合，但不消除存储成本：N210、三目标的 301 个状态约每回合 13.23 MB，七目标的 701 个状态更大。三目标批量32回合原始输入约424 MB，另有激活与梯度开销。完整默认训练可能慢、占用较大，不能把小规模通过等同于完整规模资源验证。

`initial/` 保存同一次训练的零更新策略，用于前后对照；`model/` 保存最终策略、图编码器、四分支价值网络、优化器、额外目标网络与运行统计。每批完成后原子更新 `resume.pt`，保存环境、随机数、回放清单、策略与优化器状态；用原训练命令加 `--resume` 继续。N35 的图模型及原特征模型暂停/恢复与不间断运行逐项一致，N210 长期续跑仍需现场监测。训练日志保存动作序列、奖励、IGD、替换事件数量和权重更新次数，原始图在 `replay/` 可查。

## 文件入口

- `gnn_seq/environment.py`：后端插桩、动作掩码和 RNG 隔离。
- `gnn_seq/state.py`、`history.py`：节点、两种邻域与历史。
- `gnn_seq/model.py`：GNN、原 Seq 控制器接入、四分支图价值网络。
- `gnn_seq/experiment.py`：回放、采集、模型保存。
- `train.py`、`evaluate.py`、`compare.py`：联合训练、显式任务矩阵重载评估与同种子配对比较。
- `diagnose_learning.py`：比较不同检查点在同一批任务与种子上的参数变化、动作变化和最终 IGD；零更新策略的套件测试需在评估时显式传入 `--allow-untrained`。
- `tests/test_system.py`：演化一致性、历史对齐、置换不变性、梯度、回放与模型重载。
- `EXPERIMENT_REPORT.md`：本次实测结果与适用范围。
- `original_files.sha256.json`、`verify_originals.py`：既有文件完整性记录与核验。

建议正式实验再做“无历史”“只有权重边”“只有目标边”“无消息传递池化”消融，并使用多个训练种子和未见问题测试。这些消融尚未作为 CLI 选项实现，本次结果不构成它们的证据。

较长的小规模联合训练、八函数配对结果以及训练前后动作和 Q 值诊断见[初步验证报告](PRELIM_VALIDATION_REPORT.md)。
后续针对图感受野、特征变化、参数容量及稀疏邻域单因素试验的证据与改进顺序见[优化分析](OPTIMIZATION_ANALYSIS.md)。
