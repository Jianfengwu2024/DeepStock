# StockTransformer v1 实现与实验报告

## 1. 目标

StockTransformer 是 DeepStock Lab 中与 BNHMM 并列的新策略。它保留每日
价格-成交量分布中可重复的统计结构，但不直接把任意相邻直方柱送入神经网络。
整个流程严格拆成三个不可混合的阶段：

1. 在训练期发现随机守恒方程。
2. 只根据守恒流、密度梯度和随机残差分词、断句。
3. Transformer 只读取第二层输出的守恒 token。

现有 BNHMM 的代码、CLI、历史任务和测试结果均保留。Quant Platform 不加载该
模型；训练、推理、检查点和报告仍留在独立的 `deep_stock` 仓库及外部存储中。

## 2. 数据定义

- 数据集：半导体概念板块，100 个板块成员中 96 个有可用分钟线。
- 原始输入：每只股票自己的 1 分钟 OHLCV。
- 每日密度态：以前一交易日收盘价为对数价格零点，以 10 BP 为默认原始价格柱宽。
- 相对成交量：使用该股票前 10 个已完成交易日的平均分钟成交量归一化。
- 密度：当日各价格柱的相对成交量质量，归一化后总质量为 1。
- 标签：下一交易日收益，按信号时点以前 20 日波动率标准化；默认四状态为强空、
  弱空、弱多、强多。

股票之间从不构造状态转移。板块联合训练只共享守恒参数和模型参数，每条相邻
转移始终来自同一股票。

## 3. 第一层：随机守恒量发现

### 3.1 固定坐标

每个交易日原本具有不同长度的价格直方图。`DensityProjector` 先把归一化价格
坐标投影到固定的 `[-1, 1]` 网格，并再次归一化质量：

```text
sum_x rho_t(x) = 1
```

固定网格只用于比较密度场，不改变原始 10 BP 价格柱的构造规则。

### 3.2 连续性方程

由相邻两个交易日的密度变化反演零边界流：

```text
Delta rho_t(x) + div J_t(x) = 0
```

再在训练期拟合构成方程：

```text
J_t(x) = -D * grad rho_t(x) + v * rho_t(x) + xi_t(x)
E[xi_t(x)] = 0
```

其中 `D` 是扩散系数，`v` 是漂移系数，`xi` 是不能被扩散和漂移解释的随机流。
使用 ridge 正则拟合 `D` 和 `v`，并记录随机流噪声尺度。质量守恒是硬约束，
随机流均值为零是拟合后的统计约束。

实现：

- `src/bnhmm/stock_transformer/conservation.py`
- 输入：每日价格坐标和归一化成交量密度。
- 输出：只读 `ConservationLaw` 和 `ConservationTransition`。

## 4. 第二层：守恒分词与断句

分词器不使用固定相邻 2 柱或 3 柱。候选边界来自：

- 密度梯度变号，即密度盆地或峰值的边界。
- 守恒流方向反转。
- 标准化随机流残差突变。

边界按强度排序，并受最小词宽和每日最大 token 数约束。句子在以下位置开始：

- 第一枚 token。
- 净流方向发生反转。
- 随机残差超过阈值。

每个 token 固定包含 15 个统计特征：质量、质量变化、中心、宽度、峰值密度、
平均密度梯度、净流、绝对流、流一致性、随机流均值与 RMS、连续性残差、密度
熵、边界强度和句首标记。

实现：

- `src/bnhmm/stock_transformer/tokenization.py`
- 输入：第一层的 `ConservationTransition` 和 `ConservationLaw`。
- 输出：只读 `TokenizedState`。
- 分词器不能访问收益标签，也不能更新守恒方程。

## 5. 第三层：StockTransformer

Transformer 的输入仅为多日 `TokenizedState`：

- token 特征投影。
- 交易日位置嵌入。
- 日内 token 位置嵌入。
- 句首嵌入。
- 多层 Transformer Encoder 和注意力池化。
- 四状态分类头、连续收益均值/方差头、守恒辅助头。

训练损失为：

```text
L = L_return_nll
  + w_cls * L_cross_entropy
  + w_brier * L_brier
  + w_cons * (L_token_mass_change + L_daily_mass_zero)
```

质量变化是辅助目标，在主模型输入中被遮蔽，避免模型直接复制答案。每日预测的
token 质量变化总和被约束为零。MC Dropout 生成状态概率均值、aleatoric 方差和
epistemic 方差。

实现：

- `src/bnhmm/stock_transformer/model.py`
- `src/bnhmm/stock_transformer/training.py`

## 6. 三层隔离和防泄漏

`pipeline.py` 在每个 walk-forward 折中执行：

1. 只用调参开始日以前、且标签已经成熟的数据发现守恒方程。
2. 固定守恒方程后生成训练和调参 token，选择最佳 epoch。
3. 只用测试开始日以前的数据重新发现最终守恒方程。
4. 固定最终守恒方程后重新分词，并训练最终 Transformer。
5. 测试信号只在下一可交易时点生效。

训练、调参、外测三个时间区间不共享未来标签。检查点保存训练截止日、三层配置、
守恒方程和模型权重，便于复现。

## 7. DeepStock Lab

原 BNHMM Lab 已改名为 DeepStock Lab。左侧策略切换包含：

- `BNHMM`：保留原相空间、隐状态和贝叶斯转移配置。
- `StockTransformer`：显示守恒发现、守恒分词/断句和 Transformer 三层配置。

StockTransformer 当前强制使用板块联合训练。界面会显示数据集、96 只可用股票、
GPU 总显存与任务峰值显存；历史任务显示所属策略，结果页显示已发现的守恒系数和
token 诊断。

接口：

```text
POST /api/jobs
strategy = "stock-transformer"
```

CLI：

```bash
python -m bnhmm.cli stock-transformer-walk-forward \
  --duckdb-path /home/share/data/sectors/semiconductor.duckdb \
  --symbols <sector-symbols> --dataset-id semiconductor \
  --start 2023-01-01 --end 2024-08-01 \
  --output-dir <external-storage> --device cuda:0
```

## 8. 自动化测试

新增测试覆盖：

- 守恒质量、均值为零随机流、可变直方图投影和禁止跨股票转移。
- 非固定词宽、完整覆盖、最大 token 数和句子编号。
- Transformer 前向、MC 后验形状和守恒辅助损失。
- 双股票小板块的完整 walk-forward 和三层产物。
- Web 请求校验及独立 CLI 命令构造。

本地和 145 服务器均执行完整测试集：73 项通过。Python 编译、JavaScript 语法、
Git diff 格式检查均通过。已知的 PyTorch `norm_first` nested tensor 警告不影响
计算正确性。

开发期间 TriadMind watch 持续同步 Python/JavaScript 拓扑。最终
`triadmind verify --strict --json` 通过：32 个拓扑节点、13 个顶点、25 个运行时
节点、7 条运行时边全部一致；ghost、空顶点、尺度混合和焦点闭合违规均为 0。

## 9. 半导体板块实验

实验名称：`stocktransformer-semiconductor-smoke-20260815`

配置：96 只半导体股票，2023-01-01 至 2024-08-01，200 个最少训练日、40 个
调参日、30 个外测日、1 折、3 个最大 epoch、5 日记忆、64 维模型、3 层、4 个
注意力头、64 点守恒网格、每日最多 16 token。

实际运行时间约 3 分 14 秒，GPU 峰值显存 382 MiB。96 只可用股票被读取，其中
80 只在该时间窗口内具有足够历史并产生外测预测，共 2295 条。

发现的守恒参数：

| 指标 | 数值 |
| --- | ---: |
| 扩散系数 `D` | 0.00006783 |
| 漂移系数 `v` | 0.04358013 |
| 构成方程残差偏置 | -0.00029068 |
| 随机流噪声尺度 | 0.21546033 |
| 训练期同股票转移数 | 14107 |
| 连续性重构 RMSE | 1.90e-17 |

连续性 RMSE 接近机器精度是因为流由质量变化和零边界条件反演，表示离散质量守恒
实现正确，不应解释成价格可预测性。可预测部分应由扩散/漂移解释量和剩余随机流
的外测表现判断。

分词诊断：

| 指标 | 数值 |
| --- | ---: |
| 日均 token | 4.3424 |
| 90% 分位 token | 7 |
| 日均句子 | 3.3953 |

训练阶段的守恒辅助损失从 0.1481 降至 0.0675，验证守恒损失从 0.0864 降至
0.0518，说明模型确实学会了质量变化约束。最佳验证总损失出现在第 2 个 epoch，
第 3 个 epoch 已开始恶化。

严格样本外结果：

| 指标 | StockTransformer | 基线/说明 |
| --- | ---: | ---: |
| 命中率 | 28.93% | 多数类 29.72% |
| 均衡命中率 | 24.76% | 四分类随机 25.00% |
| Brier | 0.7575 | 训练期先验 0.7534 |
| Log Loss | 1.4042 | 训练期先验 1.3930 |
| 对数信息增益 | -0.0112 | 需大于 0 |
| 收益相关性 | -0.0630 | 需大于 0 |
| 收益 RMSE | 2.6245% | 零收益 2.4323% |
| 策略收益 | -0.89% | 等权基准 -13.11% |
| 最大回撤 | 0.93% | 平均绝对仓位仅 4.01% |

预测类别高度集中在弱空：四类预测数为 `[29, 2081, 184, 1]`。低回撤主要来自
平均仓位只有 4.01%、持仓日比例只有 8.06%，不是预测优势。发布门槛的分类、
概率、收益误差和相关性检查均未通过，因此实验状态为 `research-only`。

这次结果证明三层隔离、守恒分词、GPU 训练、样本外评估和 UI 链路可以完整运行；
它没有证明 StockTransformer v1 已获得可交易信号。

## 10. 当前结论和限制

StockTransformer v1 已验证三层工程链路和严格样本外执行，但一次小规模 smoke
实验不能证明可交易性。后续应优先比较：

- 守恒 token 与原 BNHMM 价格通道在完全相同窗口上的增量信息。
- 不同板块的扩散、漂移和随机流尺度是否稳定。
- 分词边界阈值、每日 token 上限和记忆长度的消融实验。
- 守恒辅助权重是否改善外测 Brier、Log Loss 和收益相关性，而非只降低训练损失。
- 多折和跨市场阶段验证；任何未通过发布门槛的版本只保留为研究结果。

模型训练仍只在独立服务器仓库中进行。Quant Platform 只能读取以后正式发布的
预计算信号，不能安装 PyTorch、加载权重或在回测时触发训练。
