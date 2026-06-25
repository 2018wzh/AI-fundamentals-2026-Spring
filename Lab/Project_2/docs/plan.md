# Project Two 实验计划

## 1. 实验目标

本实验采用“分层比较 + 统一评测”的设计思路，将纯时序、时序+文本、时序+文本+图像三类任务放到同一条研究主线中，回答以下三个问题：

1. 在纯时序预测任务上，经典时序模型与基础模型谁更稳定。
2. 在加入新闻文本后，文本模态是否能为金融预测带来稳定增益。
3. 在进一步加入图像与事件校准后，多模态模型是否能够超过纯数值或弱多模态方法。

## 2. 总体实验框架

| 数据集 | 模态 | 推荐任务 | 主设置 | 参与模型 |
|---|---|---|---|---|
| Electricity | 纯时序 | 多变量长期预测 | `H=336, F=96`；扩展 `H=720, F=192` | DLinear、NLinear、PatchTST、TimesNet、Chronos-2、Aurora（unimodal） |
| FNSPID | 时序 + 文本 | 金融短中期预测 | `H=60, F=1`；扩展 `H=120, F=5` | DLinear、NLinear、PatchTST、TimesNet、Chronos-2、Aurora、Chronos-2-ECHO |
| OilETF-TimeMMD | 时序 + 文本 + 图像 | 金融多模态预测 | 直接使用现成 `H=60/F=1`、`H=120/F=5` | DLinear、NLinear、PatchTST、TimesNet、Chronos-2、Aurora、Chronos-2-ECHO |

## 3. 分层比较逻辑

- `Electricity` 用于比较纯数值时序预测能力。
- `FNSPID` 用于比较文本模态是否带来额外增益。
- `OilETF-TimeMMD` 用于比较文本、图像和协变量共同参与时的完整多模态增益。
- `Chronos-2-ECHO` 不建议强行放入 `Electricity` 主结果表，因为其优势主要来自事件、文本和图像校准；在该数据集上更适合作为去模态消融的补充。

## 4. 模型分组与比较方式

### 4.1 经典时序基线

- DLinear
- NLinear
- PatchTST
- TimesNet

这组模型统一只使用数值输入，采用相同的历史窗口、预测步长、数据划分和随机种子设置，作为 full-shot 训练基线。

### 4.2 基础时序模型

- Chronos-2
- Aurora（单模态或多模态）

推荐先进行 zero-shot 测试，作为“无需重训即可预测”的代表，再根据算力补充轻量 fine-tune 或 LoRA 微调实验。

### 4.3 多模态增强模型

- Chronos-2-ECHO

该模型重点在 `FNSPID` 和 `OilETF-TimeMMD` 上评测，并进行以下消融：

- `+text`
- `+image`
- `+text + image`

推荐先运行 `echo_only`，再补充 `lora` 设置。

## 5. 数据集与任务设置

### 5.1 Electricity

- 任务：多变量长期预测。
- 主设置：`H=336, F=96`。
- 扩展设置：`H=720, F=192`。
- 输入：仅保留数值时序。
- 划分：按时间顺序划分 train / val / test，避免未来信息泄露。

### 5.2 FNSPID

- 任务：金融短中期预测。
- 主设置：`H=60, F=1`。
- 扩展设置：`H=120, F=5`。
- 输入模态：
  - 数值：`OHLCV + return + technical indicators`
  - 文本：按日聚合新闻
  - 辅助统计：`sentiment`、`news_count`
- 数据规模建议：不直接使用全部 `4775` 只股票，优先选取 `50-100` 只新闻覆盖高、交易连续的股票组成实验子集。

### 5.3 OilETF-TimeMMD

- 任务：金融多模态预测。
- 设置：直接使用现成 `H=60/F=1` 与 `H=120/F=5` 样本。
- 输入模态：
  - 时序数值
  - 文本事件
  - 图像信号
  - 协变量
- 图像可直接使用已有 K 线相关图片资源。

## 6. 环境与依赖规划

建议将不同模型放在独立环境中，避免依赖冲突：

1. `Aurora` 单独环境，依赖以其仓库说明为准。
2. `Chronos-2 / Chronos-2-ECHO` 单独环境，统一放在 `chronos-forecasting` 仓库中运行。
3. 经典基线单独环境，建议使用 TSLib 或现有基线仓库统一训练。

## 7. 统一结果目录

所有实验输出统一保存到如下结构：

```text
results/{dataset}/{model}/{setting}/
```

每次运行至少保存以下文件：

- `metrics.csv`
- `predictions.parquet`
- `runtime.json`
- `plots/`

这样便于后续复现实验、画图和答辩展示。

## 8. 实施步骤

### 8.1 准备环境

1. 为 `Aurora`、`Chronos-2/Chronos-2-ECHO`、经典基线分别创建独立环境。
2. 固定 Python 版本、CUDA 版本、随机种子与核心依赖版本。
3. 记录每个环境的安装命令与依赖文件。

### 8.2 构建 Electricity 实验

1. 准备标准数值时序数据。
2. 按统一 train / val / test 划分处理。
3. 运行 DLinear、NLinear、PatchTST、TimesNet 的 full-shot 训练。
4. 运行 Chronos-2 与 Aurora 的 zero-shot 预测。
5. 如算力允许，再补充轻量 fine-tune。

### 8.3 构建 FNSPID 实验

1. 先筛选 `50-100` 只股票构成子集。
2. 将新闻按日聚合，并与股票日频数据对齐。
3. 构造 `OHLCV + return + technical indicators + sentiment/news_count` 特征。
4. 数值基线模型仅使用数值特征。
5. Chronos-2 使用数值 + 协变量。
6. Aurora 与 Chronos-2-ECHO 进一步使用文本输入。
7. 图像模态可由过去 `60` 或 `120` 天滚动 K 线图生成。

### 8.4 构建 OilETF-TimeMMD 实验

先在仓库中构建数据：

```powershell
cd D:\Workspace\OilETF-TimeMMD
python -m src.pipeline.build_oil_dataset --config configs/data_config.yaml
```

重点使用以下现成资源：

- `samples_H60_F1.parquet`
- `samples_H120_F5.parquet`
- `data/images/OilETF/*.png`

### 8.5 运行经典基线

- 所有经典模型使用相同数值输入、相同 `H/F`、相同数据划分和相同随机种子策略。
- 每个设置运行 `3` 个随机种子。
- 汇报均值和标准差。

### 8.6 运行 Chronos-2

- 先运行 zero-shot，作为基础模型对照。
- 在 `FNSPID` 与 `OilETF-TimeMMD` 上增加一组 LoRA 或轻量 fine-tune 实验。
- 充分利用其协变量输入能力。

### 8.7 运行 Aurora

- 在 `Electricity` 上进行 unimodal zero-shot。
- 在 `FNSPID` 与 `OilETF-TimeMMD` 上进行 multimodal 预测。

### 8.8 运行 Chronos-2-ECHO

- 重点在 `FNSPID` 与 `OilETF-TimeMMD` 上评测。
- 优先运行 `echo_only`。
- 后续补充 `lora` 微调。
- 完成三组消融：`+text`、`+image`、`+text+image`。

### 8.9 统一汇总结果

- 主表：按“数据集 × 设置 × 模型”组织结果。
- 附表：单独展示 zero-shot、fine-tune 和 multimodal ablation。
- 不将不同训练范式直接混在同一主表中。

## 9. 统一评测指标

### 9.1 点预测指标

- MAE
- RMSE
- MSE
- sMAPE

说明：金融收益率任务不建议将 `MAPE` 作为主指标。

### 9.2 概率预测指标

- Pinball Loss（`q10/q50/q90`）
- CRPS
- PICP
- PINAW

这组指标重点用于比较 Chronos-2、Aurora、Chronos-2-ECHO 的不确定性预测能力。

### 9.3 金融任务附加指标

- Directional Accuracy
- F1（up/down）
- Hit Rate@long

这些指标用于说明预测改善是否具有交易意义，而不仅是数值误差略有下降。

### 9.4 效率指标

- 参数量
- 单批推理时间
- 显存占用

### 9.5 统计显著性

- 对主指标进行 Diebold-Mariano test；如果实现受限，至少对窗口级误差进行配对检验。
- 避免只报告均值而不讨论波动性和显著性。

## 10. 可视化方案

建议统一产出以下图表：

- 预测曲线图
- 误差箱线图
- 分位区间覆盖图
- 按市场状态切片的对比图

针对金融数据，进一步按以下场景分组分析：

- 高波动日 / 普通日
- 有新闻日 / 无新闻日
- EIA 发布日 / 非 EIA 发布日

## 11. 报告主线

最终报告建议围绕以下三条主线展开：

1. 在纯时序场景中，经典模型与基础模型谁更稳。
2. 在加入文本后，`FNSPID` 上是否出现稳定增益。
3. 在加入图像与事件校准后，`OilETF-TimeMMD` 上 `Chronos-2-ECHO` 是否超过 `Chronos-2` 与 `Aurora`。

## 12. 预期产出

本项目最终应至少形成以下成果：

- 一套统一的实验配置与结果目录
- 三个数据集上的主结果表
- zero-shot / fine-tune / multimodal ablation 附表
- 指标统计与显著性检验结果
- 关键可视化图表
- 可直接写入课程报告的实验分析框架

## 13. 参考资料

- FNSPID 官方仓库：[https://github.com/Zdong104/FNSPID_Financial_News_Dataset](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)
- FNSPID 论文：[https://arxiv.org/abs/2402.06698](https://arxiv.org/abs/2402.06698)

