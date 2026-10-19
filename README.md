# FEMFusion

**FEMFusion** 是基于 [AgentFEM-Bracket-Elasticity](https://huggingface.co/datasets/HaomingLuo/AgentFEM-Bracket-Elasticity) 数据集的有限元多模态表征学习项目。当前版本（v1.0.0）包含三个独立的 Encoder：

- **CAD Encoder**：从 STEP / B-rep 几何构建面图，学习全局几何表示与 Face Tokens。
- **FEM Encoder**：从表面网格与位移、应力场学习全局物理表示与 Region Tokens。
- **Text Encoder**：从工程参数生成的模板文本学习表示；目前**不使用真实技术文档**。

三种 Encoder 均使用 `femfusion.common.EncoderOutput` 统一输出协议（128 维）。**跨模态对齐和特征融合尚未实现**，属于后续开发阶段。FEM Encoder 用于已有 FEM 场的表征学习，并非有限元求解器。

## 本地运行

推荐 Python 3.11；macOS / Linux：

```bash
./run.sh setup                 # 创建 .venv，安装依赖
./run.sh download              # 下载官方数据
./run.sh check                 # 检查安装、数据和接口
./run.sh train cad             # 训练 CAD（可换 fem / text / all）
./run.sh evaluate cad          # 评估 CAD（可换 fem / text / all）
```

安装完成后，也可用 `./run.sh full` 一次执行下载、全部训练及评估；只运行 CAD 和 FEM 时可分别调用两套命令。使用 Apple Silicon 或 NVIDIA GPU 时可在训练命令末尾加 `--device mps` 或 `--device cuda`（需本机 PyTorch 支持）。Windows 可用 `python run.py` 代替 `./run.sh`。

默认从 Hugging Face 下载预训练 Text backbone；如果当前只研究 CAD–FEM 对齐，可以暂时跳过 Text。完整训练可能需要较多时间和内存。

## 项目目录

```text
FEMFusion/
├── src/femfusion/
│   ├── common/                 # 三种 Encoder 的统一输出协议
│   └── encoders/               # cad/、fem/、text/ 模型、数据、训练和评估源码
├── configs/                    # 训练配置
├── scripts/                    # 数据下载、环境检查和旧结果迁移
├── data/                       # 数据/缓存（仓库内仅保留占位文件）
├── outputs/                    # 所有模型权重与训练、评估输出（不上传 GitHub）
│   ├── cad/{training,evaluation}/
│   ├── fem/{training,evaluation}/
│   └── text/{training,evaluation}/
├── run.py                      # 跨平台执行入口
├── run.sh                      # macOS/Linux 启动脚本
└── pyproject.toml
```

训练结果和测试数据默认不会被 Git 跟踪。重要的输出文件包括：

- CAD：`outputs/cad/training/best.pt`、`outputs/cad/evaluation/metrics.json`
- FEM：`outputs/fem/training/best.pt`、`outputs/fem/training/history.csv`、`outputs/fem/evaluation/report.json`
- Text：`outputs/text/training/best.pt`、`outputs/text/evaluation/evaluation_test.json`

若需沿用旧项目 `experiments/runs/fem_full/` 的训练成果（**只迁移本机文件，不包含在仓库中**），可执行：

```bash
python3 scripts/migrate_fem_run.py --source experiments/runs/fem_full
# 或者直接从已有的 experiments.zip 迁移：
python3 scripts/migrate_fem_run.py --source /path/to/experiments.zip
```

迁移后可直接运行 `./run.sh evaluate fem`（需已有官方数据）。如需在现有 `last.pt` 基础上继续训练，可将 `configs/encoders/fem.yaml` 的 `training.resume_checkpoint` 设置为 `outputs/fem/training/last.pt`，并根据需要提高总 `epochs`。**运行 `full` 会重新训练**，因此仅使用旧权重评估时应执行 `evaluate fem`。

## 发布与复现

项目不包含原始数据、预处理缓存、模型权重或历史输出。运行前使用 `./run.sh setup` 安装依赖；数据需从上述官方来源下载，训练/验证/测试沿用官方划分。可执行 `python3 -m unittest discover -s tests -v` 运行不依赖训练数据的基础检查。

上传 GitHub 时，建议新建名为 `FEMFusion` 的空仓库，在本地执行 `git init`、`git add .`、`git commit -m "Initial release: FEMFusion v1.0.0"`，再按 GitHub 页面提供的 `git remote add origin` 和 `git push` 命令上传。开源许可证（LICENSE）需由项目所有者自行选择。
