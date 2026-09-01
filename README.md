# halo_mw_LMC

本仓库实现 Zhu 等人的经验轨道叠加模型，并在显式
`(R,z,phi)` 网格上比较银河系恒星晕的密度和可选速度分布。

同一条执行管线支持两种权重模型：稳定基线使用星表 `halo_clean_N.txt` 中的
逐星 `w`；实验性 No-Fixed 模式在每一组候选势中，用三维目标密度反解非负
轨道权重，再以同一组权重计算速度似然。两种模式都由 AGAMA 积分轨道，速度
似然只使用 `r>=8 kpc` 的恒星。

## 目录结构

```text
halo_mw_lmc/
  core/             数值核心：网格、势、轨道、权重、密度、速度
  data/             星表和目标密度的文件格式适配
  workflows/        数据准备、优化、coverage 和报告工作流
  visualization/    Matplotlib 图形构建与批量输出
  configuration.py  严格 TOML 解析与类型化配置
  artifacts.py      可移植运行产物的读写与校验
  cli.py             八个日常运行生命周期命令及兼容入口
  inspection.py      从权威 artifacts 重建运行与报告状态
configs/
  recipes/          可复用科学模型配置
  runs/             数据路径、输出、随机种子和运行设置
.agents/skills/     仓库级科学设计与日常维护工作流
apps/results.py     只读 Marimo 结果应用
archive/            重构前代码快照，不属于当前生产路径
```

依赖方向和核心数组契约见
[`docs/architecture.md`](docs/architecture.md)。

## 项目设计与 Codex 工作流

科学目标、非主张、实验准入阶梯和当前开放决策统一记录在
[`docs/project_blueprint.md`](docs/project_blueprint.md)。根
`AGENTS.md` 只保留跨任务稳定的仓库契约，具体实验阈值和运行状态由对应
`docs/` 文档维护。

重大研究方向、实验设计或架构变化应显式调用
`$scientific-project-blueprint`，先完成方案评审再修改科学代码。日常代码、
配置、测试和图表修改由仓库级 `scientific-repo-maintainer` skill 按 blueprint
和现有模块边界检查，避免为单次调试复制新脚本。

## 配置

科学模型与单次运行分成两个文件：

- [`configs/recipes/zhu_2026_fixed_weight.toml`](configs/recipes/zhu_2026_fixed_weight.toml)：
  网格、拟合掩膜、速度项、轨道采样、参数边界；
- [`configs/recipes/zhu_2026_density_solved.toml`](configs/recipes/zhu_2026_density_solved.toml)：
  No-Fixed 的非负权重求解器、正则化和外层 objective；
- [`configs/runs/fix_weight.toml`](configs/runs/fix_weight.toml)：
  数据路径、run id、输出目录、迭代数、随机种子和报告设置。

所有相对路径都相对于声明它们的 TOML 文件解析，不依赖执行命令时所在的
shell 目录。未知或拼错的字段会直接报错。

No-Fixed 示例只需换一份 run 配置，命令不变：

```bash
halo-mw-lmc run configs/runs/density_solved.toml
```

该 recipe 使用 `unit_mass` 目标归一化、`lsq_linear` 非负最小二乘和 L2
正则化。它强制 `density_fit.normalization = "none"`，因为轨道权重已经决定
模型密度振幅，不能再引入第二个全局 density scale。详细统计定义见
[`docs/density_solved_weights.md`](docs/density_solved_weights.md)。

正式扫描前先在生产服务器执行一次完整星表、单势的 paper-best benchmark：

```bash
halo-mw-lmc run configs/runs/density_solved_benchmark.toml
```

该配置只执行一个 trial，并写入独立输出目录，不会修改 1000-trial 配置。数据
部署、依赖预检、计时和验收步骤见
[`docs/no_fixed_benchmark.md`](docs/no_fixed_benchmark.md)。

### 从 DESI 解析模型生成 density target

当前 No-Fixed run 使用 DESI year-1 K-giant 三轴分段幂律模型生成的相对密度：

```bash
conda run -n dp-jax python scripts/generate_synthetic_density.py \
  configs/synthetic_density/desi_year1_kgiants.toml
```

生成网格由引用的 No-Fixed recipe 定义，不通过长命令行参数传入。每个 cell
保存的是包含柱坐标 Jacobian 的体积平均，而不是中心点取值；目标 NPZ 同时携带
grid edges、模型参数、源文件 SHA-256、积分误差和明确的 synthetic fractional
error。生成文件位于 `data_for_model/synthetic/`，属于忽略的本地研究数据，且
不会覆盖已有文件。模型方程、未使用的拟合参数和坐标假设见
[`docs/desi_density_model.md`](docs/desi_density_model.md)。

历史 ASCII target 没有网格元数据，因此只允许用于原始的 `25×25×4`、
`R,z=0..50 kpc` 网格。修改网格时，`target_density` 必须指向包含
`target_density`、`target_error`、`r_edges`、`z_edges` 和 `phi_edges` 的
NPZ；边界不一致会在积分前报错。

## 日常运行 CLI

普通工作流重点记住四条命令：

```bash
halo-mw-lmc coverage configs/runs/fix_weight.toml
halo-mw-lmc run configs/runs/fix_weight.toml
halo-mw-lmc inspect runs/fix-weight
halo-mw-lmc report runs/fix-weight --overwrite
```

完整生命周期接口为：

```text
halo-mw-lmc run CONFIG
halo-mw-lmc optimize CONFIG
halo-mw-lmc evaluate CONFIG
halo-mw-lmc coverage CONFIG
halo-mw-lmc validate CONFIG [--json]
halo-mw-lmc preflight CONFIG [--stage run|optimize|evaluate|coverage] [--json]
halo-mw-lmc report RUN_DIR [--overwrite]
halo-mw-lmc inspect RUN_DIR [--json] [--save]
```

`run` 固定执行 validate、一次性 preflight/prepare、配置对应的 fixed evaluation
或 adaptive optimization、数值 artifact 校验、报告与 inspection。`evaluate`
只接受显式 `fixed_points`，不导入 skopt；`optimize` 只接受 adaptive 配置并独占
`ask/tell`。`validate` 不读取数据，`preflight` 只检查而不创建输出目录。
`coverage` 只读取 catalogue，输出仍解释为未校正 selection function 的原始
sampling density。

旧入口继续兼容：`python -m halo_mw_lmc CONFIG`、`-v`、`-c` 和 `-o`；其中
`-o` 按配置自动选择 fixed 或 adaptive 数值路径。它们不再是文档推荐入口。
所有 cold-start 数值命令和 coverage 都要求配置的输出目录尚不存在。

运行优化需要 NumPy、scikit-optimize 和单独安装的 AGAMA；No-Fixed 权重求解
还需要 SciPy；绘图需要 Matplotlib。Astropy 可提供更宽容的 ASCII 读取，但简单命名列文件有 NumPy
fallback。项目的可选依赖组定义在 `pyproject.toml`。不要在共享科研环境中
未经确认自行升级或安装依赖。

## 运行产物

默认 run 写入 `runs/fix-weight/`：

```text
resolved_config.json       完整展开的配置、网格、路径和 Git provenance
fixed_seed_weights.npz     固定权重、格点审计、target 与网格边界
weight_model_inputs.npz    No-Fixed 输入 target 与网格边界（替代上一项）
sample.dat                 每个 trial 的参数和标量评分
best/metadata.json         当前 best 的参数、iteration 和 objective
best/evaluation.npz        当前 best 的密度、速度和完整轨道权重快照
inspection.json            可重新计算的运行/solver/report 派生摘要
report/manifest.json       受管理报告的 generation、版本与文件清单
report/                    artifact-only 静态 PDF 与 summary.md
```

数值工作流不导入绘图。报告只读取 `resolved_config.json`、`sample.dat` 和
`best/`，不重新打开 catalogue/target，也不调用 AGAMA、optimizer 或权重求解器。
已有 `report/` 默认拒绝覆盖；`--overwrite` 先完整生成并校验临时目录，再以
可恢复的目录交换发布。历史 `figures/` 不删除，也不参与 report current 判定。
`inspection.json` 只是缓存；`inspect` 总是重新读取权威 numerical artifacts。

## Marimo 结果展示

[`apps/results.py`](apps/results.py) 只读取运行目录中的配置、样本、权重审计和
best snapshot：

```bash
marimo run apps/results.py
```

应用提供运行选择、收敛轨迹、逐 `phi` 密度比较、内层求解状态和权重集中度。缺少 snapshot
时只显示说明，不会自动补算。Marimo 是 `analysis` 可选依赖，本仓库不会在导入
核心包时加载它。`apps/` 和 `configs/` 是源码 checkout 中的研究工作区，不作为
wheel package data 分发；安装后的 CLI 可以读取用户自行保存的 TOML。

## 历史代码

重构前的顶层入口、轴对称/LMC/GPRy 实验、预处理脚本、`back/` 和 `funcs/`
完整保存在 [`archive/legacy_workflows/`](archive/legacy_workflows/)。归档保持旧
相对布局并记录已知缺失依赖和语法问题，但不承诺可直接运行，也不允许当前生产
路径反向导入。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall -q halo_mw_lmc apps/results.py
```

测试包括核心科学行为、旧密度轴序适配、TOML 严格校验、artifact 往返、固定
权重语义、稀疏轨道响应、非负密度权重求解、解析 tracer density 的柱坐标
体积积分、Marimo 只读边界和核心依赖方向。
