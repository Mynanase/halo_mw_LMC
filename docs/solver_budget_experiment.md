# 权重求解提速实验：分步骤 Agent 执行方案

状态：2026-09-10 计划发布，尚未实现或执行本方案的新 runner。
代码起点为 `ec380dd01b6ca13044b8712b49c6ebab9b5eac2e`。
现有修改和历史证据见 [状态摘要](refactor_and_experiment_status.md)。
本文件是逐步骤执行契约：用户要求“执行步骤 N”时，只执行该步骤，
提供验收回执后停止，不自动进入下一昂贵阶段。

## 目标与不变量

科学问题不是能否逐个恢复轨道权重，而是：较少的求解预算能否保持预测、
joint objective 和势参数排序，并把单势完整评估控制在 **600 秒以内**。
先做三个势点、两个方法的六个逻辑 case，再依据门槛扩大。

固定以下内容，不在本实验中做联合调参：

- `density_solved`、density-only NNLS 内层、`density_velocity` 外层；
  `lambda=1e-6`，密度 scale 为一，不事后归一化权重。
- 使用 stage-1 screen 的数据、势、网格、掩膜、unit-mass target 和 likelihood；
  密度 `8 <= r < 40 kpc`、`|z| >= 2 kpc`，速度 201 bins。
- 10 个轨道周期、每轨道 1000 samples、坐标三位小数、seed 0。
- 同一生产环境、输入文件、干净 Git commit 和线程设置；不同方法同题比较。
- 每次权重求解独立 cold start。不把冻结轨道缓存解释为允许优化器 warm-start，
  不注入历史 GP 观测，不修改正式接受策略、公式或 artifact schema。
- 不覆盖已有输出、不追加已有 `sample.dat`、不自动安装依赖或停止已有任务。

三个 pilot 点按以下顺序固定；标签中的 iteration 从零开始：

| 名称 | 来源 | qhalo | phalo | rho0 | rho0_plus_2logrs | gamma |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `paper_best` | 论文锚点 / s01:0 | 0.920 | 0.800 | 6.200 | 9.890 | 1.000 |
| `accepted_best` | s09:2 | 0.985 | 0.714 | 6.099 | 9.829 | 0.751 |
| `capped_candidate` | s07:2 | 1.071 | 0.738 | 5.687 | 9.541 | 1.214 |

## 步骤 0：只读确认源码、历史证据和生产环境

任务：读取 `AGENTS.md`、blueprint、architecture、权重模型文档、本方案和
本地 runtime notes（若存在）。记录 Git 状态、数据哈希、依赖、线程池、
CPU/内存及负载；先确认服务器是否已经在运行 stage-2 或其他重任务。
不要把历史 dirty run 当成严格数值基线。

在对应仓库根目录执行以下只读命令；生产命令必须在生产服务器执行：

```bash
git status --short --branch
git log -8 --oneline
conda env list
conda run -n halo_lmc python -c "import sys, numpy, scipy; print(sys.version); print(numpy.__version__, scipy.__version__); numpy.show_config()"
PYTHONPATH="$PWD/Agama-master${PYTHONPATH:+:$PYTHONPATH}" conda run -n halo_lmc python -c "import agama; print(agama.__file__)"
ps -eo pid,etime,pcpu,pmem,args
```

额外记录实际加载的 BLAS/OpenMP 库与运行时线程数；若已有 `threadpoolctl`
可以调用，不为采集信息安装包。输入哈希必须按实际 resolved paths 读取，
不能只复制旧文档中的哈希。

验收：来源、输入、生产依赖、资源和已有任务状态明确；缺失则提交缺失清单。
不积分、不求解、不启动 GP。实现步骤可在本地继续，但生产步骤必须等依赖和
资源确认。交付 `step0` 回执及忽略目录中的环境/来源清单。

## 步骤 1：实现最小共享评估边界与薄 runner

本节列出的新文件和命令接口是**待实现接口**，不是当前已存在的功能。

需要修改：

1. 在 `halo_mw_lmc/workflows/evaluation.py` 增加
   `evaluate_orbit_library(library, prepared, *, response=None) -> ModelEvaluation`。
   现有 `evaluate_prepared_model(parameters, prepared)` API 不变：建势、积分后
   调用共享评分路径。移动现有积分后的操作，不重写公式、改变运算顺序或 dtype。
   外部提供的 response 在入口校验网格、列映射、有限样本计数及来源。
2. 新增 `halo_mw_lmc/workflows/solver_budget.py`，承载本实验顺序执行与持久化。
   新增薄入口 `scripts/benchmark_density_solved_r8_40_solver_budget.py`，支持
   `CONFIG --phase preflight|prepare|parity|pilot|budget|repeat|holdout-prepare|holdout|confirm`，
   `parity` 另接收 `--baseline-ref GIT_REF`。
3. 新增只读比较入口 `scripts/compare_density_solved_r8_40_solver_budget.py CONFIG --phase PHASE`。
   它只能读取保存的数组和 metadata；缺失证据时报错，不能调用 AGAMA 补算。
4. 新增 `configs/benchmarks/r8_40_solver_budget.toml`，显式保存点表、方法、
   phase 顺序、重复数、线程、超时、路径和验收阈值。参考
   `configs/runs/density_solved_r8_40_stage1_screen_shard01.toml` 的科学与数据配置，
   但不能继承其四点调度。每个 case 的 resolved config 必须记录实际坐标和
   solver override，不得误存基础 run 的完整旧 schedule。
5. 默认输出根为 `.agent-local/benchmarks/r8_40_solver_budget_v1/`，按 phase 和
   case 分目录。已有 case 默认拒绝覆盖；合法重试使用新的 attempt 标识并保存来源。
   原有 paper-best 三 backend benchmark 和比较器继续独立工作，不改其定义。

代码保持线性研究脚本风格；不新建插件系统、通用任务 DAG 或缓存数据库。
共享评估边界有生产评估和实验评估两个真实消费者，其余一次性步骤直接展开。

增加 `tests/test_solver_budget.py`：覆盖旧/新评分路径的小数组一致性、同轨道库
多方法只积分一次、仅 solver 设置可变、raw/正式 objective 区分、输出冲突、
输入与缓存不匹配、缺失/非有限数组、比较器不导入或调用 AGAMA。

```bash
conda run -n dp-jax python -m unittest discover -s tests -p 'test_pipeline.py' -v
conda run -n dp-jax python -m unittest discover -s tests -p 'test_objectives.py' -v
conda run -n dp-jax python -m unittest discover -s tests -p 'test_solver_budget.py' -v
conda run -n dp-jax python -m compileall -q halo_mw_lmc scripts tests
git diff --check
```

验收：现有 API/结果保持不变，薄入口和配置可预检，小数组测试通过。
单独提交实现；不在此步骤运行真实星表。

## 步骤 2：补齐产物、计时和自动门槛

### 轨道缓存与每个 case 的证据

`prepare` 对每个点只积分一次，冻结 phase space、time、seed index、成功 seed
映射、有限样本数、网格和 CSR response。使用 NPY/NPZ，不使用 pickle。
保存输入文件与缓存 SHA-256、坐标、代码版本、依赖/线程、建势/积分/response
耗时。比较时必须验证缓存与当前输入、配置、轨道映射匹配。

每次求解用新进程从零开始，读取同一点的冻结库；逐 case 串行执行，默认一个
worker，禁止内部自动批量并发。每个 case 保存：

```text
case.json
resolved_config.json
best/metadata.json
best/evaluation.npz
stdout.log
stderr.log
time-v.txt
```

复用现有 best snapshot 写入边界和 schema。实验 case 不伪装成常规 run，
不追加正式 sample，不让常规 run discovery 误识别它们。
保存全精度的权重、密度、三个速度分量预测与评分所需观测占用数；不能只存
三位小数的 sample 行。

`case.json` 至少包含：phase/point/method/repeat/attempt、坐标、输入和缓存哈希、
problem fingerprint、实际 solver 设置、状态/message/iterations/KKT、raw J、
正式 selected J、密度与速度分项、内层目标与正则项、轨道计数、权重摘要、
各段 wall time、超时/异常类型、源码和环境 provenance。
fingerprint 的相等约束适用于**同一点不同方法**，不要求不同势点指纹相同。

### 资源和时间定义

生产阶段使用相同环境变量，且必须在 Python 启动前设置：

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export OMP_NUM_THREADS=16
export PYTHONPATH="$PWD/Agama-master${PYTHONPATH:+:$PYTHONPATH}"
```

默认每个数值 case 超时 3600 秒，记录 timeout 并停止对应进程组，不能留下孤儿
求解进程或把缺失时间/目标记成零。用 GNU `/usr/bin/time -v` 隔离记录 peak RSS；
预检确认工具可用，不能把 macOS BSD time 当作同一接口。
高精度参考允许超过 600 秒，但仍受 case 超时和可用内存限制。

冻结库下估算搜索成本为该点共享的建势 + 积分 + response 构建时间，加本方法
solve + score 时间。缓存读取、写盘和初始数据准备分别报告；不能把 solve-only
时间叫完整评估时间。最终 600 秒门槛由无轨道缓存的实际评估确认，排除画图。

### 预先固定的数值门槛

以下是本轮预设的工程门槛，不是已经校准的天体物理误差；写入 TOML，禁止看过
结果后悄悄放宽。`J_ref` 始终是合格参考解在同一外层定义下的分数。

| 检查 | 门槛 |
| --- | --- |
| 高精度参考资格 | backend 成功，normalized KKT `<= 1e-8`，数组有限、权重非负且总和为正 |
| 内层目标 sanity check | 参考内层 F 不得高于其他可行解 F 超过 `1e-8 * max(1, abs(F_ref))` |
| 输入和轨道可比 | 同点输入、缓存、fingerprint、映射一致；零失败轨道 |
| 单点 joint objective | `abs(J - J_ref) <= 1` |
| 分项误差 | `abs(chi2/2 - chi2_ref/2) <= 1` 且 `abs((-log L) - (-log L_ref)) <= 1` |
| 点对差值 | 任意比较点对 `abs((J_i-J_j) - (J_ref_i-J_ref_j)) <= 1` |
| 排序 | 参考差值绝对值 `> 2` 不许翻转；`<= 2` 标记近似并列，不宣称顺序 |
| 密度预测 | fit mask 上 `sqrt(mean(((rho-rho_ref)/sigma_ref)**2)) <= 0.01` |
| 速度预测 | 每个速度分量的观测占用数加权条件 cell TV `<= 1e-3`；观测数 `>= 20` 的 cell 最大 TV `<= 0.01` |
| 同方法重复一致性 | 整数状态/坐标/映射完全相同；浮点权重、预测与 J 用 `rtol=1e-12, atol=1e-12` |

TV 在原始 201-bin likelihood support 上计算：对每个有效观测 cell，
`TV = 0.5 * sum(abs(p - p_ref))`，用该 cell 观测计数加权平均。
`p` 必须沿用正式评分的条件分布、support 和零计数处理；三个分量分别验收，
不能用绘图降采样或平均三个分量掩盖单项失败。

高精度参考不合格时该点无可用 ground-truth proxy，不能选赢家。
跨方法不要求逐权重 `1e-12` 接近；该阈值用于同实现重复或行为保持 parity。
候选还必须通过现有生产接受规则；预算封顶但预测稳定不等于可以解除 `1e30`。

增加测试：不合格参考、分项误差抵消、并列/翻转、超时、fingerprint 不匹配、
非有限数值和只读比较器。运行步骤 1 的 focused tests，并补跑
`test_weight_solver_benchmark.py`。验收后单独提交，不运行生产 case。

## 步骤 3：干净部署、冻结 pilot 库和实现 parity

先在本地运行完整 unittest、compileall 和 diff 检查，记录实现 SHA；
再以常规 Git 同步部署到服务器。服务器 dirty 或分支冲突时停止处理并报告，
不强制覆盖。步骤 0 已有任务的资源结论必须仍然有效。

```bash
conda run -n dp-jax python -m unittest discover -s tests -v
conda run -n dp-jax python -m compileall -q halo_mw_lmc apps/results.py scripts tests
git diff --check
```

在生产服务器设置步骤 2 的环境变量后：

```bash
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase preflight
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase prepare
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase parity --baseline-ref ec380dd01b6ca13044b8712b49c6ebab9b5eac2e
conda run -n halo_lmc python scripts/compare_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase parity
```

parity 在隔离子进程中分别加载旧基线和新实现。测试 harness 仅替换旧实现的
建势/积分调用以注入同一冻结库，不改生产公式；新旧都用默认 solver。
状态、坐标、轨道映射完全一致，浮点数组和 J 用 `rtol=1e-12, atol=1e-12`。
忽略时间、生成时间和源码 provenance。当前实现的三个默认求解结果若来源完全
匹配，可作为下一步的三个 baseline case，避免重复计算。

验收：缓存完整且哈希一致，三个点零失败轨道，新旧评分 parity 通过。
失败轨道先隔离积分问题，不能靠换 solver 忽略。parity 的旧基线是本方案之前的
已合并代码，**不能替代原始风格重构前后的生产验收**，后者仍按状态摘要要求补齐。

## 步骤 4：三点、两方法 pilot

| 方法标签 | backend | lsmr_tol | max_iter | 参考要求 |
| --- | --- | --- | ---: | --- |
| `trf_default` | `lsq_linear` | `1e-6` | 20000 | 保留历史成功规则，记录 KKT |
| `dense_reference` | `dense_nnls` | 不适用 | 60000 | solver_tolerance `1e-8`，通过参考资格 |

不同算法的 iteration 数含义不同，不能当作等价工作量。三个点乘两个方法共六个
逻辑 case；步骤 3 中已验证可复用的默认 case 不重跑。

```bash
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase pilot
conda run -n halo_lmc python scripts/compare_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase pilot
```

交付逐点时间、RSS、状态、KKT、内层 F、raw/正式 J、分项误差、预测误差和
点对 Delta J 表。任何参考不合格或不可比，停在这里；不要追加 GP 或更多预算点。
参考外层分数更高不直接表示参考解错误：内层没有优化速度项。

## 步骤 5：有限预算曲线

仅当三个参考全部合格时，增加三档 `lsq_linear` 的独立 cold-start 求解：
`max_iter = 300, 1000, 5000`，固定 `lsmr_tol=1e-6`，其他内容不变。
共新增九个逻辑 case。max_iter 是上限，不能强制已经成功的求解继续迭代。

```bash
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase budget
conda run -n halo_lmc python scripts/compare_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase budget
```

比较器生成时间–J 误差、时间–速度预测误差、时间–KKT、预算–点对 Delta J
的静态图及明细表。图只读取产物。

分类与选择规则：

- 数值门槛和现行生产成功规则均通过：可以作为候选。
- 数值门槛通过但迭代封顶：`approximation_promising`，正式值仍为 `1e30`，
  需要另批接受策略实验，不能在本轮改生产规则。
- 任一数值门槛失败：拒绝；不能用较小总 J 掩盖分项或预测误差。

在 pilot 和 budget 所有方法中，只考虑三个点全部通过且每点估算完整时间
`<= 600 s` 的方法，选择最差点耗时最小者；耗时差在 10% 内优先较小的
最大 KKT，再比较 peak RSS。参考方法若满足同样条件也可入选。
仅封顶方法够快则停止并报告 `approximation_policy_needed`；没有合格提速方法
则报告失败证据，不自动试新算法或改正则化。

## 步骤 6：选中方法与参考各重复到三次

每个 pilot 点、每个选中方法与参考各有三次独立进程 cold-start 结果，
计入来源一致的初次有效结果。交替运行候选和参考，避免方法与负载时间混淆；
若选中方法就是 dense_reference，按同一方法重复，不重复建两个名称相同的组。

```bash
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase repeat
conda run -n halo_lmc python scripts/compare_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase repeat
```

验收：每次均通过数值门槛、状态相同，重复浮点比较达到步骤 2 阈值；报告中位
和最大耗时。相同 case 最大/最小时间 `> 1.25` 时标记环境不稳定，调查负载与
线程池，不能先归因算法。生成 `selected_method.json`，记录选择依据、完整配置
和来源。重复性失败时停止，不偷偷放宽容差。

## 步骤 7：八点留出验证与无缓存完整确认

### 7A. 先冻结留出点，再运行

从已审计的 stage-1 48 点排除三个 pilot，按以下规则选择八个不同的历史
零失败轨道点，不查看其新求解结果后再选择：

1. 两个剩余的零失败封顶点。
2. 剩余已接受且零失败点中，历史 raw J 最小的三个点。
3. 从剩余已接受且零失败点中选三个 maximin 覆盖点：在 stage-1 Sobol 设计盒
   `q=[0.80,1.28], p=[0.70,0.96], rho0=[5.55,6.55], r2=[9.25,10.20], gamma=[0.70,1.45]`
   归一化坐标，逐点最大化到已选 pilot/holdout 集合的最小欧氏距离。

注意设计盒不同于允许 anchor 的更宽 recipe 校验边界。分数或距离并列时按
shard、iteration 升序打破并列。保存冻结点表、选择日志、历史来源和 SHA-256，
然后才开始新的积分。来源缺失或数量不足时停止，不临时改选点规则。

```bash
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase holdout-prepare
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase holdout
conda run -n halo_lmc python scripts/compare_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase holdout
```

每点运行选中方法和 dense reference；若两者相同，用两次独立重复检查。
八点全部要求合格参考、数值门槛、现行生产接受规则和零失败轨道，点对排序
遵守差值大于 2 不翻转。报告每点及最大耗时，八点小样本不宣称可靠 p95。
失败点不得从汇总分母删除或悄悄换点。

### 7B. 三个 pilot 点各三次完整无轨道缓存评估

留出通过后，选中方法返回正式完整评估路径，每个 pilot 点从建势、重新积分
开始运行三次，不读冻结轨道库，不画图；缓存仅用于结果核对。

```bash
conda run -n halo_lmc python scripts/benchmark_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase confirm
conda run -n halo_lmc python scripts/compare_density_solved_r8_40_solver_budget.py configs/benchmarks/r8_40_solver_budget.toml --phase confirm
```

验收：九次实际完整评估都 `<= 600 s`；生产规则和数值门槛均通过，轨道映射和
response fingerprint 与该点冻结基线一致，同方法重复达到 `1e-12` 阈值。
若完整积分改变轨道库，先诊断积分/环境，不能把它归为 solver 误差。
留出点亦须每点估算完整成本 `<= 600 s`；通过不代表已经覆盖整个搜索空间的尾部。

## 步骤 8：形成结论并交接，不自动启动搜索

交付 artifact-only 汇总：方法及精确设置、逐点来源/耗时/RSS、预测与分项误差、
KKT、点对排序与近似并列、重复性、轨道失败/封顶/超时数量、完整来源哈希。
更新本方案状态和 owning experiment 文档，区分已验证、假设和待验证。

最终状态只能是以下之一：

- `ready_for_bounded_search_review`：提速与精度证据通过，可评审新的有界搜索。
- `approximation_policy_needed`：只有不满足现行成功规则的近似解达到目标，需要另批设计。
- `solver_redesign_needed`：可比且完整的结果表明现有候选无法满足速度/精度要求。
- `evidence_incomplete`：参考、环境、来源、重复性或完整评估证据不足。

第一个状态不是 GP 启动授权。原始重构生产验收、support 等科学开放问题以及
搜索范围/预算仍需单独评审。不得直接启动现有 stage-2、修改默认 backend、
放宽成功门槛或将历史点注入优化器。

每步统一回执：执行步骤、Git commit/dirty 状态、配置与输入哈希、环境和线程、
实际命令、输出目录、通过/失败门槛、未解决问题、是否满足下一步前提。
可复制的委派语句：

> 阅读 `docs/solver_budget_experiment.md`，只执行步骤 N。先核实前置验收，
> 保持科学不变量和线性研究代码风格；缺证据时停止，不自动扩大实验。
> 按文档提交该步骤回执，等待我批准下一步。
