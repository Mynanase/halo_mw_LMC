# 重构、合并与实验状态摘要

整理日期：2026-09-10。代码基线为 `ec380dd01b6ca13044b8712b49c6ebab9b5eac2e`，
分支为 `codex/no-fixed-weights`。本文记录本轮已完成的工作和证据边界；
下一步执行契约见 [权重求解预算实验方案](solver_budget_experiment.md)。

## 1. 本轮修改与提交组织

原有 joint objective 修改先单独提交，之后才进行线性研究代码风格重构：

| 提交 | 内容 |
| --- | --- |
| `9c13e4f` | 保存原有 10 个文件的 joint density–velocity objective 修改，不混入风格修改 |
| `70dbbe6` | 固化紧凑、顺序执行的代码风格，清理旧 CLI 与重复 density 入口 |
| `fe20ec0` | core/data 分层内线性化，保留数学函数、数据契约与求解器 backend |
| `008b5b7` | configuration/workflows/artifacts 等顺序化，保留输入与持久化边界校验 |
| `7f057e5` | 分析、绘图、应用和研究脚本简化，保持 artifact-only 展示边界 |
| `579279e` | 测试与文档对齐研究代码风格 |
| `ec380dd` | 合并远端 stage-1 筛选、stage-2 配置和五参数 corner 图工作 |

合并接入的远端历史包括 `270da3a`、`a4f0ee7`、`0f15ed8`、`5e2c82e`、
`cdc042f`。这些提交分别记录固定线程下的正则化探针、stage-1 基础设施及修正、
stage-2 协议与配置、五参数 profiled corner 图。

合并中处理了两个实际重叠：

- 本地和远端同名 joint recipe 的搜索边界不同。本地较窄的
  `zhu_2026_density_solved_r8_40_joint.toml` 保持原样；远端筛选版本另存为
  `zhu_2026_density_solved_r8_40_joint_screen.toml`，对应 12 个 shard 和
  stage-2 anchor 改为引用它。没有统一或悄悄缩窄两种实验的边界。
- 远端 corner 图复用了本地重构时内联的绘图辅助步骤；恢复真正共享的
  `_fallback_reason` 和 `_panel_sample_coordinates`，保留紧凑调用。

合并同时补充 stage-1 的 48 点/anchor/边界、stage-2 side-run 一致性和
corner 报告输出的回归检查。

## 2. 新架构保留了什么

此次是分层内的实现重构，不是取消科学模块边界：

| 层 | 职责与边界 |
| --- | --- |
| `halo_mw_lmc/core/` | 势、轨道、稀疏响应、非负权重、密度和速度的数值计算 |
| `halo_mw_lmc/data/` | 星表与 density target 文件适配；数组进入 core 前完成格式转换 |
| `configuration.py` | 严格 TOML 解析；recipe 管科学选择，run 管数据路径、输出和调度 |
| `halo_mw_lmc/workflows/` | 一次准备、逐势积分与评分、fixed-point 或 adaptive ask/tell |
| `artifacts.py` / `inspection.py` | 版本化数组与配置持久化；从权威产物重建状态 |
| `visualization/` / `apps/` / 分析脚本 | 只读已有产物，生成 Matplotlib 图或 Marimo 展示，不补跑积分 |

活跃 Python 使用从上到下的数据流、直接数组操作和少量有明确意义的函数。
不引入 formatter 或固定行宽；保留公开函数、dataclass、数学概念、算法 backend、
原子写入和框架 callback。`archive/`、`Agama-master/`、数据与运行产物不参与风格重构。

当前入口为 `halo-mw-lmc SUBCOMMAND` 或 `python -m halo_mw_lmc SUBCOMMAND`，
保留 `run/optimize/evaluate/coverage/validate/preflight/report/inspect` 八个子命令。
无子命令的 `CONFIG/-v/-c/-o` 旧路由已经删除；`halo-mw-lmc-density` 和
`python -m halo_mw_lmc.generate_density` 已删除。
density target 生成保留 `scripts/generate_synthetic_density.py CONFIG`。

TOML 科学字段、sample columns、单位、dtype、shape 和权重语义没有借重构修改。
当前 best artifact v4 继续读取 v2–v4，resolved-config v7 继续读取 v4–v7；
旧文件名 fallback 和历史 ASCII density target 读取仍保留。
完整数组契约见 [architecture.md](architecture.md)。

## 3. 当前科学问题

`catalogue_fixed` 仍是保守基线；实验性 `density_solved` 在每个势中求解
密度约束下的非负权重，再用同一组权重评分速度。
内层是带 L2 正则的 density-only 问题；外层 joint objective 为
`J = chi2_density / 2 - log L_velocity`。速度没有进入内层求解，
正则项没有再次加入外层，权重不作事后归一化。

活跃 r8–40 实验保留密度 `|z| >= 2 kpc` 掩膜；速度没有对应的垂直掩膜。
201-bin 拟合网格独立于绘图降采样。这些尚未解决的科学边界不会在提速实验中改变。
joint 模式取消的是密度质量硬门槛，不是求解器失败门槛：失败或封顶点的
正式 objective 仍为 `1e30`。

## 4. 已有实验说明了什么

### Stage-1：48 点筛选已产生结果，但不能直接当作可靠搜索训练集

此前对 12 个 shard 的只读产物审计核对了 159 个提取文件的 SHA-256，
48 个坐标唯一且与各自 resolved config 一致。以下分数来自历史 sample，
保留精度约为 0.001，不能用它们做 `1e-12` 数值对照。

- 39/48 点被历史求解器成功规则接受，9/48 点达到 20,000 次迭代上限。
- 15/48 点有轨道失败；接受且零失败轨道的点为 30，封顶且零失败的点为 3。
- 39 个接受点中仅 7 个满足 normalized KKT residual `<= 1e-8`；
  “success”不表示所有点达到了统一精度。
- 9 个封顶点占累计求解 wall time 的 53.47%。单点求解最小/中位/最大约为
  83.1/373.1/4825.5 秒；并行 worker 的累计耗时不能解释为批次经过时间。

参数顺序为 `(qhalo, phalo, rho0, rho0_plus_2logrs, gamma)`：

| 点 | 坐标 | 原始 J | 正式状态 | KKT residual |
| --- | --- | ---: | --- | ---: |
| 最佳已接受 `s09:2` | `(0.985, 0.714, 6.099, 9.829, 0.751)` | 132834.861 | accepted，零失败轨道 | 2.51e-5 |
| 待复核 `s07:2` | `(1.071, 0.738, 5.687, 9.541, 1.214)` | 132812.150 | capped，正式值 1e30，零失败轨道 | 3.07e-5 |

候选的原始领先约 22.711，不足以绕过求解状态。更精确的内层密度解也不保证
外层 joint J 单调下降，所以封顶点的原始 J **不是经过证明的下界**。
不能据此宣布它是新的最优势。

历史数值运行记录为 `a4f0ee7` 且 `git_dirty=true`，后续报告来自 `0f15ed8`；
未保存的 dirty diff 无法恢复。该批数据适合发现问题和选择复核点，
不构成干净源码的严格重构前后基线。原始数据和机器证据保留在忽略的本地目录，
不随本次文档发布。

### 已有求解器与线程实验：有线索，尚无可替换默认求解器的结论

此前 paper-best 同题三次重复的 solver benchmark 中，`lsq_linear` 求解中位约
498 秒、KKT 约 `5.88e-5`；`dense_nnls` 约 995 秒、KKT 约 `9.57e-16`；
`dual_ridge` 约 2329 秒、KKT 约 12.73，未通过。该实验未选出生产替换方案。

另外的固定 BLAS 单线程实验中，paper-best 完整 case 曾达到约 209–218 秒，
其中 AGAMA 积分约 11–15 秒。线程设置会影响浮点归约与求解路径；
这些不同批次时间不能直接当作算法加速比。固定线程后的 `lambda=1e-5` 探针
没有支持“增大正则化即可解决慢求解”的判断。

历史五点容差对照保留了相同最佳点，但出现过一个排序对翻转；
它也没有给当前 48 点的求解误差提供可直接套用的上界。
下一步应测量：在完全相同的轨道库上，减少求解预算会怎样影响速度预测和 `Delta J`。

### Stage-2：基础设施已合入，执行与准入分开判断

仓库已有 60 点 cold-start GP 配置、精确 anchor `rho0_plus_2logrs=9.353`
和封顶候选 max-iteration 60,000 的 side-run 配置。已有配置不等于本轮已经执行。
本次发布没有查询生产服务器实时进程，也不声称服务器从未运行这些配置。
任何后续 Agent 应先核实已有运行，再按新方案逐阶段执行；不得重复启动、
停止已有进程或自动启动 GP。

五参数 corner 图已接入报告，但筛选成功点与训练支持仍须满足其准入条件。
图中的 profiled contours 不是 posterior，也不是已校准的参数不确定度。

## 5. 验证状态与剩余工作

合并时已经执行的本地验证记录（不是本次文档发布重新运行的结果）：

- `dp-jax` 全套 unittest：177 个测试，176 通过、1 个可选依赖测试跳过。
- 56 个 TOML 可加载；远端 19 个相关配置在明确的 recipe 重命名之外保持语义一致。
- 活跃 package、apps、scripts、tests 的 compileall、相关 shell 语法检查及
  `git diff --check` 通过。

2026-09-10 发布前再次运行 `dp-jax` 全套 unittest：177 个测试，176 通过、
1 个跳过；`python -m compileall -q halo_mw_lmc apps/results.py scripts tests`
及 `git diff --check` 通过。新增文档的相对链接已核对。本次提交仅更新说明文件，
没有实现新预算 runner、改变科学配置或启动生产计算。

本地缺少可用的生产 `halo_lmc`/AGAMA 部署，生产重构验收仍待完成：
需要同输入、同环境、新输出目录的 paper-best 重构前后完整评估，整数状态、
轨道计数和坐标完全一致，浮点数组及 objective 用 `rtol=1e-12, atol=1e-12`
对照，排除时间和 Git provenance。通过前不把整仓库重构标成生产验收完成。

新预算实验还会增加自身的冻结轨道库 parity 和无缓存完整确认；这两项不能
替代上述原始重构前后的生产验收。

当前优先级是 [solver budget 实验](solver_budget_experiment.md)：先三个点、
两个方法，按证据决定是否扩大；目标是每次完整评估不超过 600 秒，同时保护
下游预测与参数排序。未实现的 runner、比较器和配置在方案中明确标为待建，
本次只发布现有修改和计划，不启动数值实验或 adaptive scan。

## 6. 2026-09-13 追记：扁平化实验分支

用户已批准并在 `codex/flatten-research-native` 分支上完成研究原生扁平化实验：
`core/`、`data/`、`workflows/`、`visualization/` 分层目录已取消，改为每管线
阶段一个平铺模块（potential/orbits/grids/density/weights/velocity/coverage/
catalogue/config/prepare/evaluate/optimize/run/report/plot_* 等）；冻结
dataclass 配置树与 exact-field TOML 校验已替换为普通 dict 配置
（`load_recipe_configuration` / `load_run_configuration` /
`load_synthetic_density_configuration` / `resolve_model`，键契约在
`config.py` docstring 记录一次），科学函数改为签名内默认值并以
`**cfg["density_fit"]` 形式穿线；校验收敛到三条薄防线（输入数据边界、run
目录隔离、产物 provenance），`ConfigurationError` 等异常层级删除，CLI
不再捕获异常。这只改变代码组织与配置表示，不改变任何科学假设、数组契约
或阈值；本分支的完整 unittest 套件（207 项）与 compileall 通过，55 个
check-in 配置全部可加载。本文 §2 记录的分层边界在 `main` 上继续有效；
生产 parity 验收结论仅适用于扁平化之前的代码，扁平分支合并前需重做同标准
验收。分支契约见 [architecture.md](architecture.md) 顶部的分支说明。

## 7. 2026-09-13 追记：扁平化生产 parity 验收与 runs/ 两层规范

同标准 parity 验收已在扁平分支上完成：以新结构配置
`configs/runs/flatten_parity_paper_best.toml` 重跑
paper-best 单点 8–40 kpc 基准（同 recipe、同数据、同 seed、冷启动新目录），
best objective `133666.96061010682` 与扁平化前基准
`runs/legacy/density-solved-r8-40-paper-best-benchmark` 逐位一致；
`sample.dat` 数值列逐位相同（新 sample.dat 按新 schema 多 3 列诊断列：
`zero_weight_fraction`、`weight_solver_kkt_residual`、`weight_solver_wall_seconds`），
`best/evaluation.npz` 与
`weight_model_inputs.npz` 的共有数组全部 bit-exact（NaN 模式相同），
差异仅为新 evaluation schema v4 新增的 solver 诊断列
（backend/kkt/fingerprint/wall_seconds）与 `weight_solver_cost`
尾数差 1.5e-13（远小于 rtol=1e-12 标准）。验收证据在新结构首个
family `runs/20260913-flatten-parity/paper-best`。

同日完成 runs/ 目录清理与两层规范：历史 36 个平铺 run 目录精简为 19 个
保留目录（17 个 retired 至本地 `.archive/runs-retired/`），随后一次性迁移至
`runs/legacy/`（叶子名不变，双 manifest 证明零内容变化）；新实验统一采用
`runs/<YYYYMMDD>-<family>/<variant>` 两层结构与同族
`coverage/<variant>` 输出，约定见本机 `runs/README.md`（不入库）；
`discover_runs` 已支持两层发现（带嵌套用例测试）。

## 8. 2026-09-17 追记：分束修复战役归档（nphi1→nphi4）

2026-09-13 求解器提速战役（`solver_settings_sweep`，本机 `.agent-local/`）
的分束发现在 09-14 至 09-16 展开为一轮完整的方法修复与诊断战役，
记录、脚本与配置本轮入库；完整实验记录见
[nphi1 分束修复方案](nphi1_bundling_repair_plan.md) §1–§11。脉络：

1. **修复（§1–§5）**：审计发现首轮 nphi1 分束实验五处结论性错误
   （random 对照求解/回填映射不一致、chi2 误带体积归一化、提速比计时
   口径不一、势参数不一致、面板图误用四扇区 target），全部撤回并固定
   科学定义；生产评分抽为共享边界 `score_orbit_weights`（两条 weight
   分支 parity 1e-12），full/bundled/random 在同一冻结库上同点复核。
2. **二维基与机制（§6–§7）**：换守恒且正交的二维基（(E,ω_z)、(Jr,Jz)）
   不改变模式——物理分组改善内层密度、速度 J 输给 random；四项可分离
   检验定位机制：内层 NNLS 欠定，逐轨道权重纹理对 held-out 的速度 J 是
   净损害，"random 赢 J"=以密度拟合换权重平滑度，生产语义被密度门槛
   封死。
3. **基准反转（§8）**：生产 25×25×4 网格上 1024 束全基门槛失败——
   nphi1 的"分束轻松过门槛"是轴对称化约束过少的人为现象；可行工作点
   4096 束（物理优于 random，solve 约 4.5×，密度代价 chi2/bin
   0.455→1.36）。
4. **减稀否定（§9）**：full 路径 z 向减 bin（25→13→7）既不提速（TRF
   迭代数支配耗时）也不缓解权重集中；n_z=7 的 J 下降是信息丢失。
   生产网格维持 25×25×4。
5. **新基（§10）**：(f_z=Jz/Jtot, E) 基把工作点 J 代价从 +926 降到
   +39.5（约 23×）且过门槛；轨道画像给出机制证据（旧 (λ_z,E) 16×16
   格混合薄/厚轨道 z_rms 1.4 vs 13.3 kpc，新基将其分开）。
6. **已运行（§11）**：第三不变量审计完成（2026-09-17，tycho）——第三轴候选
   （λ_z、xy 主轴角）在 1024 束不能挽救门槛；k-means 响应相似性 oracle 在同
   束数把失真 D 从 0.93–0.95 压到 0.45、投影 chi2/bin 5.44 ≈ (f_z,E)@4096 的
   5.41——判读为变量问题而非束数问题。详见修复方案 §11。

边界：以上全部是单冻结库、单次测量的方法诊断，不构成生产替代或参数
排序结论；修复方案的阶段3（生产锚点对齐）与阶段4（三势点排序 pilot）
未进入。产物在本机 `.agent-local/benchmarks/nphi*_*/`（不入库）。

本轮提交组织：

| 提交 | 内容 |
| --- | --- |
| `4e2d27e` | 忽略本地 `.zcode/` 目录 |
| `d8b6c2c` | 生产评分抽共享边界 `score_orbit_weights` + 双模式 parity 测试 |
| `e3f5087` | 战役主体：实验记录（§1–§10）+ 5 个诊断/驱动脚本 + 15 项小数组测试 + 21 个配置（提交正文含脚本↔§对照） |
| `0fe4b83` | §11 计划小节 + 第三不变量审计脚本（已写待运行） |
| 本提交 | 状态文档追记 |

测试基线随本役更新：unittest 226 项全绿（halo_lmc 环境），compileall
通过。

2026-09-17 补记：§11 审计在延续分支 `codex/third-invariant-audit` 上运行完成
（运行代码 = `08ccc8c` + 两处审计脚本报告/图修复 `29b341e`、`9f3521b`；
审计全程无求解，wall 124 s）。结果与判读见修复方案 §11：1024 束门槛失败主要
是变量问题——响应相似性 oracle 在同束数把 D 减半，不变量坐标划分留有约 2 倍
失真空间；第三轴候选与束数曲线都未在 1024 束附近给出接近可行的改善。tycho 复跑
unittest 225/226 通过，唯一 error 为 benchmark 冷启动目录检查撞上当日上午
module-CLI orx 基准留在 `runs/density-solved-r8-40-paper-best-benchmark` 的
输出目录（测试行为符合设计，与本役改动无关）。

同日再记（§11 更正 + §12 oracle 重解）：§12 实现的小数组测试发现
`equal_weight_distortion` 漏乘成员数 n_k，§10/§11 的 D 与 coherence 数值
全部作废并已重算（投影 chi2 与重解结果不受影响；排序结论幸存，oracle 的
D 优势实为约 3 倍）。oracle 束重解实验（`grouping_mode="response_kmeans"`）：
k=1024 重解 chi2 2.644 不过门槛——1024 束即使分组接近已实现最优也不可行；
**k=2304 过门槛（1.162，优于 (f_z,E)@4096 的 1.606，束数近半，solve 13.4 s，
N_eff 698 对 66.6）**，成为新的已知最优工作点；其 J 低于 full 是 §7 平滑
机制在门槛内侧的表现，非更优拟合；k-means 分组 74.6 s 使 end-to-end
（100.6 s）仍慢于 (f_z,E)@4096（约 68 s）。完整记录见修复方案 §10/§11
更正与 §12。
