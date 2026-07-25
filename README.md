# halo_mw_LMC

这条分支实现 Zhu 等人的经验轨道叠加方法，并把原来的轴对称
`(R, z)` 数据—模型比较扩展为显式的 `(R, z, phi)` 比较。

当前主流程为：

1. 在目标密度 `nu_target(R,z,phi)` 的每个空间格中统计 6D 种子星；
2. 一次性计算固定代表性权重
   `w_i = nu_target[j] V[j] / N_seed[j]`；
3. 以观测到的 6D 恒星为轨道种子；
4. 在每个候选银河系势中用 AGAMA 积分，并等时间采样；
5. 每个轨道点继承预先固定的种子轨道权重；
6. 将轨道点放入相同的 `(R, z, phi)` 网格，以精确柱坐标体积换算密度；
7. 在全部 phi bin 上只拟合一个共同的模型幅度；
8. 逐 phi 计算 target/model 残差和 chi-square；可选地加入三个速度分量的逐星似然。

核心实现位于 `halo_mw_lmc/`。原来的函数名
`calculate_RzSB_4phi`、`Read_obsSB_4phi` 和 `int_one_model`
保留为兼容入口。

## 运行

需要在环境中安装 NumPy、Astropy、Matplotlib、scikit-optimize 和
[AGAMA](https://github.com/GalacticDynamics-Oxford/Agama)。然后执行：

```bash
python run_skopt_lamost_4phi.py \
  --base-path /path/to/halo_mw_LMC \
  --nphi 4 \
  --iterations 1000
```

若改变 `--nphi`、`--n-rz` 或 `--rz-max`，必须通过 `--density`
给出使用完全相同边界生成的 `nu_target(R,z,phi)` 文件。加入 Zhu 的速度项可使用
`--include-velocity`。

运行开始时会把固定权重、每格种子数、target density 和网格边界写入
`model_skopt/fixed_weights_rzphi.npz`。这些权重在整个势参数优化过程中
保持不变。

算法约定和旧代码差异见
[`docs/zhu_phi_binning.md`](docs/zhu_phi_binning.md)。

## 测试

```bash
python -m unittest discover -s tests -v
```
