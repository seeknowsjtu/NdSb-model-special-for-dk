# NdSb3Tmodel

NdSb3Tmodel 是一个用于 **NdSb 超快非平衡动力学** 的 Python 项目，核心是扩展的三温模型（3TM），并引入磁有序参量 `m` 与重取向自由度 `eta/phi`，用于模拟泵浦后瞬态响应并对实验数据做拟合。

---

## 1. 项目能做什么

- **时域仿真**：求解 `Te/Ts/Tl/m/eta(或phi)` 的耦合动力学。
- **GUI 交互**（Tkinter）：改参数、跑模拟、加载 CSV、执行拟合、查看曲线。
- **单数据集拟合**：支持 `Te`、`S` 或 `Te+S` 拟合。
- **多数据集全局拟合**：支持多条 `S(t)` 曲线共享全局参数、每条曲线有局部参数。
- **结果导出与诊断**：导出拟合结果，并在模型端提供能量与通道相关诊断信息。

---

## 2. 代码结构

- `main.py`：主入口。优先启动 GUI；若 tkinter 不可用则回退到命令行演示流程。  
- `solver.py`：`NdSb3TM` 主求解器，包含泵浦源项、热容、耦合、序参量动力学与 ODE 积分。  
- `physics_engine.py`：Debye/Schottky/高斯峰/magnon 热容与耦合增强函数。  
- `data_io.py`：CSV 自动识别、基线预处理、拟合与多数据集导出逻辑。  
- `gui_component.py`：GUI 参数表单与滚动组件定义。  
- `config.py`：默认参数、常数、参数归一化和工具函数。  
- `demo_multi_fit_backend.py`：合成数据 + 多数据集拟合的后端最小示例。  
- `run_real_multi_fit.py`：读取仓库内真实 CSV 做多数据集拟合（脚本式流程）。

---

## 3. 环境与依赖

建议 Python 3.10+。

依赖：

- `numpy`
- `scipy`
- `matplotlib`
- `tkinter`（可选；仅 GUI 需要）

安装示例：

```bash
pip install numpy scipy matplotlib
```

> Linux 若缺 tkinter，需要系统包安装（例如 `python3-tk`）。无 tkinter 时仍可运行非 GUI 路径。

---

## 4. 快速开始

### 4.1 启动主程序（推荐）

```bash
python main.py
```

- 有 tkinter：打开图形界面。
- 无 tkinter：自动运行 CLI fallback 演示。

### 4.2 运行后端多数据集演示（合成数据）

```bash
python demo_multi_fit_backend.py
```

### 4.3 运行真实 CSV 的多数据集拟合

```bash
python run_real_multi_fit.py
```

默认会读取脚本里列出的 CSV（当前启用 1.0/2.0/2.5 mW 三条）。

---

## 5. 物理模型概览

模型状态变量为：

- `Te`：电子温度
- `Ts`：自旋有效温度
- `Tl`：晶格温度
- `m`：磁有序参量
- 第五自由度：`eta`（标量）或 `phi`（角变量，输出 `eta=cos(2phi)`）

### 5.1 热容部分

- 电子热容：`Ce = gamma(Te, Ts, m, eta) * Te`
- 晶格热容：Debye 模型（带缓存插值）
- 自旋热容：可选 `magnon` 或 `AT^3`，并叠加 Schottky 与峰项

### 5.2 能量耦合通道

- `G_el0`：电子 → 晶格主通道
- `G_es0`：电子 ↔ 自旋辅助通道
- `G_sl0`：自旋/序参量 ↔ 晶格主通道

这些通道在实现中允许随 `m/eta/T` 调制（例如 `G_es_m_power`、`TR/TN` 附近增强等）。

### 5.3 序参量动力学

- `m`：通过 `m_eq(Ts)` 与 `tau_m(Ts)` 弛豫演化（含临界慢化项）。
- `eta/phi`：可用标量 Landau 形式，也可用角变量重取向势能形式。

---

## 6. 拟合能力

## 6.1 单数据集拟合（GUI/后端公用）

`data_io.py` 提供 `fit_params(...)` 等接口，支持：

- `Te` 拟合
- `S` 拟合
- `Te + S` 联合拟合

### 6.2 多数据集全局拟合

`fit_params_multi(...)` 支持：

- **global_keys**：所有数据集共享参数（如 `S_scale/G_es0/G_sl0/tau_m0`）
- **local_keys**：每个数据集独立参数（默认 `dt_local`）
- **observable_mode**：可选 `eta`、`raw_chi2q` 等映射模式

`run_real_multi_fit.py` 中默认权重与优化控制示例：

- `sigma_S=0.02`
- `max_nfev` smoke test + full run 分阶段
- 周期 heartbeat 输出运行状态

---

## 7. CSV 数据格式

读取采用 `load_csv_auto(...)` 自动识别列名。

### 7.1 时间列

支持诸如：`tps`, `t_ps`, `time_ps`, `time`, `t`。

- 若时间量级看起来像 ps（`max(|t|)>1e-6`），会自动按 ps 转换到秒。
- 内部统一使用秒。

### 7.2 观测列（可选）

- 电子温度：`tek`, `te_k`, `te`, ...
- 自旋温度：`tsk`, `ts_k`, `ts`, ...
- 晶格温度：`tlk`, `tl_k`, `tl`, ...
- 光谱量：`s`, `spec`, `weight`, `intensity`, ...

### 7.3 预处理

- 自动剔除非有限值。
- 按时间排序。
- 重复时间点做平均。
- `load_s_dataset_csv(...)` 会对 `S` 做基线扣除（优先 `t<0` 区间均值）。

---

## 8. 默认参数与可调入口

默认参数集中在 `config.default_params()`，关键锚点包括：

- `T_bath = 4 K`
- `TN = 15 K`
- `TR = 13 K`
- `ThetaD = 200 K`
- `eta_representation = "scalar"`
- `sw_model = "magnon"`

实践上，常优先调这些参数：

- 泵浦：`fluence_multiplier`, `S_scale`, `pulse_width`, `t0_pulse`
- 通道：`G_el0`, `G_es0`, `G_sl0`
- 序参量：`tau_m0`, `tau_m_crit_amp`, `Gamma_eta`, `a_eta0`, `b_eta`
- 观测映射：`S_offset`, `S_amp`, `S_power` 或多数据集中的 `A_obs/B_obs`

---

## 9. 常见使用建议

- **先模拟再拟合**：先用 GUI 看响应量级，再收窄拟合参数范围。
- **先少量参数**：多数据集拟合先开少量 `global_keys`，稳定后再加。
- **关注边界告警**：若参数常贴边，通常意味着模型/初值/边界设定需要调整。
- **检查基线处理**：实验 `S(t)` 的基线会显著影响拟合质量。

---

## 10. 已包含示例数据

仓库包含多条 `deltak12k_*mW.csv` 数据，可直接用于脚本与 GUI 测试。

---

## 11. 许可证与说明

当前仓库未单独提供 LICENSE 文件；若需开源发布，建议补充明确许可证。
