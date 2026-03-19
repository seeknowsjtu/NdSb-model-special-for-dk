# NdSb3Tmodel：NdSb 非平衡动力学模拟与拟合工具

## 项目简介

`NdSb3Tmodel` 是一个面向 **NdSb（锑化钕）超快非平衡动力学** 的 Python 模型与分析工具。它以扩展三温模型（Three-Temperature Model, 3TM）为核心，同时引入：

- 电子温度 `Te`
- 自旋温度 `Ts`
- 晶格温度 `Tl`
- 主磁有序参量 `m`
- 自旋重取向相关参量 `eta`

从而描述 NdSb 在泵浦激发后的能量传输、磁有序演化以及光谱代理量变化。项目同时提供：

- **图形界面（Tkinter + Matplotlib）**：用于交互式调参、模拟、拟合和结果预览。
- **拟合能力（SciPy least_squares）**：支持将模型与实验 CSV 数据中的电子温度 `Te(t)` 和/或谱权重 `S(t)` 进行有界鲁棒拟合。
- **无界面模式回退**：当本机缺少 tkinter 时，程序仍可执行默认演示仿真。

该仓库比较适合以下工作：

1. 快速试算 NdSb 泵浦-探测后的三温耦合响应。
2. 调整有效耦合、热容、序参量动力学参数，观察瞬态行为。
3. 读取实验 CSV 数据并完成 `Te`、`S` 或联合拟合。
4. 检查能量守恒闭合误差和有效耦合随时间的变化。

---

## 核心功能概览

### 1. 扩展三温模型求解

求解器并不只处理传统的电子/晶格/自旋三温模型，还额外加入了两个与 NdSb 相变相关的自由度：

- `m(t)`：磁有序参量；
- `eta(t)`：1q/2q 自旋重取向相关自由度。

因此模型实际求解的是五个耦合变量的常微分方程组：

- `Te(t)`
- `Ts(t)`
- `Tl(t)`
- `m(t)`
- `eta(t)`

### 2. 物理上更细的热容建模

项目包含多种热容来源：

- **电子热容**：使用 `gamma(T, m, eta)` 形式，允许能隙抑制电子热容。
- **晶格热容**：使用 Debye 模型，并通过插值缓存提升计算速度。
- **自旋热容**：包含以下部分：
  - CEF 多能级 Schottky 贡献；
  - 低温自旋波 / magnon 热容；
  - Néel 温度 `TN` 附近峰；
  - 重取向温度 `TR` 附近峰。

### 3. 状态依赖的能量耦合

能量交换通道不是固定常数，而是可以随磁有序度和温度变化：

- `G_el_eff`：电子 → 晶格主通道；
- `G_es_eff`：电子 ↔ 自旋辅助交换通道；
- `G_sl_eff`：自旋 / 有序参量 ↔ 晶格主弛豫通道。

其中：

- `G_es_eff` 可依赖 `m` 和 `eta`；
- `G_sl_eff` 可在 `TR`、`TN` 附近增强；
- 所有这些量都可在 GUI 中直接调整并在图上查看。

### 4. 图形交互 + 数据拟合

GUI 提供：

- 参数编辑；
- 一键模拟；
- CSV 数据读取；
- `Fit Te`、`Fit S`、`Fit Te + S` 三种拟合模式；
- 将拟合结果回写到主参数；
- 结果曲线与实验点叠加显示。

### 5. 能量平衡诊断

程序会在仿真后尝试生成能量诊断量，包括：

- 输入能量积分；
- 与热浴耦合导致的损耗；
- 内能变化；
- 能量闭合误差；
- 各耦合通道与功率项随时间的曲线。

这对于判断参数是否物理合理很有帮助。

---

## 仓库结构与各文件功能

### `main.py`

项目主入口，负责：

- 检测是否可用 `tkinter`；
- 在可用时启动 GUI；
- 管理参数输入、按钮逻辑、日志输出；
- 调用求解器执行仿真；
- 加载 CSV；
- 调用拟合函数；
- 在无 GUI 环境下运行一个默认 CLI 演示。

你平时使用本项目时，通常直接运行：

```bash
python main.py
```

### `solver.py`

项目的核心求解器，主要包含 `NdSb3TM` 类，负责：

- 构建激光泵浦源项；
- 定义 `m` 的平衡值和弛豫时间；
- 定义 `eta` 的自由能、平衡态和动力学；
- 计算 `Ce / Cs / Cl`；
- 计算有效耦合 `G_el_eff / G_es_eff / G_sl_eff`；
- 构建 ODE 右端项 `rhs`；
- 通过 `solve_ivp` 进行数值积分；
- 输出能量诊断信息。

### `physics_engine.py`

物理辅助模块，主要负责热容和耦合增强模型：

- Debye 晶格热容缓存与插值；
- CEF Schottky 热容；
- 高斯峰形热容；
- 基于 LSWT 的 magnon 热容；
- 预计算 `MagnonCvLUT` 加速拟合；
- `exchange_scale()`：控制 `G_es_eff` 的状态依赖；
- `spin_lattice_enhancement()`：控制 `G_sl_eff` 在 `TR/TN` 附近增强。

### `data_io.py`

数据输入与拟合模块，负责：

- 读取 CSV 并自动识别列名；
- 自动识别时间单位（秒或皮秒）；
- 对重复时间点做平均；
- 为拟合参数提供上下界；
- 使用 `scipy.optimize.least_squares` 进行鲁棒拟合。

### `config.py`

集中定义默认物理参数与工具函数，包括：

- 常数与单位换算；
- NdSb 摩尔体密度；
- 默认热力学锚点（如 `TN=15 K`、`TR=13 K`、`ThetaD=200 K`）；
- 默认泵浦参数；
- 默认耦合强度；
- 默认 magnon / CEF / 序参量参数；
- 数值辅助函数 `clipT`、`safe_float`、`fmt_num`。

### `gui_component.py`

一个自定义可滚动 Tkinter Frame 组件，用于承载大量参数输入框，避免 GUI 面板过长难以操作。

---

## 模型中各物理量的含义

### 温度自由度

- `Te`：电子温度，代表激光首先加热的热电子子系统。
- `Ts`：自旋扇区有效温度，用于承载磁相关自由度的热化。
- `Tl`：晶格温度，对应声子/晶格热浴。

### 磁有序参量 `m`

- `m_eq(Ts)` 在 `Ts < TN` 时非零，在 `Ts >= TN` 时为 0；
- 动力学采用趋近于平衡值的弛豫形式；
- 弛豫时间 `tau_m(Ts)` 包含临界慢化项。

### 自旋重取向参量 `eta`

- `eta` 用于描述 `TR` 附近的重取向自由度；
- 支持 **二级相变式** 与 **一级相变式** 两种自由能形式；
- 由 `Gamma_eta`、`a_eta0`、`b_eta`、`c_eta` 等参数控制。

### 光谱代理量 `S(t)`

程序中并没有直接计算完整光谱，而是提供一个简化代理量：

```text
S(t) = S_offset + S_amp * m(t)^S_power
```

这个量主要用于与实验提取出来的谱权重、强度或某类 order-sensitive observable 进行拟合。

---

## 默认参数的物理锚点

当前默认值体现的主要假设包括：

- `T_bath = 4 K`
- `TN = 15 K`
- `TR = 13 K`
- `ThetaD = 200 K`
- 默认启用 `eta` 动力学
- 默认自旋热容模型为 `magnon`
- 默认开启 Debye、Schottky、临界峰与 magnon 贡献

如果你的实验样品、泵浦条件或目标 observable 与默认设置差异较大，建议优先修改：

- 泵浦参数：`fluence_multiplier`、`pulse_width`、`delta_opt`、`S_scale`
- 耦合参数：`G_el0`、`G_es0`、`G_sl0`
- 磁动力学参数：`tau_m0`、`tau_m_crit_amp`、`nu`
- `S(t)` 映射参数：`S_offset`、`S_amp`、`S_power`

---

## 安装说明

### Python 版本

建议使用 **Python 3.8 及以上**。

### 依赖安装

项目主要依赖：

- `numpy`
- `scipy`
- `matplotlib`
- `tkinter`（GUI 模式需要，很多 Python 发行版默认自带）

安装示例：

```bash
pip install numpy scipy matplotlib
```

如果你在 Linux 上发现无法启动 GUI，通常需要额外安装 tkinter，例如：

```bash
sudo apt-get install python3-tk
```

> 注意：在无图形界面或缺失 tkinter 的环境下，程序会自动进入无界面演示模式，而不是直接报错退出。

---

## 快速开始

### 1. 启动程序

```bash
python main.py
```

- 若 `tkinter` 可用：启动 GUI。
- 若 `tkinter` 不可用：运行一个默认仿真并在终端输出峰值温度和最终序参量。

### 2. GUI 中最常见的使用流程

1. 打开程序。
2. 在左侧参数面板中修改参数。
3. 设定仿真时间范围：
   - `t0 (ps)`
   - `t1 (ps)`
   - `N`
4. 点击 **Simulate**。
5. 在右侧查看四张图：
   - 温度演化；
   - `m / eta / S`；
   - 有效耦合；
   - 功率流。

### 3. 无界面环境下的行为

当前仓库中的“CLI 模式”不是完整命令行参数系统，而是一个 **headless demo**。它会：

- 读取默认参数；
- 运行一段固定时间窗口内的仿真；
- 在终端打印 `max Te / max Ts / max Tl / final m / final eta`。

如果你要做真正的批处理拟合，建议直接在 Python 脚本中导入 `NdSb3TM`、`load_csv_auto`、`fit_params` 自行调用。

---

## GUI 说明

GUI 主要分为四个参数标签页：

### 1. Basic

用于设置：

- 泵浦参数；
- 电子-晶格、自旋等有效耦合；
- 与热浴的时间常数；
- 电子热容与热力学锚点；
- 自旋热容模型选择。

### 2. Spin / Magnon

用于设置：

- `J1/J2` 有效交换常数缩放；
- LSWT magnon 积分网格；
- `S_eff`、magnon gap；
- 旧版 `A*T^3` 自旋热容近似参数。

### 3. Eta

用于设置：

- 是否启用 `eta`；
- `eta_mode`（一级/二级）；
- `eta` 的符号、裁剪范围；
- `Gamma_eta`、`eta_dT`；
- Landau 系数。

### 4. Advanced

用于设置：

- `G_es`、`G_sl` 的更细致形状参数；
- `TN/TR` 附近热容峰参数；
- CEF 激发能级；
- `m` 动力学参数；
- `S(t)` 映射参数。

### 5. 按钮说明

- `Simulate`：按当前参数执行仿真。
- `Load CSV...`：加载实验数据。
- `Show Main (p)`：显示主参数字典。
- `Show Fit (p_fit)`：切换查看拟合后参数。
- `Fit Te`：仅拟合电子温度。
- `Fit S`：仅拟合谱权重代理量。
- `Fit Te + S`：联合拟合。
- `Apply Fit → Params`：将拟合参数写回主参数并立即重画结果。

---

## 输出图的含义

程序模拟后会在 GUI 中显示四个子图：

### 温度图（Temperatures）

显示：

- `Te`
- `Ts`
- `Tl`
- `TN`
- `TR`

可用于观察：

- 电子是否瞬时过热；
- 自旋与晶格热化先后顺序；
- 是否越过 `TR` 或 `TN`。

### 序参量 / 代理量图（Order parameter / proxy）

显示：

- `m(t)`
- `eta(t)`
- `S_m(t)`

可用于观察：

- 磁有序是否塌陷；
- 重取向自由度是否响应；
- 用于拟合的光谱代理量如何变化。

### 有效耦合图（Effective couplings）

显示：

- `G_el_eff`
- `G_es_eff`
- `G_sl_eff`

可用于判断：

- 参数变化后能量耦合是否过强或过弱；
- `TR/TN` 附近是否出现预期增强。

### 功率流图（Power flow）

显示：

- `P_el`
- `P_es`
- `P_sl`

定义上：

- `P_es > 0` 表示能量从电子流向自旋；
- `P_el > 0` 表示能量从电子流向晶格；
- `P_sl > 0` 表示能量从自旋流向晶格。

---

## CSV 数据格式说明

### 必须有表头

`load_csv_auto()` 使用 `numpy.genfromtxt(..., names=True)` 读取数据，因此 CSV **必须包含首行表头**。

### 时间列自动识别

程序会自动识别以下时间列名之一：

- `tps`
- `t_ps`
- `time_ps`
- `t(ps)`
- `time(ps)`
- `time`
- `t`

### 温度列自动识别

电子温度 `Te` 可识别：

- `tek`
- `te_k`
- `te`
- `te(k)`
- `temp_e`
- `electron_temp`

自旋温度 `Ts` 可识别：

- `tsk`
- `ts_k`
- `ts`
- `ts(k)`
- `temp_s`
- `spin_temp`
- `tspin`

晶格温度 `Tl` 可识别：

- `tlk`
- `tl_k`
- `tl`
- `tl(k)`
- `temp_l`
- `lattice_temp`

谱权重 `S` 可识别：

- `s`
- `sw`
- `spec`
- `weight`
- `spectral_weight`
- `intensity`
- `amp`

### 时间单位自动判断

如果时间数据的绝对值最大值大于 `1e-6`，程序会把它视为 **皮秒（ps）** 并自动换算成秒；否则视为已经是秒。

### 重复时间点处理

如果 CSV 中存在重复时间点，程序会：

1. 先排序；
2. 再对重复时间点的数据取平均。

### 推荐示例

```csv
time_ps,Te,spectral_weight
-2,4.1,1.00
0,20.3,0.95
1,17.8,0.91
5,12.2,0.85
20,6.7,0.93
```

---

## 拟合功能说明

项目当前提供三种拟合策略。

### 1. `Fit Te`

只拟合电子温度曲线，默认拟合参数为：

- `G_el0`
- `G_es0`
- `S_scale`
- `t0_pulse`

适用场景：

- 你主要关心电子升温和冷却过程；
- `S(t)` 数据不可靠或未测量。

### 2. `Fit S`

只拟合 `S(t)`，默认拟合参数为：

- `tau_m0`
- `tau_m_crit_amp`
- `nu`
- `S_offset`
- `S_amp`
- `S_power`
- `t0_pulse`

适用场景：

- 你更关心磁序塌陷与恢复；
- 希望通过代理量反推出序参量动力学。

### 3. `Fit Te + S`

联合拟合时，默认拟合参数包括：

- `G_el0`
- `G_es0`
- `G_sl0`
- `G_es_m_power`
- `G_sl_TR_boost`
- `G_sl_TN_boost`
- `tau_l_sink`
- `S_scale`
- `tau_m0`
- `tau_m_crit_amp`
- `nu`
- `S_offset`
- `S_amp`
- `S_power`
- `t0_pulse`

适用场景：

- 想同时利用温度与光谱信息约束模型；
- 希望更稳地确定能量耦合与磁动力学参数。

### 拟合方法特点

拟合内部使用：

- `scipy.optimize.least_squares`
- `method="trf"`
- `loss="soft_l1"`

此外，部分正参数会自动采用 **log 参数化**，避免在搜索中出现负值或跨数量级不稳定。

---

## 作为 Python 模块调用

如果你不想使用 GUI，也可以直接在脚本中调用。

### 最小仿真示例

```python
import numpy as np
from config import default_params
from solver import NdSb3TM

p = default_params()
model = NdSb3TM(p)

t = np.linspace(-2e-12, 100e-12, 1200)
sim = model.simulate_aligned(t)

print(sim.keys())
print(sim["Te"].max(), sim["m"][-1])
```

### 读取实验数据

```python
from data_io import load_csv_auto

t, Te, Ts, Tl, S, names, unit = load_csv_auto("example.csv")
```

### 执行拟合

```python
from config import default_params
from data_io import fit_params

p0 = default_params()
fit_keys = ["G_el0", "G_es0", "S_scale", "t0_pulse"]
p_best, res = fit_params(t, Te, None, p0, fit_keys)
```

---

## 关键参数调参建议

### 如果 `Te` 峰值太低

优先检查：

- `fluence_multiplier` 是否太小；
- `S_scale` 是否太小；
- `G_el0` / `G_es0` 是否过大导致电子太快泄能；
- `delta_opt` 是否过大导致单位体积吸收能量偏小。

### 如果 `Ts` 升温过慢

优先检查：

- `G_es0` 是否过小；
- `G_es_m_power` 是否让磁有序消失时通道过于关闭；
- `Cs_scale` 或 magnon/CEF 参数是否使 `Cs` 过大。

### 如果 `m` 恢复太慢

优先检查：

- `tau_m0`
- `tau_m_crit_amp`
- `nu`
- `TN`

### 如果 `eta` 行为异常

优先检查：

- `eta_enable`
- `eta_mode`
- `Gamma_eta`
- `a_eta0`、`b_eta`、`c_eta`
- `g_m2eta2`
- `eta_clip`

### 如果 `S(t)` 与实验量纲不一致

优先检查：

- `S_offset`
- `S_amp`
- `S_power`

因为这里的 `S(t)` 本质上是代理量映射，而不是严格从能带结构直接算出的光谱强度。

---

## 注意事项与当前限制

1. **当前 headless 模式只是演示，不是完整 CLI 工具。**
   如果需要批量任务，建议写 Python 脚本直接调用模块。

2. **`G_el_Tpow` 目前在求解器里没有真正参与 `G_el_eff()` 的计算。**
   虽然 GUI 与参数中有该项，但当前版本的电子-晶格通道仍主要按常数基值处理。

3. **拟合依赖初值。**
   对强非线性模型来说，初值不同可能收敛到不同局部解。建议先手动调到与实验趋势大致一致后再拟合。

4. **`S(t)` 是经验代理量。**
   它适合描述与磁有序强相关的实验 observable，但不应直接等同于完整 ARPES 谱函数。

5. **LSWT / magnon 热容计算在高分辨网格下会更慢。**
   如果拟合速度太慢，可适当降低 `magnon_grid` 或切换到 `AT3` 模型做初步扫描。

---

## 推荐使用流程

对于第一次使用本项目，建议按下面顺序：

1. 直接运行 `python main.py`。
2. 使用默认参数先点击 `Simulate`，熟悉四张图的变化。
3. 载入包含 `time + Te (+ S)` 的实验 CSV。
4. 先手动调整让曲线大致接近。
5. 先做 `Fit Te` 或 `Fit S` 单目标拟合。
6. 再尝试 `Fit Te + S` 联合拟合。
7. 检查状态栏中的能量闭合误差是否过大。
8. 如需沉淀结果，再把最终参数写入 `config.py` 或单独保存到你的分析脚本中。

---

## 后续可扩展方向

如果你准备继续扩展该项目，比较自然的方向包括：

- 增加真正的命令行参数接口；
- 增加参数保存 / 载入 JSON 功能；
- 支持更多实验 observable 的联合拟合；
- 引入更真实的非平衡分布或多泵浦脉冲；
- 对 `S(t)` 使用更复杂的物理映射而不只是 `m^p`。

---

## 一句话总结

这是一个围绕 **NdSb 超快动力学** 构建的、带 GUI 的 **扩展三温模型模拟 + 拟合工具**：既能做交互式物理试算，也能读取实验 CSV 进行参数反演，适合做泵浦-探测或相关时间分辨实验的数据解释。
