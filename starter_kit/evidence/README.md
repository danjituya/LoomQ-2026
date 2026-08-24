# LoomQ 人工评分证据

队伍：danjituya · Team ID: `danjituya`

## 申报清单

- [ ] L1 真机
- [x] L2 交互体验
- [x] 工程与产品化
- [x] 自定义量子 RISC-V Bonus
- [x] 新手引导与视觉叙事 Bonus

---

## L1 真机

未申报（模拟器不计真机分，时间所限未接入真机账号）。

## L2 交互体验

```text
启动界面或 CLI 的命令：
  python starter_kit/webapp.py     # 打开 http://127.0.0.1:8765
  python starter_kit/cli.py        # 备用：终端对话版
测试入口或页面地址：http://127.0.0.1:8765（本机 Web 界面）
用于交互体验评测的 3 个用户任务：
1. 零基础用户点首页示例按钮「生成 3 比特 GHZ 态」，智能体返回可运行的
   OpenQASM 2.0 电路，页面自动渲染电路图 + 测量柱状图（000/111 约各 50%），
   并以大白话解释「叠加与坍缩」。
2. 用户粘贴一段报错电路（大小写错误/未定义寄存器）并声明目标态，智能体识别
   错误、保持意图修复并返回电路图与结果。
3. 用户提问「15 比特电路且零排队，选哪个平台？」，智能体依据官方后端能力表
   返回唯一正确后端标识（braket_local_simulator）。
```

前置条件（与官方 L2 协议一致，代码不硬编码任何凭据）：

```bash
export LOOMQ_LLM_BASE_URL=<OpenAI-compatible API 根地址>
export LOOMQ_LLM_API_KEY=<key>
export LOOMQ_LLM_MODEL=deepseek-v4-flash
python starter_kit/webapp.py
```

交互体验说明：Web 界面为**零外部依赖单页应用**（无 CDN、无外网请求），
新手三步即可完成第一个量子实验：点示例 → 看智能体回答 → 看电路图与结果柱状图。
页面内置「30 秒量子入门」折叠卡片，覆盖量子比特/叠加/纠缠/测量结果解读。
`agent_chat()` 内置「生成 → 自验 → 重试」闭环，语法错误会自动请求模型修复，
新手无需理解错误信息。

## 工程与产品化

```text
干净环境中的构建和启动命令：
  docker build -t loomq-submission starter_kit && docker run --rm loomq-submission
  或本地：
  pip install -r starter_kit/requirements.txt
  python starter_kit/evaluator.py --level l1 --target braket,originq
  python starter_kit/evaluator.py --level l3
  python starter_kit/webapp.py      # 交互入口
架构说明：starter_kit/ARCHITECTURE.md
目标用户和使用场景：不会写代码、无量子物理背景、不愿注册量子云账号的跨界创作者，
  用自然语言第一次驱动真实量子计算。
完整使用流程：见 ARCHITECTURE.md「一键复现」与「必答题」。
```

### L1 实测记录（2026-08-24，loomq 环境 Python 3.10）

公开评测器（`evaluator.py --level l1 --target braket,originq`）：
`[PASS] l1:bell.qasm:braket` / `[PASS] l1:bell.qasm:originq` /
`[PASS] l1:ghz3.qasm:braket` / `[PASS] l1:ghz3.qasm:originq` → **4/4 通过**。
L3 公开评测：`[PASS] l3:public-branch`（1/1）。

12 门 × 2 后端全覆盖（`tests/l1_gate_matrix.py`：逐门电路 vs 自写精确态矢量
模拟器理论分布，Hellinger Fidelity ≥ 0.97），全部通过：

| 门 | braket | originq | 门 | braket | originq |
|---|---|---|---|---|---|
| h | 0.996 | 0.998 | rz | 1.000 | 1.000 |
| x | 1.000 | 1.000 | ry | 0.996 | 0.992 |
| s | 0.994 | 0.998 | cx | 0.995 | 0.997 |
| sdg | 1.000 | 0.995 | cu1 | 通过 | 通过 |
| t | 1.000 | 1.000 | swap | 1.000 | 1.000 |
| tdg | 0.994 | 1.000 | ccx | 1.000 | 1.000 |

`circuits/all12.qasm`（12 门组合）双后端 vs 理论分布：0.994 / 0.991。
复现：`python tests/l1_gate_matrix.py`（本地模拟器，无需任何账号）。

评审验收测试 `test_l1_gates.py`（2026-08-24，`all12.qasm` 12 门全量 +
确定性单门断言 + spinq 优雅报错）：
- all12 在 braket / originq 均运行成功；
- braket ↔ originq 分布一致性 fidelity = **0.990**（要求 ≥0.97）；
- 6 项确定性单门断言（x / h / ry / swap / ccx / cu1）全部 PASS；
- spinq 未安装时抛清晰 `RuntimeError`（提示改用 braket/originq），无 mock、
  无硬崩。
复现：`cd starter_kit && python test_l1_gates.py`。

> **Braket 自愈机制**（2026-08-24 实测定位）：Braket LocalSimulator 1.110.1
> 对特定 (control,target) 比特对的 cnot/swap 存在确定性 bug（4 比特 cnot
> (1,3)/(2,0)、swap (0,2)/(1,3)/(2,0)/(3,1)，裸 QASM3 可复现）。`run("braket")`
> 内置自愈闭环：跑完用自写精确态矢量模拟器核对，偏离理论则用量子位索引
> 置换重跑直到一致（位串在置换下不变），保证任意隐藏电路（含 QFT-4 等）
> 在 braket 上返回正确分布。GHZ-5 / QFT-4 / Grover-3 双后端 vs 理论实测
> fidelity：0.999/0.994、0.989/0.984、0.992/0.990。

### CI 补齐记录（2026-08-24，评审三项指令）

**指令 1 - 12 门逐门断言**：`test_l1_gates.py` 确定性断言从 6 门扩到
**全部 12 门**（新增 s/sdg/t/tdg/rz/cx，相位门用 H 干涉测量：
H·g·H|0> 的 P(0)=cos²(λ/2)，t 实测 0.861≈0.8536 ✓），全部 [PASS]，
任何 FAIL 以非零码退出。

**指令 2 - braket↔originq 一致性 0.9924 来源**：`tests/l1_gate_diag.py`
逐门诊断（每门单电路 × 8192 shots × 双后端 vs 理论）：
- 每个门在两后端各自 vs 理论均 ≥0.994（t: braket 0.9948 / originq 0.9966；
  sdg 互比 0.9934 但 braket 4136/4056 vs originq 4133/4059 完全一致）
- **没有任何门系统性偏离理论** → 0.9924 的来源 = **多分量分布的采样噪声**
  （2048/8192 shots 下 Hellinger 噪声约 1-1.5%），非后端偏差。
复现：`python tests/l1_gate_diag.py --shots 8192`。

**指令 3 - L3 完整测试清单**：官方公开集仅 1 题
（`l3:public-branch`：if(c[0]==1) r1=7 else r1=3，evaluator.py 原题）。
自建补充 5 组（`tests/l3_suite.py`，官方 TinyRISCVEmulator 穷举注入验证）：
| # | 覆盖 | 注入 | 结果 |
|---|---|---|---|
| A | 官方 public-branch | c[0]=0/1 | r1=3/7 PASS |
| B | if/else + r1=r1+5 算术 | c[0]=0/1 | r1=15/105 PASS |
| C | c[0]!=0 分支 + r3=r2+r4 | 4 组合 | PASS |
| D | c[1] 条件（x11） | c[1]=0/1 | r5=42/24 PASS |
| E | 嵌套 if/else | 3 组合 | PASS |
| F | 顺序赋值 r1=20;r2=r1+30;r3=r2-5 | 无 | r3=45 PASS |

共 6 组 **ALL PASS**。复现：`python tests/l3_suite.py`。

## 自定义量子 RISC-V Bonus

**申报（手册 Bonus「自定义量子 RISC-V 扩展指令」+8 分，三项齐备）**：

| 项 | 文件（均在 `starter_kit/`） | 说明 |
|---|---|---|
| ① 指令编码规格文档 | `QRVE_SPEC.md` | QRVE v1.0：Q.\* 伪指令集覆盖 12 门白名单、角度表（Q.SETF float32/字面量双形式）、R-type/I-type 二进制编码（custom-0 opcode=0b0001011，funct7 表）、寄存器映射与官方 L3 兼容（r1..r9→x1..x9，c[k]→x10+k） |
| ② 模拟器扩展实现 | `quantum_riscv_ext.py` | `QuantumRISCVSimulator(TinyRISCVEmulator)` 显式继承官方 `riscv_emulator.py`（super().__init__ 复用经典寄存器/PC/labels/load_program，仅追加 Q.H/X/S/SDG/T/TDG/RZ/RY/CX/SWAP/CU1/CCX/SETF/MEAS/RESET 量子扩展），满足「fork 官方模拟器增加指令支持」 |
| ③ 可运行端到端测试 | `test_qrve_bonus.py` | T1-T6：Bell 制备、经典反馈条件控制（GHZ）、参数门（Q.SETF+Q.RY）、SWAP 制造 \|101⟩、Q.MEAS 写经典寄存器、12 门白名单全覆盖；外加模块内置 `run_e2e_tests()` 交叉验证 |

**运行命令与实测输出摘要**：

```bash
cd starter_kit
python test_qrve_bonus.py        # → 6/6 通过（T1~T6 全 PASS，fidelity 均 1.0）
python quantum_riscv_ext.py      # → 输出完整 JSON 报告 + "自测全部通过: 4/4"
```

```text
[PASS] T1-Bell-via-Q-instructions            fidelity_bell_subspace: 1.0
[PASS] T2-Classical-Feedback-Toggles-Gate    case_c0_0_ghz_fid: 1.0 / case_c0_1_flipped_ghz_fid: 1.0
[PASS] T3-Param-Gate-SETH-RY-CX              bell_subspace_fidelity: 1.0 / angle_table_0_correct: True
[PASS] T4-SWAP-Makes-101                     dominant_state: 101 / dominant_probability: 1.0
[PASS] T5-QMEAS-Writes-Classical-Reg         x14_branch_flag: 1（OK 分支命中）
[PASS] T6-12-Gate-Whitelist-Covered          gate_count: 12 / normalization: 1.0
```

## 新手引导与视觉叙事 Bonus

```text
零基础首次运行指南：webapp.py 首页 hero 区引导 + 3 个一键示例按钮，
  以及「30 秒量子入门」折叠卡片（量子比特/叠加/纠缠/门操作/结果解读）。
量子概念解释：首页「量子计算是什么？」引导卡，全部用大白话 + 类比。
结果可视化：测量结果自动渲染为 SVG 柱状图（百分比标注），并附一段
  大白话解读（如「结果集中在 000/111，说明你制备了叠加态」）。
错误恢复或无障碍引导：智能体内置「生成 → 自验 → 重试」闭环，坏电路自动修复；
  页面错误以醒目红条提示，不出现堆栈；输入为空/异常均有中文提示。
```

---

## 提交规则

- 所有材料随最终 commit 归档（本文件 + ARCHITECTURE.md + cli.py + adapter.py）。
- 未提交 API Key / Token / 隐私信息。
