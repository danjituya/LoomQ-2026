# LoomQ 人工评分证据

队伍：danjituya · Team ID: `danjituya`

## 申报清单

- [ ] L1 真机
- [x] L2 交互体验
- [x] 工程与产品化
- [ ] 自定义量子 RISC-V Bonus
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

12 门 × 2 后端全覆盖（`tests/test_l1_all12.py`：逐门电路 vs 自写精确态矢量
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
复现：`python tests/test_l1_all12.py`（本地模拟器，无需任何账号）。

## 自定义量子 RISC-V Bonus

未申报。

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
