# LoomQ 人工评分证据

队伍：danjituya · Team ID: `danjituya`

## 申报清单

- [ ] L1 真机
- [x] L2 交互体验
- [x] 工程与产品化
- [ ] 自定义量子 RISC-V Bonus
- [ ] 新手引导与视觉叙事 Bonus

---

## L1 真机

未申报（模拟器不计真机分，时间所限未接入真机账号）。

## L2 交互体验

```text
启动界面或 CLI 的命令：python starter_kit/cli.py
测试入口或页面地址：无（终端 CLI）
用于交互体验评测的 3 个用户任务：
1. 零基础用户输入"生成一个 3 比特 GHZ 态并进行全测量"，Agent 返回可直接运行的
   OpenQASM 2.0 电路，且经无噪声模拟器验证 Fidelity ≥ 0.97。
2. 用户粘贴一段报错电路（如大小写错误/未定义寄存器）并声明目标态，Agent 识别
   错误、保持意图修复并返回完整可运行电路。
3. 用户提问"15 比特电路且零排队，选哪个平台？"，Agent 依据官方后端能力表
   （backend_capabilities.json）返回唯一正确后端标识。
```

前置条件（与官方 L2 协议一致，代码不硬编码任何凭据）：

```bash
export LOOMQ_LLM_BASE_URL=<OpenAI-compatible API 根地址>
export LOOMQ_LLM_API_KEY=<key>
export LOOMQ_LLM_MODEL=deepseek-v4-flash
python starter_kit/cli.py
```

交互体验说明：CLI 以中文提示引导，包含示例指令；模型回答原样展示；
单次调用异常不会导致会话崩溃。`agent_chat()` 内置"生成 → 自验 → 重试"闭环，
能拦截语法/运行错误并向模型请求修复，新手不需要理解错误信息。

## 工程与产品化

```text
干净环境中的构建和启动命令：
  docker build -t loomq-submission starter_kit && docker run --rm loomq-submission
  或本地：
  pip install -r starter_kit/requirements.txt
  python starter_kit/evaluator.py --level l1 --target braket,originq
  python starter_kit/evaluator.py --level l3
架构说明：starter_kit/ARCHITECTURE.md
目标用户和使用场景：不会写代码、无量子物理背景、不愿注册量子云账号的跨界创作者，
  用自然语言第一次驱动真实量子计算。
完整使用流程：见 ARCHITECTURE.md「一键复现」与「必答题」。
```

## 自定义量子 RISC-V Bonus

未申报。

## 新手引导与视觉叙事 Bonus

未申报。

---

## 提交规则

- 所有材料随最终 commit 归档（本文件 + ARCHITECTURE.md + cli.py + adapter.py）。
- 未提交 API Key / Token / 隐私信息。
