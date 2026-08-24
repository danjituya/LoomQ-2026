#!/usr/bin/env python3
"""二轮补丁自测：CRY/CRX/CRZ 等价分解 + 中文纠缠 GHZ 兜底。

指令 1：synthesize_from_ops 对受控旋转做白名单门等价分解（数学验证：
        ctrl=1 时 target 应用 rx/ry/rz(θ)，ctrl=0 时 target 不变）。
指令 2：4 条中文纠缠说法命中模板（GHZ/Bell）+ fidelity >= 0.97。
回归：GHZ3/Bell/W3/Grover(标记 110)/选后端 仍正确。
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starter_kit"))
from l2_oracle import (classify, synthesize_from_ops, simulate_statevector,  # noqa: E402
                       oracle_fidelity, hellinger_fidelity)

FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    FAILS += 0 if ok else 1
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# ============ 指令 1：受控旋转等价分解 ============
print("=" * 72)
print("指令 1：CRY/CRX/CRZ 等价分解")
print("=" * 72)

# 1a. 验收输入：原 FAIL 用例
qasm = synthesize_from_ops('{"ops":[["RX",0.5236,"q0"],["CRY",0.5236,"q1","q2"]]}')
check("原 FAIL 用例可合成 QASM", qasm is not None, f"{len(qasm)} 字符" if qasm else "")
if qasm:
    n, dist = simulate_statevector(qasm)
    check("合成电路可模拟运行", n == 3 and abs(sum(dist.values()) - 1.0) < 1e-6)
    # 展开后不得出现受控旋转门（白名单校验）
    check("展开无白名单外门",
          not any(g in qasm for g in ("cry(", "crx(", "crz(")),
          "含 h/ry/rz/cx 即视为白名单内")
    print("  展开电路:", " ".join(l for l in qasm.splitlines() if "measure" not in l and l not in (
        "OPENQASM 2.0;", 'include "qelib1.inc";', "qreg q[3];", "creg c[3];")))

# 1b. 数学验证：cry(θ) ctrl=1 → ry(θ) 作用于 target；ctrl=0 → 不变
THETA = 1.0
# ctrl=0（q0=0, tgt=q1=0）→ target 不变，全 0
q0 = synthesize_from_ops('{"ops":[["CRY",%s,"q0","q1"]]}' % THETA)
_, d0 = simulate_statevector(q0)
check("cry ctrl=0 → target 不变", abs(d0.get("00", 0) - 1.0) < 1e-6, f"P(00)={d0.get('00',0):.4f}")
# ctrl=1（q0=1, tgt=q1=0）→ target 旋转 ry(θ)|0>
q1 = synthesize_from_ops('{"ops":[["X","q0"],["CRY",%s,"q0","q1"]]}' % THETA)
_, d1 = simulate_statevector(q1)
p_expect0 = math.cos(THETA / 2) ** 2
# 位序：bitstring 最右字符 = q[0]（LSB）。ctrl=q0=1、tgt=q1=0 → "01"
check("cry ctrl=1 → ry(θ) 作用于 target",
      abs(d1.get("01", 0) - p_expect0) < 1e-6,
      f"P(01)={d1.get('01',0):.4f} 期望 {p_expect0:.4f}")

# 1c. crx(θ) ctrl=1 → rx(θ)（分布同 ry，但实现用 h 共轭路径）
q2 = synthesize_from_ops('{"ops":[["X","q0"],["CRX",%s,"q0","q1"]]}' % THETA)
_, d2 = simulate_statevector(q2)
check("crx ctrl=1 → rx(θ) 作用于 target",
      abs(d2.get("01", 0) - p_expect0) < 1e-6,
      f"P(01)={d2.get('01',0):.4f} 期望 {p_expect0:.4f}")
# crx ctrl=0 → 不变
q3 = synthesize_from_ops('{"ops":[["CRX",%s,"q0","q1"]]}' % THETA)
_, d3 = simulate_statevector(q3)
check("crx ctrl=0 → target 不变", abs(d3.get("00", 0) - 1.0) < 1e-6)

# 1d. crz(θ) ctrl=1 → rz(θ)（相位门，测量分布不变；验证 tgt=|1> 保持 |1>）
q4 = synthesize_from_ops('{"ops":[["X","q0"],["X","q1"],["CRZ",%s,"q0","q1"]]}' % THETA)
_, d4 = simulate_statevector(q4)
check("crz ctrl=1 → rz(θ) 作用于 target(相位,测量不变)",
      abs(d4.get("11", 0) - 1.0) < 1e-6, f"P(11)={d4.get('11',0):.4f}")

# 1e. 非法输入拒绝
bad = synthesize_from_ops('{"ops":[["CRY","q0","q1"]]}')  # 缺参数
check("CRY 缺参数 → 拒绝(None)", bad is None)

# ============ 指令 2：中文纠缠兜底 ============
print("=" * 72)
print("指令 2：中文纠缠说法 → 模板 + fidelity 校验")
print("=" * 72)
zh_cases = [
    ("帮我把 3 个量子比特做成互相纠缠的状态，然后每个都量一下", "GHZ 态(3 比特)"),
    ("搞一个 5 个量子比特全部纠缠的态并测量", "GHZ 态(5 比特)"),
    ("做个纠缠", "GHZ 态(3 比特)"),
    ("量子超密编码需要的那个双比特纠缠态", "Bell 态(2 比特)"),
    ("帮我把 4 个比特纠缠在一起", "GHZ 态(4 比特)"),
]
for prompt, want in zh_cases:
    r = classify(prompt)
    if r is None or r[3] != "template":
        check(prompt[:24], False, f"未命中模板: {r}")
        continue
    qasm, expected, name, kind = r
    fid = oracle_fidelity(qasm, expected)
    ok = (want in name) and fid >= 0.97
    check(prompt[:24], ok, f"-> {name} fidelity={fid:.4f}")

# ============ 回归 ============
print("=" * 72)
print("回归：GHZ3 / Bell / W3 / Grover(110) / 选后端")
print("=" * 72)
reg = [
    ("ghz 3 个量子比特", "GHZ 态(3 比特)"),
    ("制备一个 bell 态", "Bell 态(2 比特)"),
    ("帮我构造 W 态", "W 态(3 比特)"),
    ("Grover 搜索目标标记 110", "Grover 搜索(3 比特, 标记 110)"),
    ("选哪个后端跑", None),  # backend_select
]
for prompt, want in reg:
    r = classify(prompt)
    if want is None:
        check(f"{prompt[:18]} -> backend_select", r is not None and r[3] == "backend_select",
              str(r[3]) if r else "None")
        continue
    ok = r is not None and r[3] == "template" and want in r[2]
    check(f"{prompt[:18]}", ok, f"-> {r[2] if r else 'None'}")

print("=" * 72)
print(f"结论: {'ALL PASS' if FAILS == 0 else f'{FAILS} 项 FAIL'}")
sys.exit(0 if FAILS == 0 else 1)
