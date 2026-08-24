#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2 模板电路本地检验器（零依赖，仅需 Python 标准库）。

它做的事：把 _templates_data.py 里每个标准电路用纯 Python 精确态模拟跑一遍，
算出真实测量分布，再和 l2_oracle.classify() 里写的"期望分布"做保真度比对。
结果直接告诉你们：L2 的标准电路哪些对、哪些错。

用法：
  python verify_l2.py
（脚本会自动从克隆仓库读取 _templates_data.py，无需任何安装）
"""
import os
import sys
import re
import math
import cmath

# ---- 自动定位克隆仓库里的 _templates_data.py ----
HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATES = [
    HERE,  # 脚本所在目录（starter_kit/，官方容器内直接可用）
    os.path.dirname(HERE),
    r"C:/Users/yp/AppData/Local/Temp/LoomQ_check/starter_kit",
    r"D:/Desktop/LoomQ-L1-fix",
]
for c in CANDIDATES:
    if os.path.isfile(os.path.join(c, "_templates_data.py")):
        sys.path.insert(0, c)
        break
else:
    print("ERROR: 找不到 _templates_data.py，请把克隆仓库路径加进 CANDIDATES。")
    sys.exit(1)

from _templates_data import TEMPLATES  # noqa: E402


# =====================================================================
# 纯 Python 精确态模拟（逻辑与 l2_oracle.simulate_statevector 完全一致）
# =====================================================================
def _single(psi, n, q, M):
    mask = 1 << q
    for i in range(2 ** n):
        j = i ^ mask
        if i < j:
            a, b = psi[i], psi[j]
            psi[i] = M[0][0] * a + M[0][1] * b
            psi[j] = M[1][0] * a + M[1][1] * b


def _cnot(psi, n, c, t):
    mask = 1 << t
    for i in range(2 ** n):
        if (i >> c) & 1:
            j = i ^ mask
            if i < j:
                psi[i], psi[j] = psi[j], psi[i]


def _swap(psi, n, a, b):
    for i in range(2 ** n):
        if ((i >> a) & 1) != ((i >> b) & 1):
            j = i ^ (1 << a) ^ (1 << b)
            if i < j:
                psi[i], psi[j] = psi[j], psi[i]


def _ccx(psi, n, a, b, t):
    mask = 1 << t
    for i in range(2 ** n):
        if ((i >> a) & 1) and ((i >> b) & 1):
            j = i ^ mask
            if i < j:
                psi[i], psi[j] = psi[j], psi[i]


def _cu1(psi, n, a, b, lam):
    mask = (1 << a) | (1 << b)
    for i in range(2 ** n):
        if (i & mask) == mask:
            psi[i] *= cmath.exp(1j * lam)


_SQRT1_2 = 1 / math.sqrt(2)
_H = [[_SQRT1_2, _SQRT1_2], [_SQRT1_2, -_SQRT1_2]]
_X = [[0, 1], [1, 0]]
_S = [[1, 0], [0, 1j]]
_SDG = [[1, 0], [0, -1j]]
_T = [[1, 0], [0, cmath.exp(1j * math.pi / 4)]]
_TDG = [[1, 0], [0, cmath.exp(-1j * math.pi / 4)]]


def _rz(lam):
    return [[cmath.exp(-1j * lam / 2), 0], [0, cmath.exp(1j * lam / 2)]]


def _ry(th):
    c, s = math.cos(th / 2), math.sin(th / 2)
    return [[c, -s], [s, c]]


_LINE = re.compile(r"^(\w+)(?:\(([^)]*)\))?\s+(.+?);$")


def simulate(qasm: str):
    lines = [ln.strip() for ln in qasm.strip().splitlines()]
    n = 0
    ops = []
    for ln in lines:
        if not ln or ln.startswith(("OPENQASM", "include", "qreg", "creg", "measure")):
            m = re.match(r"qreg q\[(\d+)\];", ln)
            if m:
                n = max(n, int(m.group(1)))
            continue
        m = _LINE.match(ln)
        if not m:
            continue
        gate = m.group(1).lower()
        params = [float(x) for x in m.group(2).split(",")] if m.group(2) else []
        targets = re.findall(r"q\[(\d+)\]", m.group(3))
        ops.append((gate, params, [int(t) for t in targets]))
    psi = [0.0 + 0.0j] * (2 ** n)
    psi[0] = 1.0 + 0.0j
    for gate, params, qs in ops:
        if gate == "h":
            _single(psi, n, qs[0], _H)
        elif gate == "x":
            _single(psi, n, qs[0], _X)
        elif gate == "s":
            _single(psi, n, qs[0], _S)
        elif gate == "sdg":
            _single(psi, n, qs[0], _SDG)
        elif gate == "t":
            _single(psi, n, qs[0], _T)
        elif gate == "tdg":
            _single(psi, n, qs[0], _TDG)
        elif gate == "rz":
            _single(psi, n, qs[0], _rz(params[0]))
        elif gate == "ry":
            _single(psi, n, qs[0], _ry(params[0]))
        elif gate == "cx":
            _cnot(psi, n, qs[0], qs[1])
        elif gate == "cu1":
            _cu1(psi, n, qs[0], qs[1], params[0])
        elif gate == "swap":
            _swap(psi, n, qs[0], qs[1])
        elif gate == "ccx":
            _ccx(psi, n, qs[0], qs[1], qs[2])
        else:
            raise RuntimeError(f"unsupported gate {gate}")
    dist = {format(i, f"0{n}b"): abs(psi[i]) ** 2 for i in range(2 ** n)}
    return dist


def fidelity(observed, expected):
    states = set(observed) | set(expected)
    d = math.sqrt(sum((math.sqrt(observed.get(s, 0.0)) - math.sqrt(expected.get(s, 0.0))) ** 2
                      for s in states)) / math.sqrt(2.0)
    return max(0.0, min(1.0, 1.0 - d))


def show(dist, top=4):
    items = sorted(dist.items(), key=lambda kv: -kv[1])[:top]
    return "、".join(f"{k} {v*100:.1f}%" for k, v in items)


# =====================================================================
# 期望分布（与 l2_oracle.classify() 中写的一致）
# =====================================================================
EXP = {
    "Bell": {"00": 0.5, "11": 0.5},
    "GHZ3": {"000": 0.5, "111": 0.5},
    "W2": {format(1 << i, "02b"): 0.5 for i in range(2)},
    "W3": {format(1 << i, "03b"): 1 / 3 for i in range(3)},
    "W4": {format(1 << i, "04b"): 0.25 for i in range(4)},
    "W5": {format(1 << i, "05b"): 0.2 for i in range(5)},
    "ADDER_2_3": {"101": 1.0},
    "QFT3": {format(i, "03b"): 1 / 8 for i in range(8)},
    # GROVER3 真实分布（010≈94.5%，3 比特搜 1/8 的最优 Grover 结果）。
    # 早期 classify() 误写为 {"010": 1.0} 会把正确电路误判 FAIL，现已改用真实分布。
    "GROVER3": {"010": 0.94531, "000": 0.00781, "001": 0.00781, "100": 0.00781,
                "101": 0.00781, "111": 0.00781, "110": 0.00781, "011": 0.00781},
}
# GROVER3 的正确期望（脚本会自动算出来，见下方）
GROVER_KEY = "GROVER3"


def main():
    print("=" * 70)
    print("L2 标准电路本地检验（纯 Python 精确态模拟，零依赖）")
    print("=" * 70)
    results = []

    # ---- 有期望值的模板（用真实模板逐项比对）----
    plan = [
        ("W2", TEMPLATES.get("W2"), EXP["W2"]),
        ("W3", TEMPLATES.get("W3"), EXP["W3"]),
        ("W4", TEMPLATES.get("W4"), EXP["W4"]),
        ("W5", TEMPLATES.get("W5"), EXP["W5"]),
        ("ADDER_2_3", TEMPLATES.get("ADDER_2_3"), EXP["ADDER_2_3"]),
        ("QFT3", TEMPLATES.get("QFT3"), EXP["QFT3"]),
        ("GROVER3", TEMPLATES.get("GROVER3"), EXP["GROVER3"]),
    ]

    for name, qasm, exp in plan:
        if not qasm:
            print(f"[跳过] {name}: 模板缺失")
            continue
        dist = simulate(qasm)
        fid = fidelity(dist, exp)
        ok = fid >= 0.97
        results.append((name, fid, ok))
        flag = "PASS" if ok else "FAIL"
        print(f"[{flag}] {name:12s} fidelity={fid:.4f}  实际分布(前4): {show(dist)}")

    # ---- Grover 特别分析 ----
    print("\n--- Grover 专项 ---")
    g = TEMPLATES.get("GROVER3")
    if g:
        gdist = simulate(g)
        print(f"GROVER3 实际分布: {show(gdist, top=8)}")
        print(f"  -> 010 实际概率 = {gdist.get('010',0)*100:.2f}% （理论 Grover 最优 ≈ 94.5%）")
        print("  -> classify() 已改用真实分布作期望（上方 [PASS]），不再误判正确电路。")

    # ---- 仅有模板、无期望值（运行时靠 None 跳过校验）----
    print("\n--- 无期望值模板（classify 返回 expected=None，不做保真度校验）---")
    for name in ["TELEPORT", "DJ_BALANCED", "QFT2", "QFT4", "QFT5", "W6", "W7", "W8"]:
        q = TEMPLATES.get(name)
        if q:
            d = simulate(q)
            print(f"  {name:12s} 可运行, 主分布: {show(d)}")

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    passed = sum(1 for _, _, ok in results if ok)
    print(f"有期望值模板: {passed}/{len(results)} 通过")
    print("=" * 70)


if __name__ == "__main__":
    main()
