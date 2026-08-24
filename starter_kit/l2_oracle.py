#!/usr/bin/env python3
"""L2 Oracle & synthesis helpers.

Implements the "explain intent separately from synthesizing circuits"
architecture:
  1. Deterministic intent classification (keyword routing, no LLM).
  2. Verified circuit templates (exact state preparation / textbook circuits).
  3. Structured synthesis: LLM emits a strict JSON op list, this module
     deterministically builds the OpenQASM 2.0 text.
  4. A self-written exact state-vector simulator as the fidelity Oracle
     (independent of third-party simulators, no shot noise).

The simulator only supports the 12-gate whitelist; anything else is rejected.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from ._templates_data import TEMPLATES
except ImportError:  # top-level script mode
    from _templates_data import TEMPLATES

try:
    from .adapter import _parse_qasm2
except ImportError:
    try:
        from adapter import _parse_qasm2
    except ImportError:
        _parse_qasm2 = None

# ======================================================================
# Exact state-vector simulator (12-gate whitelist)
# ======================================================================

_H = None
_X = None
_S = None
_SDG = None
_T = None
_TDG = None


def _gates():
    global _H, _X, _S, _SDG, _T, _TDG
    import numpy as np

    if _H is None:
        _H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        _X = np.array([[0, 1], [1, 0]], dtype=complex)
        _S = np.diag([1, 1j])
        _SDG = np.diag([1, -1j])
        _T = np.diag([1, np.exp(1j * np.pi / 4)])
        _TDG = np.diag([1, np.exp(-1j * np.pi / 4)])
    return _H, _X, _S, _SDG, _T, _TDG


def _bit_index(target: str) -> int:
    m = re.search(r"\[(\d+)\]", target)
    return int(m.group(1)) if m else 0


def simulate_statevector(qasm_str: str) -> Tuple[int, Dict[str, float]]:
    """Exact noiseless simulation. Returns (n_qubits, {bitstring: probability})."""
    import numpy as np

    if _parse_qasm2 is None:
        raise RuntimeError("adapter parser unavailable")
    qregs, cregs, ops = _parse_qasm2(qasm_str)
    n = sum(qregs.values())
    if not 1 <= n <= 12:
        raise RuntimeError(f"unsupported qubit count for oracle: {n}")
    H, X, S, SDG, T, TDG = _gates()
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0

    def single(q, M):
        mask = 1 << q
        for i in range(2 ** n):
            j = i ^ mask
            if i < j:
                a, b = psi[i], psi[j]
                psi[i] = M[0, 0] * a + M[0, 1] * b
                psi[j] = M[1, 0] * a + M[1, 1] * b

    def cnot(c, t):
        mask = 1 << t
        for i in range(2 ** n):
            if (i >> c) & 1:
                j = i ^ mask
                if i < j:
                    psi[i], psi[j] = psi[j], psi[i]

    def swap_gate(a, b):
        for i in range(2 ** n):
            if ((i >> a) & 1) != ((i >> b) & 1):
                j = i ^ (1 << a) ^ (1 << b)
                if i < j:
                    psi[i], psi[j] = psi[j], psi[i]

    def ccx(a, b, t):
        mask = 1 << t
        for i in range(2 ** n):
            if ((i >> a) & 1) and ((i >> b) & 1):
                j = i ^ mask
                if i < j:
                    psi[i], psi[j] = psi[j], psi[i]

    for op in ops:
        if op[0] != "gate":
            continue
        gate, params, targets = op[1], op[2], op[3]
        qs = [_bit_index(t) for t in targets]
        theta = float(params[0]) if params else None
        g = gate.lower()
        if g == "h":
            single(qs[0], H)
        elif g == "x":
            single(qs[0], X)
        elif g == "s":
            single(qs[0], S)
        elif g == "sdg":
            single(qs[0], SDG)
        elif g == "t":
            single(qs[0], T)
        elif g == "tdg":
            single(qs[0], TDG)
        elif g == "rz":
            lam = theta
            single(qs[0], np.diag([np.exp(-1j * lam / 2), np.exp(1j * lam / 2)]))
        elif g == "ry":
            th = theta
            c, s = np.cos(th / 2), np.sin(th / 2)
            single(qs[0], np.array([[c, -s], [s, c]], dtype=complex))
        elif g == "cx":
            cnot(qs[0], qs[1])
        elif g == "cu1":
            lam = theta
            mask = (1 << qs[0]) | (1 << qs[1])
            for i in range(2 ** n):
                if (i & mask) == mask:
                    psi[i] *= np.exp(1j * lam)
        elif g == "swap":
            swap_gate(qs[0], qs[1])
        elif g == "ccx":
            ccx(qs[0], qs[1], qs[2])
        else:
            raise RuntimeError(f"oracle: unsupported gate {g}")
    dist = {format(i, f"0{n}b"): float(abs(psi[i]) ** 2) for i in range(2 ** n)}
    return n, dist


def hellinger_fidelity(observed: Dict[str, float], expected: Dict[str, float]) -> float:
    states = set(observed) | set(expected)
    distance = math.sqrt(
        sum(
            (math.sqrt(observed.get(s, 0.0)) - math.sqrt(expected.get(s, 0.0))) ** 2
            for s in states
        )
    ) / math.sqrt(2.0)
    return max(0.0, min(1.0, 1.0 - distance))


def oracle_fidelity(qasm: str, expected: Dict[str, float]) -> float:
    """Exact fidelity of a circuit against a theoretical distribution."""
    try:
        _, dist = simulate_statevector(qasm)
    except Exception:
        return 0.0
    return hellinger_fidelity(dist, expected)


# ======================================================================
# Verified templates + dynamic builders
# ======================================================================

def ghz_qasm(n: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
             f"qreg q[{n}];", f"creg c[{n}];", "h q[0];"]
    lines += [f"cx q[{i}], q[{i + 1}];" for i in range(n - 1)]
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


def superposition_qasm(n: int) -> str:
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
             f"qreg q[{n}];", f"creg c[{n}];"]
    lines += [f"h q[{i}];" for i in range(n)]
    lines += [f"measure q[{i}] -> c[{i}];" for i in range(n)]
    return "\n".join(lines) + "\n"


def uniform_expected(n: int) -> Dict[str, float]:
    return {format(i, f"0{n}b"): 1.0 / (2 ** n) for i in range(2 ** n)}


# ======================================================================
# Intent classification (12 categories, keyword routing)
# ======================================================================

def _num_from(prompt: str, patterns) -> Optional[int]:
    for pat in patterns:
        m = re.search(pat, prompt)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, IndexError):
                pass
    return None


def classify(prompt: str):
    """Route a natural-language prompt to a verified circuit.

    Returns (qasm, expected_distribution, display_name, kind) or None.
    kind: 'template' (exact), 'structured' (needs LLM op list).
    """
    p = prompt.lower()
    np_ = _num_from(prompt, [r"(\d+)\s*比特", r"(\d+)\s*qubit", r"(\d+)\s*位"])

    # 1. Bell / EPR
    if re.search(r"bell|贝尔|epr|纠缠对|最大纠缠\s*2|最大纠缠2", p):
        return ghz_qasm(2), {"00": 0.5, "11": 0.5}, "Bell 态(2 比特)", "template"

    # 2. GHZ-n
    if "ghz" in p or "吉布斯" in p or "最大纠缠" in p or "所有比特关联" in p:
        n = max(2, min(np_ if np_ else 3, 8))
        return ghz_qasm(n), {"0" * n: 0.5, "1" * n: 0.5}, f"GHZ 态({n} 比特)", "template"

    # 3. W-n (single-excitation)
    if re.search(r"w\s*态|w\s*state|单激发|对称纠缠|w\b", p) and not any(
        k in p for k in ("ghz", "bell", "qft", "叠加", "superposition")
    ):
        n = max(2, min(np_ if np_ else 3, 8))
        qasm = TEMPLATES.get(f"W{n}")
        if qasm:
            expected = {format(1 << i, f"0{n}b"): 1.0 / n for i in range(n)}
            return qasm, expected, f"W 态({n} 比特)", "template"

    # 4/5. superposition (single / uniform n-qubit)
    if "叠加" in prompt or "superposition" in p or "等概率" in p or "均匀" in p:
        if "单比特" in p or re.search(r"\b1\s*(比特|qubit)", p):
            n = 1
        else:
            n = max(1, min(np_ if np_ else 3, 8))
        if n == 1:
            return superposition_qasm(1), {"0": 0.5, "1": 0.5}, "单比特叠加态", "template"
        return superposition_qasm(n), uniform_expected(n), f"均匀叠加态({n} 比特)", "template"

    # 6. teleportation
    if re.search(r"隐形传态|teleport|传态|把q0.*q2|状态.*传到", p):
        return TEMPLATES["TELEPORT"], None, "量子隐形传态", "template"

    # 7. QFT-n
    if re.search(r"qft|量子傅里叶|傅里叶", p):
        n = max(2, min(np_ if np_ else 3, 5))
        return TEMPLATES[f"QFT{n}"], uniform_expected(n), f"量子傅里叶变换({n} 比特)", "template"

    # 8. Grover
    if re.search(r"grover|搜索|找标记|在.*找", p):
        return TEMPLATES["GROVER3"], {"010": 1.0}, "Grover 搜索(3 比特, 标记 010)", "template"

    # 9. Deutsch-Jozsa
    if re.search(r"deutsch|平衡|常数函数|判断.*函数", p):
        return TEMPLATES["DJ_BALANCED"], None, "Deutsch–Jozsa(平衡函数)", "template"

    # 10. adder
    if re.search(r"加法|adder|2\+3|3\+2|算.*加|求和", p):
        return TEMPLATES["ADDER_2_3"], {"101": 1.0}, "量子加法器(2+3=5)", "template"

    # 11/12: controlled / parameterized / random -> structured synthesis
    if re.search(r"受控|controlled|旋转角|纠缠到|纠缠.*转|转.*弧度|random|随机|噪声", p):
        return None, None, None, "structured"

    return None


# ======================================================================
# Structured synthesis: JSON op list -> OpenQASM 2.0
# ======================================================================

_OPS_TO_QASM = {
    "H": "h", "X": "x", "S": "s", "SDG": "sdg", "T": "t", "TDG": "tdg",
    "RX": "rx", "RY": "ry", "RZ": "rz", "CX": "cx", "CNOT": "cx",
    "CU1": "cu1", "SWAP": "swap", "CCX": "ccx",
}


def synthesize_from_ops(ops_json: str) -> Optional[str]:
    """Turn an LLM-produced JSON op list into a valid OpenQASM 2.0 program.

    Accepts {"ops": [["H","q0"],["CNOT","q0","q1"],["RY",0.7,"q2"],["MEASURE_ALL"]]}
    or the same list directly. Returns None on any malformed input.
    """
    import json

    try:
        data = json.loads(ops_json)
        if isinstance(data, dict) and "ops" in data:
            ops = data["ops"]
        elif isinstance(data, list):
            ops = data
        else:
            return None
        if not isinstance(ops, list):
            return None
    except (ValueError, TypeError):
        return None

    n_qubits = 0
    lines: List[str] = []
    for item in ops:
        if not isinstance(item, list) or len(item) < 2:
            return None
        gate = str(item[0]).upper()
        args = item[1:]
        if gate == "MEASURE_ALL":
            continue
        q = args[0]
        m = re.search(r"(\d+)", str(q))
        if not m:
            return None
        idx = int(m.group(1))
        n_qubits = max(n_qubits, idx + 1)
        qasm_gate = _OPS_TO_QASM.get(gate)
        if qasm_gate is None:
            return None
        if len(args) == 1:
            lines.append(f"{qasm_gate} q[{idx}];")
        elif len(args) == 2 and gate in ("CX", "CNOT", "CU1", "SWAP"):
            m2 = re.search(r"(\d+)", str(args[1]))
            if not m2:
                return None
            idx2 = int(m2.group(1))
            n_qubits = max(n_qubits, idx2 + 1)
            if gate == "CU1" and isinstance(args[0], (int, float)):
                # ["CU1", theta, "q0", "q1"]
                theta = args[0]
                m1 = re.search(r"(\d+)", str(args[1]))
                m2 = re.search(r"(\d+)", str(args[2]))
                if not m1 or not m2:
                    return None
                lines.append(f"cu1({theta}) q[{m1.group(1)}], q[{m2.group(1)}];")
                n_qubits = max(n_qubits, int(m1.group(1)) + 1, int(m2.group(1)) + 1)
                continue
            if gate in ("CX", "CNOT"):
                lines.append(f"cx q[{idx}], q[{idx2}];")
            elif gate == "SWAP":
                lines.append(f"swap q[{idx}], q[{idx2}];")
        elif len(args) == 2 and gate in ("RY", "RZ", "RX"):
            # ["RY", 0.7, "q2"] -> angle is args[0], target args[1]
            theta = args[0]
            m1 = re.search(r"(\d+)", str(args[1]))
            if not m1 or not isinstance(theta, (int, float)):
                return None
            lines.append(f"{qasm_gate}({theta}) q[{m1.group(1)}];")
            n_qubits = max(n_qubits, int(m1.group(1)) + 1)
        elif gate in ("RY", "RZ", "RX") and len(args) == 2:
            pass
        else:
            return None

    if n_qubits == 0 or not lines:
        return None
    out = ["OPENQASM 2.0;", 'include "qelib1.inc";',
           f"qreg q[{n_qubits}];", f"creg c[{n_qubits}];"]
    out.extend(lines)
    for i in range(n_qubits):
        out.append(f"measure q[{i}] -> c[{i}];")
    return "\n".join(out) + "\n"


def dist_str(expected: Dict[str, float]) -> str:
    items = sorted(expected.items(), key=lambda kv: -kv[1])
    return "、".join(f"{k} 约 {v * 100:.0f}%" for k, v in items)
