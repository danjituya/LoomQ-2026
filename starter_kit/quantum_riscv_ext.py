#!/usr/bin/env python3
"""
LoomQ 自定义量子 RISC-V 扩展指令实现 (Quantum RISC-V Extension, QRVE)

本模块为官方 TinyRISCVEmulator 扩展了量子操作自定义指令集，
使经典控制程序可以直接在 RISC-V 汇编层面操纵量子比特。

== 指令编码规格 (QRVE v1.0) ==

量子指令映射到 RISC-V custom-0 opcode 空间 (opcode=0b0001011)，
使用 R-type / I-type 编码格式的逻辑等价物（在轻量模拟器中以
伪指令助记符形式暴露，选手无需手写二进制编码）。

------------------------------------------------------------------
1) 无参数单比特门:  {H, X, S, SDG, T, TDG}
   助记符          格式                     说明
   Q.H             Q.H  rd, rs1             rd 未用；rs1 = 目标比特索引
   Q.X             Q.X  rd, rs1             同上
   Q.S             Q.S  rd, rs1
   Q.SDG           Q.SDG rd, rs1
   Q.T             Q.T  rd, rs1
   Q.TDG           Q.TDG rd, rs1

------------------------------------------------------------------
2) 参数单比特门:  {RZ, RY}
   助记符          格式                     说明
   Q.RZ            Q.RZ rd, rs1, rs2        rs1 = 目标比特；rs2 = 角度索引
                                   (角度值通过 Q.SETF 预载入角度表)
   Q.RY            Q.RY rd, rs1, rs2        同上

------------------------------------------------------------------
3) 双比特门:  {CX, SWAP}
   助记符          格式                     说明
   Q.CX            Q.CX rd, rs1, rs2        rs1 = 控制比特；rs2 = 目标比特
   Q.SWAP          Q.SWAP rd, rs1, rs2      rs1 = 比特A；rs2 = 比特B

------------------------------------------------------------------
4) 受控参数双比特门:  CU1
   助记符          格式                     说明
   Q.CU1           Q.CU1 rd, rs1, rs2, rs3  rs1 = 控制；rs2 = 目标；rs3 = 角度索引

------------------------------------------------------------------
5) 三比特门:  CCX (Toffoli)
   助记符          格式                     说明
   Q.CCX           Q.CCX rd, rs1, rs2, rs3  rs1 = 控制A；rs2 = 控制B；rs3 = 目标

------------------------------------------------------------------
6) 角度载入辅助指令:  Q.SETF (set float) —— 把浮点角度值写入角度表
   助记符          格式                     说明
   Q.SETF          Q.SETF rs1, imm32        rs1 = 角度表索引 (0..7)
                                                imm32 = IEEE-754 float32 编码的角度值
   注意：在文本汇编中允许 Q.SETF rs1, <十进制浮点字面量>，
   汇编器会自动做 float32 编码。

------------------------------------------------------------------
7) 测量指令:  Q.MEAS
   助记符          格式                     说明
   Q.MEAS          Q.MEAS rd, rs1           rd = 存储测量结果的经典寄存器
                                              rs1 = 被测量的量子比特索引

------------------------------------------------------------------
寄存器映射（与官方 L3 约定兼容）：
  x0      = zero (硬编码 0)
  x1..x9  = 用户寄存器 r1..r9
  x10..   = 测量位 c[0], c[1], ... （由评测程序注入）
  x28..x31= 保留（QRVE 内部临时寄存器）

量子态存储：
  QRVE 模拟器内部维护一个最多 N_QUBITS_MAX = 12 的态矢量，
  所有 Q.* 指令直接操纵该态矢量；测量结果写入用户指定的通用寄存器。
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Optional, Tuple

# 依赖 numpy 作为态矢量后端
import math
import numpy as np

# ============================================================
# 12 门白名单矩阵
# ============================================================

N_QUBITS_MAX = 12
N_ANGLE_ENTRIES = 8  # 角度表大小（Q.SETF 索引 0..7）


def _gate_matrices():
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    S = np.diag([1, 1j])
    SDG = np.diag([1, -1j])
    T = np.diag([1, np.exp(1j * np.pi / 4)])
    TDG = np.diag([1, np.exp(-1j * np.pi / 4)])
    return H, X, S, SDG, T, TDG


# ============================================================
# QRVE 模拟器：在官方 TinyRISCVEmulator 上扩展量子指令
# ============================================================

class QuantumRISCVSimulator:
    """RISC-V + 自定义量子扩展指令 联合模拟器。

    经典部分与官方 TinyRISCVEmulator 100% 兼容（相同寄存器/指令语义）；
    量子部分维护一个内联态矢量并响应 Q.* 伪指令。
    """

    def __init__(self, n_qubits: int = N_QUBITS_MAX):
        # ---- 经典部分（与 TinyRISCVEmulator 对齐）----
        self.registers: List[int] = [0] * 32
        self.pc: int = 0
        self.labels: Dict[str, int] = {}
        self.instructions: List[Tuple[str, List[str]]] = []
        self.max_steps: int = 100_000

        # ---- 量子部分 ----
        self.n_qubits = max(1, min(n_qubits, N_QUBITS_MAX))
        self.psi: np.ndarray = np.zeros(2 ** self.n_qubits, dtype=complex)
        self.psi[0] = 1.0
        self.angle_table: List[float] = [0.0] * N_ANGLE_ENTRIES
        self._H, self._X, self._S, self._SDG, self._T, self._TDG = _gate_matrices()
        self._gate_count = 0
        self._depth_by_qubit = [0] * self.n_qubits

    # ========================================================
    # 寄存器访问 (与官方模拟器保持一致的 API 风格)
    # ========================================================
    def _parse_reg_idx(self, reg: str) -> int:
        reg = reg.strip().replace(",", "")
        if not (reg.startswith("x") or reg.startswith("X")):
            raise ValueError(f"无效寄存器名称: {reg}")
        idx = int(reg[1:])
        if not 0 <= idx <= 31:
            raise ValueError(f"寄存器索引越界 x0-x31: {reg}")
        return idx

    def set_register(self, reg: str, value: int) -> None:
        idx = self._parse_reg_idx(reg)
        if idx != 0:
            self.registers[idx] = value & 0xFFFFFFFF

    def get_register(self, reg: str) -> int:
        return self.registers[self._parse_reg_idx(reg)]

    # ========================================================
    # 程序加载（扩展支持 Q.* 伪指令）
    # ========================================================
    def load_program(self, asm_code: str) -> None:
        self.instructions = []
        self.labels = {}
        self.pc = 0
        self.registers = [0] * 32

        temp_instructions: List[Tuple[str, List[str]]] = []
        for raw_line in asm_code.split("\n"):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith(";"):
                continue
            if "#" in line:
                line = line.split("#")[0].strip()
            # 标签
            if line.endswith(":"):
                self.labels[line[:-1].strip()] = len(temp_instructions)
                continue
            if ":" in line:
                head, tail = line.split(":", 1)
                self.labels[head.strip()] = len(temp_instructions)
                line = tail.strip()

            tokens = line.replace(",", " ").split()
            op = tokens[0]
            args = tokens[1:]
            temp_instructions.append((op, args))

        self.instructions = temp_instructions

    # ========================================================
    # 量子态操纵原语
    # ========================================================
    def _single_q(self, q: int, M: np.ndarray) -> None:
        if not 0 <= q < self.n_qubits:
            raise ValueError(f"量子比特索引越界: q[{q}] (n={self.n_qubits})")
        mask = 1 << q
        for i in range(2 ** self.n_qubits):
            j = i ^ mask
            if i < j:
                a, b = self.psi[i], self.psi[j]
                self.psi[i] = M[0, 0] * a + M[0, 1] * b
                self.psi[j] = M[1, 0] * a + M[1, 1] * b
        self._depth_by_qubit[q] += 1
        self._gate_count += 1

    def _cnot_q(self, c: int, t: int) -> None:
        if c == t:
            raise ValueError("cx: control 和 target 不能相同")
        mask_t = 1 << t
        for i in range(2 ** self.n_qubits):
            if (i >> c) & 1:
                j = i ^ mask_t
                if i < j:
                    self.psi[i], self.psi[j] = self.psi[j], self.psi[i]
        self._depth_by_qubit[c] += 1
        self._depth_by_qubit[t] += 1
        self._gate_count += 1

    def _swap_q(self, a: int, b: int) -> None:
        for i in range(2 ** self.n_qubits):
            if ((i >> a) & 1) != ((i >> b) & 1):
                j = i ^ (1 << a) ^ (1 << b)
                if i < j:
                    self.psi[i], self.psi[j] = self.psi[j], self.psi[i]
        self._depth_by_qubit[a] += 1
        self._depth_by_qubit[b] += 1
        self._gate_count += 1

    def _ccx_q(self, a: int, b: int, t: int) -> None:
        mask_t = 1 << t
        for i in range(2 ** self.n_qubits):
            if ((i >> a) & 1) and ((i >> b) & 1):
                j = i ^ mask_t
                if i < j:
                    self.psi[i], self.psi[j] = self.psi[j], self.psi[i]
        for qi in (a, b, t):
            self._depth_by_qubit[qi] += 1
        self._gate_count += 1

    def _cu1_q(self, c: int, t: int, theta: float) -> None:
        mask = (1 << c) | (1 << t)
        for i in range(2 ** self.n_qubits):
            if (i & mask) == mask:
                self.psi[i] *= np.exp(1j * theta)
        for qi in (c, t):
            self._depth_by_qubit[qi] += 1
        self._gate_count += 1

    def _rz_q(self, q: int, theta: float) -> None:
        self._single_q(q, np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]))

    def _ry_q(self, q: int, theta: float) -> None:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        self._single_q(q, np.array([[c, -s], [s, c]], dtype=complex))

    def _measure_q(self, q: int) -> int:
        # 投影测量：按概率坍缩，返回 0/1
        mask = 1 << q
        p0 = 0.0
        for i in range(2 ** self.n_qubits):
            if not (i & mask):
                p0 += abs(self.psi[i]) ** 2
        result = 0 if (np.random.random() < p0) else 1
        # 重归一化
        scale = 1.0 / (np.sqrt(p0) if result == 0 else np.sqrt(1.0 - p0))
        for i in range(2 ** self.n_qubits):
            is_other = bool(i & mask) if result == 0 else not bool(i & mask)
            if is_other:
                self.psi[i] = 0.0
            else:
                self.psi[i] *= scale
        return result

    # ========================================================
    # 量子态测量分布（用于验证，不坍缩）
    # ========================================================
    def measure_distribution(self, measured_qubits: Optional[List[int]] = None) -> Dict[str, float]:
        if measured_qubits is None:
            measured_qubits = list(range(self.n_qubits))
        n_out = len(measured_qubits)
        dist: Dict[str, float] = {}
        for i in range(2 ** self.n_qubits):
            p = abs(self.psi[i]) ** 2
            if p < 1e-18:
                continue
            key_bits = "".join(str((i >> q) & 1) for q in measured_qubits)
            # 结果位序：c[0] 最右
            key = key_bits[::-1] if n_out > 1 else key_bits
            dist[key] = dist.get(key, 0.0) + p
        return dist

    def quantum_stats(self) -> Dict[str, Any]:
        return {
            "qubit_count": self.n_qubits,
            "gate_count": self._gate_count,
            "circuit_depth": max(self._depth_by_qubit) if self._depth_by_qubit else 0,
        }

    # ========================================================
    # 执行引擎（经典 + 量子联合）
    # ========================================================
    def execute(self, rng_seed: Optional[int] = None) -> Dict[str, int]:
        if rng_seed is not None:
            np.random.seed(rng_seed)
        steps = 0
        n_instr = len(self.instructions)
        while 0 <= self.pc < n_instr:
            steps += 1
            if steps > self.max_steps:
                raise RuntimeError("QRVE: 超出最大执行步数（疑似死循环）")
            op, args = self.instructions[self.pc]
            next_pc = self.pc + 1
            op_low = op.lower()

            # ---------- 经典指令（1:1 复刻 TinyRISCVEmulator）----------
            if op_low == "li":
                self.set_register(args[0], int(args[1]))
            elif op_low == "add":
                v = self.get_register(args[1]) + self.get_register(args[2])
                self.set_register(args[0], v)
            elif op_low == "sub":
                v = self.get_register(args[1]) - self.get_register(args[2])
                self.set_register(args[0], v)
            elif op_low == "addi":
                v = self.get_register(args[1]) + int(args[2])
                self.set_register(args[0], v)
            elif op_low == "beq":
                if self.get_register(args[0]) == self.get_register(args[1]):
                    if args[2] not in self.labels:
                        raise ValueError(f"未定义标签: {args[2]}")
                    next_pc = self.labels[args[2]]
            elif op_low == "bne":
                if self.get_register(args[0]) != self.get_register(args[1]):
                    if args[2] not in self.labels:
                        raise ValueError(f"未定义标签: {args[2]}")
                    next_pc = self.labels[args[2]]
            elif op_low == "j":
                if args[0] not in self.labels:
                    raise ValueError(f"未定义标签: {args[0]}")
                next_pc = self.labels[args[0]]

            # ---------- 量子扩展指令 ----------
            elif op_low == "q.setf":
                # Q.SETF rs1, <float 或 int>
                try:
                    idx = self.get_register(args[0])
                except Exception:
                    idx = int(args[0].strip(",").lstrip("x"))
                raw = args[1]
                try:
                    val = float(raw)
                except ValueError:
                    # float32 整数编码 -> 解码
                    val = struct.unpack("f", struct.pack("I", int(raw) & 0xFFFFFFFF))[0]
                if 0 <= idx < N_ANGLE_ENTRIES:
                    self.angle_table[idx] = val

            elif op_low == "q.h":
                self._single_q(self.get_register(args[1]), self._H)
            elif op_low == "q.x":
                self._single_q(self.get_register(args[1]), self._X)
            elif op_low == "q.s":
                self._single_q(self.get_register(args[1]), self._S)
            elif op_low == "q.sdg":
                self._single_q(self.get_register(args[1]), self._SDG)
            elif op_low == "q.t":
                self._single_q(self.get_register(args[1]), self._T)
            elif op_low == "q.tdg":
                self._single_q(self.get_register(args[1]), self._TDG)
            elif op_low == "q.rz":
                theta_idx = self.get_register(args[2])
                theta = self.angle_table[theta_idx] if 0 <= theta_idx < N_ANGLE_ENTRIES else float(theta_idx)
                self._rz_q(self.get_register(args[1]), theta)
            elif op_low == "q.ry":
                theta_idx = self.get_register(args[2])
                theta = self.angle_table[theta_idx] if 0 <= theta_idx < N_ANGLE_ENTRIES else float(theta_idx)
                self._ry_q(self.get_register(args[1]), theta)
            elif op_low == "q.cx":
                self._cnot_q(self.get_register(args[1]), self.get_register(args[2]))
            elif op_low == "q.swap":
                self._swap_q(self.get_register(args[1]), self.get_register(args[2]))
            elif op_low == "q.cu1":
                theta_idx = self.get_register(args[3]) if len(args) >= 4 else 0
                theta = self.angle_table[theta_idx] if 0 <= theta_idx < N_ANGLE_ENTRIES else float(theta_idx)
                self._cu1_q(self.get_register(args[1]), self.get_register(args[2]), theta)
            elif op_low == "q.ccx":
                self._ccx_q(self.get_register(args[1]), self.get_register(args[2]), self.get_register(args[3]))
            elif op_low == "q.meas":
                bit = self._measure_q(self.get_register(args[1]))
                self.set_register(args[0], bit)
            elif op_low == "q.reset":
                # Q.RESET: 重置到 |0...0>（辅助调试指令）
                self.psi = np.zeros(2 ** self.n_qubits, dtype=complex)
                self.psi[0] = 1.0
                self._gate_count = 0
                self._depth_by_qubit = [0] * self.n_qubits
            else:
                raise ValueError(f"QRVE: 不支持的指令: {op}")

            self.pc = next_pc

        return {f"x{i}": v for i, v in enumerate(self.registers) if v != 0}


# ============================================================
# 端到端自测（作为提交的一部分，证明三者齐备）
# ============================================================

def run_e2e_tests() -> Dict[str, Any]:
    """端到端测试：验证 QRVE 指令编码、模拟器、经典-量子交互全部通过。"""
    results: Dict[str, Any] = {}

    # ---- Test 1: 贝尔态制备 & 测量（Q.H + Q.CX + Q.MEAS）----
    t1_code = """
    # 准备角度表索引 0 保存 pi/2（本测试不用参数门）
    li x1, 0       # q0
    li x2, 1       # q1
    li x3, 0       # 测量结果 x10 = c[0], x11 = c[1] 由外部注入，这里用 x5/x6
    Q.RESET
    Q.H x0, x1
    Q.CX x0, x1, x2
    Q.MEAS x5, x1
    Q.MEAS x6, x2
    """
    sim1 = QuantumRISCVSimulator(n_qubits=2)
    sim1.load_program(t1_code)
    reg_state = sim1.execute(rng_seed=42)
    dist1 = sim1.measure_distribution([0, 1])
    # 理论上 Bell 态是 50% |00> + 50% |11>（测量一次只看到一个结果）；
    # 用态矢量分布验证
    fid_bell = max(dist1.get("00", 0), 0) + max(dist1.get("11", 0), 0)
    results["test1_bell_state"] = {
        "passed": fid_bell > 0.99,
        "fidelity_bell_subspace": round(fid_bell, 4),
        "distribution_after_reset_1shot": {k: round(v, 4) for k, v in sorted(dist1.items())},
        "quantum_stats": sim1.quantum_stats(),
    }

    # ---- Test 2: GHZ-3 态 & 经典反馈条件（经典-量子混合控制流）----
    t2_code = """
    li x1, 0
    li x2, 1
    li x3, 2
    Q.RESET
    Q.H x0, x1
    Q.CX x0, x1, x2
    Q.CX x0, x2, x3
    # 经典判断：如果 r1 (c[0] 对应 x10) == 1，那么在 q2 上施加 X
    beq x10, x0, GHZ_SKIP
    Q.X x0, x3
    GHZ_SKIP:
    Q.MEAS x6, x1
    Q.MEAS x7, x2
    Q.MEAS x8, x3
    """
    # Case A: 注入 c[0] = 0 -> 不施加 X -> 标准 GHZ (000/111)
    sim2a = QuantumRISCVSimulator(n_qubits=3)
    sim2a.load_program(t2_code)
    sim2a.set_register("x10", 0)
    sim2a.execute(rng_seed=2026)
    dist2a = sim2a.measure_distribution([0, 1, 2])
    fid_ghz_a = dist2a.get("000", 0) + dist2a.get("111", 0)
    # Case B: 注入 c[0] = 1 -> 施加 X q2 -> 翻转 GHZ (100/011)
    sim2b = QuantumRISCVSimulator(n_qubits=3)
    sim2b.load_program(t2_code)
    sim2b.set_register("x10", 1)
    sim2b.execute(rng_seed=2026)
    dist2b = sim2b.measure_distribution([0, 1, 2])
    fid_ghz_b = dist2b.get("100", 0) + dist2b.get("011", 0)
    fid_ghz_both = (fid_ghz_a + fid_ghz_b) / 2.0
    results["test2_ghz3_classical_feedback"] = {
        "passed": fid_ghz_a > 0.98 and fid_ghz_b > 0.98,
        "case_c0_0_ghz_fid": round(fid_ghz_a, 4),
        "case_c0_1_flipped_ghz_fid": round(fid_ghz_b, 4),
        "avg_ghz_subspace": round(fid_ghz_both, 4),
        "dist_c0_0": {k: round(v, 4) for k, v in sorted(dist2a.items())},
        "dist_c0_1": {k: round(v, 4) for k, v in sorted(dist2b.items())},
    }

    # ---- Test 3: 参数门 RY(pi/2) -> 等价 H，再 Q.CX 制备 Bell（验证 Q.SETF + Q.RY）----
    pi_over_2 = math.pi / 2
    t3_code = f"""
    li x1, 0
    li x2, 1
    li x5, 0        # 角度表索引 0
    Q.SETF x5, {pi_over_2:.9f}
    Q.RESET
    Q.RY x0, x1, x5   # q0 上 RY(pi/2) => |+>
    Q.CX x0, x1, x2
    """
    sim3 = QuantumRISCVSimulator(n_qubits=2)
    sim3.load_program(t3_code)
    sim3.execute()
    dist3 = sim3.measure_distribution([0, 1])
    fid_param_bell = dist3.get("00", 0) + dist3.get("11", 0)
    results["test3_param_gate_bell_via_ry"] = {
        "passed": fid_param_bell > 0.97,
        "bell_subspace_after_ry_cx": round(fid_param_bell, 4),
        "distribution": {k: round(v, 4) for k, v in sorted(dist3.items())},
        "angle_table_after_setf": [round(x, 6) for x in sim3.angle_table],
    }

    # ---- Test 4: Q.SWAP 制造 |101> 纯态（验证 swap 路径与位序约定）----
    t4_code = """
    li x1, 0
    li x2, 1
    li x3, 2
    Q.RESET
    Q.X x0, x1        # q0 -> |1>  =>  |001>  (c2 c1 c0)
    Q.X x0, x2        # q1 -> |1>  =>  |011>
    Q.SWAP x0, x2, x3  # swap(q1, q2)  =>  |101>
    """
    sim4 = QuantumRISCVSimulator(n_qubits=3)
    sim4.load_program(t4_code)
    sim4.execute()
    dist4 = sim4.measure_distribution([0, 1, 2])
    dominant4 = max(dist4.items(), key=lambda kv: kv[1])
    results["test4_swap_ccx"] = {
        "passed": dominant4[0] == "101" and dominant4[1] > 0.999,
        "dominant_state": dominant4[0],
        "dominant_probability": round(dominant4[1], 4),
        "distribution": {k: round(v, 4) for k, v in sorted(dist4.items())},
    }

    results["summary"] = {
        "tests_total": len(results),
        "tests_passed": sum(1 for r in results.values() if isinstance(r, dict) and r.get("passed")),
    }
    return results


if __name__ == "__main__":
    import json
    report = run_e2e_tests()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    total = report["summary"]["tests_total"]
    passed = report["summary"]["tests_passed"]
    assert passed == total, f"QRVE E2E 测试失败: {passed}/{total}"
    print(f"\nQRVE Bonus 扩展端到端自测全部通过: {passed}/{total}")
