#!/usr/bin/env python3
"""LoomQ submission adapter contract v1.0.

Implements:
  L1: transpile() + run() for spinq / originq / braket
  L2: agent_chat()  (OpenAI-compatible LLM via LOOMQ_LLM_* env vars)
  L3: compile_hybrid()  (Hybrid-QASM -> quantum ops + RISC-V assembly)
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

SUPPORTED_TARGETS = ("spinq", "originq", "braket")

# OriginIR gate name mapping (per target_ir_contract.md)
_GATE_MAP_ORIGIN = {
    "h": "H", "x": "X", "s": "S", "sdg": "SDAG", "t": "T", "tdg": "TDAG",
    "rz": "RZ", "ry": "RY", "cx": "CNOT", "cu1": "CU1", "swap": "SWAP",
    "ccx": "TOFFOLI",
}

# ======================================================================
# OpenQASM 2.0 parsing helpers
# ======================================================================

def _clean_lines(qasm_str: str) -> List[str]:
    """Strip comments and blank lines."""
    out = []
    for line in qasm_str.splitlines():
        line = line.split("//")[0].strip()
        if line:
            out.append(line)
    return out


def _parse_qasm2(qasm_str: str) -> Tuple[Dict[str, int], Dict[str, int], List[tuple]]:
    """Parse OpenQASM 2.0 -> (qregs, cregs, ops).

    op forms:
      ("measure", qname, qi_or_None, cname, ci_or_None)
      ("gate", gate, [params], [targets])
    """
    qregs: Dict[str, int] = {}
    cregs: Dict[str, int] = {}
    ops: List[tuple] = []
    for line in _clean_lines(qasm_str):
        line = line.rstrip(";").strip()
        if not line or line.startswith("OPENQASM") or line.startswith("include"):
            continue
        m = re.match(r"qreg\s+([A-Za-z_]\w*)\[(\d+)\]", line)
        if m:
            qregs[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"creg\s+([A-Za-z_]\w*)\[(\d+)\]", line)
        if m:
            cregs[m.group(1)] = int(m.group(2))
            continue
        m = re.match(
            r"measure\s+([A-Za-z_]\w*)(?:\[(\d+)\])?\s*->\s*([A-Za-z_]\w*)(?:\[(\d+)\])?",
            line,
        )
        if m:
            ops.append(("measure", m.group(1), m.group(2), m.group(3), m.group(4)))
            continue
        m = re.match(r"([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*(.+)", line)
        if m:
            gate = m.group(1)
            params = [p.strip() for p in m.group(2).split(",")] if m.group(2) else []
            targets = [t.strip() for t in m.group(3).split(",") if t.strip()]
            ops.append(("gate", gate, params, targets))
    return qregs, cregs, ops


def _measure_pairs(ops, cregs) -> List[Tuple[str, str]]:
    """Expand measurement ops into explicit (qubit, cbit) index pairs."""
    pairs: List[Tuple[str, str]] = []
    for op in ops:
        if op[0] != "measure":
            continue
        _, q, qi, c, ci = op
        if qi is not None and ci is not None:
            pairs.append((f"{q}[{qi}]", f"{c}[{ci}]"))
        else:
            size = cregs.get(c, 0)
            for i in range(size):
                pairs.append((f"{q}[{i}]", f"{c}[{i}]"))
    return pairs


def _normalize_qasm2(qasm_str: str) -> str:
    """Normalize OpenQASM 2.0: strip comments, expand whole-register measures."""
    qregs, cregs, ops = _parse_qasm2(qasm_str)
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    for name, size in qregs.items():
        lines.append(f"qreg {name}[{size}];")
    for name, size in cregs.items():
        lines.append(f"creg {name}[{size}];")
    for op in ops:
        if op[0] == "measure":
            _, q, qi, c, ci = op
            if qi is not None and ci is not None:
                lines.append(f"measure {q}[{qi}] -> {c}[{ci}];")
            else:
                size = cregs.get(c, 0)
                for i in range(size):
                    lines.append(f"measure {q}[{i}] -> {c}[{i}];")
        else:
            _, gate, params, targets = op
            if params:
                lines.append(f"{gate}({', '.join(params)}) {', '.join(targets)};")
            else:
                lines.append(f"{gate} {', '.join(targets)};")
    return "\n".join(lines) + "\n"


# ======================================================================
# Transpile: OpenQASM 2.0 -> target native IR
# ======================================================================

def _to_qasm3(qasm2: str) -> str:
    """Convert OpenQASM 2.0 -> OpenQASM 3.0 (Braket)."""
    qregs, cregs, ops = _parse_qasm2(qasm2)
    lines = ["OPENQASM 3.0;", 'include "stdgates.inc";']
    for name, size in qregs.items():
        lines.append(f"qubit[{size}] {name};")
    for name, size in cregs.items():
        lines.append(f"bit[{size}] {name};")
    for op in ops:
        if op[0] == "measure":
            _, q, qi, c, ci = op
            if qi is not None:
                lines.append(f"{c}[{ci}] = measure {q}[{qi}];")
            else:
                lines.append(f"{c} = measure {q};")
        else:
            _, gate, params, targets = op
            g = "cnot" if gate == "cx" else ("cp" if gate == "cu1" else gate)
            if params:
                lines.append(f"{g}({', '.join(params)}) {', '.join(targets)};")
            else:
                lines.append(f"{g} {', '.join(targets)};")
    return "\n".join(lines) + "\n"


def _to_originir(qasm2: str) -> str:
    """Convert OpenQASM 2.0 -> OriginIR (Origin Quantum)."""
    qregs, cregs, ops = _parse_qasm2(qasm2)
    lines = []
    for name, size in qregs.items():
        lines.append(f"QINIT {size}")
    for name, size in cregs.items():
        lines.append(f"CREG {size}")
    for op in ops:
        if op[0] == "measure":
            _, q, qi, c, ci = op
            if qi is not None:
                lines.append(f"MEASURE {q}[{qi}], {c}[{ci}]")
            else:
                size = cregs.get(c, 0)
                for i in range(size):
                    lines.append(f"MEASURE {q}[{i}], {c}[{i}]")
        else:
            _, gate, params, targets = op
            g = _GATE_MAP_ORIGIN.get(gate.lower(), gate.upper())
            if params:
                lines.append(f"{g}({', '.join(params)}) {', '.join(targets)}")
            else:
                lines.append(f"{g} {', '.join(targets)}")
    return "\n".join(lines) + "\n"


def transpile(qasm_str: str, target: str) -> str:
    """Translate OpenQASM 2.0 into the target backend's native representation."""
    if target == "spinq":
        return _normalize_qasm2(qasm_str)
    if target == "braket":
        return _to_qasm3(qasm_str)
    if target == "originq":
        return _to_originir(qasm_str)
    raise ValueError(f"unsupported target: {target}")


# ======================================================================
# Run: execute on backend simulators, return unified schema
# ======================================================================

def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_braket(qasm2: str, shots: int) -> Dict[str, Any]:
    from braket.devices import LocalSimulator
    from braket.ir.openqasm import Program

    qasm3 = _to_qasm3(qasm2)
    # Braket's interpreter opens include files relative to cwd; chdir to the
    # starter_kit dir so `include "stdgates.inc"` resolves to our local copy.
    old_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    try:
        device = LocalSimulator()
        task = device.run(Program(source=qasm3), shots=shots)
        result = task.result()
    finally:
        os.chdir(old_cwd)
    counts = {str(k): int(v) for k, v in result.measurement_counts.items()}
    job_id = getattr(result.task_metadata, "id", None) or f"braket-local-{int(time.time()*1000)}"
    return {
        "backend": "braket_local_simulator",
        "job_id": job_id,
        "shots": shots,
        "counts": counts,
        "bit_order": "little",
        "timestamp": _utcnow(),
        "meta": {"simulator": "braket_local_simulator"},
    }


def _run_spinq(qasm2: str, shots: int) -> Dict[str, Any]:
    import spinqit as sq
    from spinqit import BasicSimulatorConfig, get_basic_simulator, get_compiler

    qasm_norm = _normalize_qasm2(qasm2)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".qasm", delete=False, encoding="utf-8"
    )
    try:
        tmp.write(qasm_norm)
        tmp.close()
        compiler = get_compiler("qasm")
        ir = compiler.compile(tmp.name, 0)
    finally:
        if os.path.exists(tmp.name):
            os.unlink(tmp.name)

    engine = get_basic_simulator()
    config = BasicSimulatorConfig()
    config.configure_shots(shots)
    result = engine.execute(ir, config)
    counts = {str(k): int(v) for k, v in result.counts.items()}
    job_id = (
        getattr(result, "job_id", None)
        or getattr(result, "task_id", None)
        or f"spinq-local-{int(time.time()*1000)}"
    )
    return {
        "backend": "spinq_basic_simulator",
        "job_id": job_id,
        "shots": shots,
        "counts": counts,
        "bit_order": "little",
        "timestamp": _utcnow(),
        "meta": {"simulator": "spinq_taurus_simulator"},
    }


def _run_originq(qasm2: str, shots: int) -> Dict[str, Any]:
    import pyqpanda as pq

    machine = pq.CPUQVM()
    machine.init_qvm()
    qasm_norm = _normalize_qasm2(qasm2)
    try:
        if hasattr(pq, "convert_qasm_string_to_qprog"):
            prog, qreg, creg = pq.convert_qasm_string_to_qprog(qasm_norm, machine)
        else:
            prog = pq.convert_qasm_to_qprog(qasm_norm, machine)
            qreg = machine.get_allocate_qubits()
            creg = machine.get_allocate_cbits()
    except Exception as exc:
        machine.finalize()
        raise RuntimeError(f"QASM -> QProg conversion failed: {exc}") from exc

    result = machine.run_with_configuration(prog, creg, shots)
    num_bits = len(creg)
    counts: Dict[str, int] = {}
    for key, val in result.items():
        if isinstance(key, int):
            bits = bin(key)[2:].zfill(num_bits)
        else:
            bits = str(key).zfill(num_bits)
        # pyqpanda reports big-endian (c[0] leftmost); the competition contract
        # requires little-endian (key = c[n-1]...c[1]c[0], c[0] rightmost).
        counts[bits[::-1]] = int(val)
    machine.finalize()
    return {
        "backend": "originq_cpu_simulator",
        "job_id": f"originq-local-{int(time.time()*1000)}",
        "shots": shots,
        "counts": counts,
        "bit_order": "little",
        "timestamp": _utcnow(),
        "meta": {"simulator": "originq_local_simulator"},
    }


def run(qasm_str: str, target: str, shots: int) -> Dict[str, Any]:
    """Execute a circuit and return the unified result schema from the rules."""
    if target == "braket":
        return _run_braket(qasm_str, shots)
    if target == "spinq":
        return _run_spinq(qasm_str, shots)
    if target == "originq":
        return _run_originq(qasm_str, shots)
    raise ValueError(f"unsupported target: {target}")


# ======================================================================
# L2: agent_chat - "say it in human language" LLM agent
# ======================================================================

def _load_backend_table() -> str:
    """Load backend capability table (official selection reference)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_capabilities.json")
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        rows = []
        for b in data.get("backends", []):
            rows.append(
                f"- {b['id']}: {b['name']} | kind={b['kind']} | max_qubits={b['max_qubits']} "
                f"| queue={b['queue']} | cost={b['cost']} | account={b['requires_account']}"
            )
        return "\n".join(rows)
    except OSError:
        return "(backend table unavailable)"


_SYSTEM_PROMPT = """你是 LoomQ 量子接入平权助手。你的使命是让完全不懂量子计算的用户也能使用量子计算机。请遵循以下规则：

## 规则 1：生成 OpenQASM 2.0 电路
当用户要求生成/修复电路时，输出完整、可运行、无错误的 OpenQASM 2.0 程序，并用 ```qasm 代码块包裹。必须：
- 只使用以下 12 个标准门：h, x, s, sdg, t, tdg, rz(theta), ry(theta), cx, cu1(theta), swap, ccx
- 包含 qreg 和 creg 寄存器声明，以及 measure 测量语句
- 门名小写，参数用弧度
- 电路必须从语义上实现用户声明的目标态（如 Bell 态、GHZ 态、特定叠加态等）
示例：
```qasm
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
```

## 规则 2：修复错误电路
当用户给出报错的电路并要求修复时，识别其中的错误（未定义寄存器、门名大小写错误、缺少声明、语法错误等），在保持用户声明意图的前提下输出修复后的完整 QASM，同样用代码块包裹。

## 规则 3：智能选择后端
当用户询问选择哪个量子平台/后端时，只回复一个规范后端标识符，不要输出其他文字。依据官方后端能力表：

{backend_table}

选择规则：优先满足比特数上限；再看排队（none 优于 minutes_to_hours 优于 hours）；最后看费用（free 优先）。

## 规则 4：其他问题
对于普通问答，用简洁的中文回答。"""


def _build_system_prompt() -> str:
    return _SYSTEM_PROMPT.format(backend_table=_load_backend_table())


def _extract_qasm_block(text: str) -> str:
    """Extract the first OpenQASM 2.0 program from a model reply."""
    if not isinstance(text, str):
        return ""
    m = re.search(
        r"OPENQASM\s+2\.0;.*?(?=^\s*```|\Z)", text, re.DOTALL | re.MULTILINE
    )
    return m.group(0).strip() if m else ""


def _validate_qasm_syntax(qasm_str: str) -> bool:
    """Best-effort local syntax check via our own transpiler (no external call)."""
    try:
        qregs, cregs, ops = _parse_qasm2(qasm_str)
        if not qregs or not cregs:
            return False
        return True
    except Exception:
        return False


def agent_chat(prompt: str) -> str:
    """L2 entry point: read LOOMQ_LLM_* env, call the model, return reply text.

    Implements the recommended closed loop: generate QASM -> self-check with
    our own L1 -> on failure ask the model to fix it (max 2 attempts).
    """
    try:
        from .llm_client import chat_completion
    except ImportError:  # adapter imported as a top-level module (script mode)
        from llm_client import chat_completion

    system_prompt = _build_system_prompt()

    def call(messages: List[Dict[str, Any]]) -> str:
        resp = chat_completion(messages)
        try:
            return resp["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected LLM response shape: {exc}") from exc

    # First call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    reply = call(messages)

    # Self-check: if the reply contains QASM, verify it parses and runs on our
    # local simulator; if broken, ask the model to fix it once.
    qasm = _extract_qasm_block(reply)
    if qasm:
        try:
            run(qasm, "braket", shots=1024)  # reuse own L1 (local sim, no network)
        except Exception:
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": (
                    "你上一条回复中的 QASM 电路无法运行，请检查语法错误（寄存器声明、"
                    "门名大小写、参数格式、measure 语句）后重新输出修复版，仍然用 "
                    "```qasm 代码块包裹完整程序。"
                ),
            })
            reply = call(messages)

    return reply


# ======================================================================
# L3: Hybrid-QASM -> quantum ops + RISC-V assembly
# ======================================================================

def _compile_classical_to_asm(text: str) -> str:
    """Compile a Hybrid-QASM classical block into RISC-V assembly.

    Emitter targets TinyRISCVEmulator: only li/add/sub/addi/beq/bne/j.
    r1..r9 -> x1..x9, c[k] -> x10+k. A recursive-descent parser walks the
    mini grammar and emits straight-line assembly with branch labels.
    """
    tokens = re.findall(r"c\[\d+\]|r\d+|\d+|==|!=|[(){}=;+\-]|\w+", text)
    if not tokens:
        return ""

    # Pick temp registers that do not collide with r1..r9 or c[k] mappings.
    cbit_max = max((int(m) for m in re.findall(r"c\[(\d+)\]", text)), default=-1)
    occupied = set(range(1, 10)) | {10 + k for k in range(cbit_max + 1)}
    free = [r for r in range(31, 9, -1) if r not in occupied]
    if len(free) < 1:
        raise RuntimeError("not enough free registers to compile classical block")
    tmp_cmp = free[0]

    lines: List[str] = []
    pos = 0
    counter = [0]

    def peek() -> str:
        return tokens[pos] if pos < len(tokens) else ""

    def advance() -> str:
        nonlocal pos
        tok = peek()
        pos += 1
        return tok

    def new_label(prefix: str) -> str:
        counter[0] += 1
        return f"{prefix}{counter[0]}"

    def operand_reg(op: str) -> str:
        """Return a register name holding the operand value."""
        if op.startswith("c["):
            return f"x{10 + int(op[2:op.index(']')])}"
        if op.startswith("r"):
            return f"x{int(op[1:])}"
        reg = f"x{tmp_cmp}"
        lines.append(f"li {reg}, {op}")
        return reg

    def parse_program(stop_at_rbrace: bool = False) -> None:
        while pos < len(tokens):
            if stop_at_rbrace and peek() == "}":
                advance()
                return
            if peek() == ";":
                advance()
                continue
            if peek() == "if":
                advance()          # if
                advance()          # (
                left = advance()
                op = advance()
                right = advance()
                advance()          # )
                advance()          # {
                lreg = operand_reg(left)
                rreg = operand_reg(right)
                else_label = new_label("ELSE")
                end_label = new_label("ENDIF")
                if op == "==":
                    lines.append(f"bne {lreg}, {rreg}, {else_label}")
                else:              # !=
                    lines.append(f"beq {lreg}, {rreg}, {else_label}")
                parse_program(stop_at_rbrace=True)   # then block
                lines.append(f"j {end_label}")
                lines.append(f"{else_label}:")
                if peek() == "else":
                    advance()      # else
                    advance()      # {
                    parse_program(stop_at_rbrace=True)
                lines.append(f"{end_label}:")
                continue
            # assignment: rN = term (('+'|'-') term)*
            reg = advance()        # rN
            advance()              # =
            lhs = f"x{int(reg[1:])}"
            first = advance()
            if first.isdigit():
                lines.append(f"li {lhs}, {int(first)}")
            elif first != reg:
                lines.append(f"add {lhs}, x{int(first[1:])}, x0")
            while peek() in ("+", "-"):
                op = advance()
                term = advance()
                if term.isdigit():
                    imm = int(term)
                    lines.append(f"addi {lhs}, {lhs}, {-imm if op == '-' else imm}")
                else:
                    treg = f"x{int(term[1:])}"
                    lines.append(
                        f"add {lhs}, {lhs}, {treg}" if op == "+"
                        else f"sub {lhs}, {lhs}, {treg}"
                    )
            if peek() == ";":
                advance()

    parse_program()
    return "\n".join(lines) + "\n"


def compile_hybrid(hybrid_qasm_str: str) -> Tuple[List[str], str]:
    """Compile Hybrid-QASM: return (quantum ops list, RISC-V assembly text)."""
    qasm_lines = _clean_lines(hybrid_qasm_str)
    quantum_lines: List[str] = []
    classical_text: List[str] = []
    in_classical = False
    for line in qasm_lines:
        if line.startswith("classical"):
            in_classical = True
            body = line[len("classical"):].strip()
            if body.startswith("{"):
                classical_text.append(body[1:].rstrip("}").strip())
            continue
        if in_classical:
            classical_text.append(line.rstrip("}").strip())
            if "}" in line:
                in_classical = False
            continue
        quantum_lines.append(line)

    quantum_ops: List[str] = []
    for line in quantum_lines:
        line = line.rstrip(";").strip()
        if not line or line.startswith("OPENQASM") or line.startswith("include") \
           or line.startswith("qreg") or line.startswith("creg"):
            continue
        m = re.match(r"([A-Za-z_]\w*)\s*(?:\(([^)]*)\))?\s*(.+)", line)
        if m and m.group(3).strip():
            gate = m.group(1)
            params = [p.strip() for p in m.group(2).split(",")] if m.group(2) else []
            targets = [t.strip() for t in m.group(3).split(",") if t.strip()]
            quantum_ops.append(
                f"{gate}({', '.join(params)}) {', '.join(targets)}" if params
                else f"{gate} {', '.join(targets)}"
            )
        elif m:
            quantum_ops.append(m.group(1))

    assembly = _compile_classical_to_asm("\n".join(classical_text))
    return quantum_ops, assembly
