#!/usr/bin/env python3
"""L3 随机 fuzz 套件：复刻官方评测方式（确定性可复现）。

评测要求（赛题手册 §L3）：
  随机生成 N 组 Hybrid-QASM（不同分支结构/不同常量/不同测量位数）→
  将 compile_hybrid 输出的 RISC-V 汇编载入 TinyRISCVEmulator →
  穷举注入所有测量值组合 → 逐一比对寄存器终态与参考解释器结果 →
  同时校验量子操作序列与原电路量子部分语义等价。

本套件：
  A. 自写参考解释器（Python 直译文法语义）作为 ground truth
  B. 随机生成器：分支结构/常量(含负数)/测量位数/嵌套深度 随机
  C. 对每个用例穷举 2^n_cbit 注入，比对汇编终态 == 参考终态
  D. 校验 quantum_ops 与输入量子部分逐条等价
  E. 固定随机种子 → 每次运行结果可复现

用法： python tests/l3_fuzz.py [用例数=200] [种子=42]
"""
import random
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starter_kit"))
import adapter  # noqa: E402
from riscv_emulator import TinyRISCVEmulator  # noqa: E402


# ======================================================================
# A. 参考解释器（ground truth）
# ======================================================================
class RefInterp:
    """Directly evaluate the mini grammar (integers, r1..r9, + - == !=,
    if/else, sequential assignment, c[k] reads)."""

    def __init__(self, cbits):
        self.r = {i: 0 for i in range(1, 10)}
        self.c = [0] * cbits

    def inject(self, values):
        for k, v in values.items():
            self.c[k] = v

    def val(self, tok):
        tok = tok.strip()
        if tok.startswith("c["):
            return self.c[int(tok[2:tok.index("]")])]
        if tok.startswith("r"):
            return self.r[int(tok[1:])]
        return int(tok)  # 含负数字面量

    def run(self, stmts):
        for s in stmts:
            if s[0] == "assign":
                _, lhs, first, pairs = s  # pairs: [("+", tok), ("-", tok), ...]
                acc = self.val(first)
                for op, tok in pairs:
                    v = self.val(tok)
                    acc = acc + v if op == "+" else acc - v
                self.r[lhs] = acc
            else:  # if/else
                _, cond_left, cond_op, cond_right, then_s, else_s = s
                lv, rv = self.val(cond_left), self.val(cond_right)
                hit = (lv == rv) if cond_op == "==" else (lv != rv)
                self.run(then_s if hit else else_s)


# ======================================================================
# B. 随机生成器
# ======================================================================
def gen_term(rng, cbits):
    kind = rng.random()
    if kind < 0.45:
        return f"r{rng.randint(1, 9)}"
    if kind < 0.70:
        return f"c[{rng.randint(0, cbits - 1)}]"
    return str(rng.randint(-30, 30))  # 含负数常量


def gen_expr(rng, cbits):
    first = gen_term(rng, cbits)
    pairs = []
    for _ in range(rng.randint(0, 3)):
        pairs.append((rng.choice(["+", "-"]), gen_term(rng, cbits)))
    return first, pairs


def gen_stmts(rng, cbits, depth):
    stmts = []
    n = rng.randint(1, 4)
    for _ in range(n):
        if depth > 0 and rng.random() < 0.4:
            left = gen_term(rng, cbits)
            op = rng.choice(["==", "!="])
            right = gen_term(rng, cbits)
            then_s = gen_stmts(rng, cbits, depth - 1)
            else_s = gen_stmts(rng, cbits, depth - 1) if rng.random() < 0.8 else []
            stmts.append(("if", left, op, right, then_s, else_s))
        else:
            lhs = rng.randint(1, 9)
            stmts.append(("assign", lhs, *gen_expr(rng, cbits)))
    return stmts


def render_stmts(stmts, indent=0):
    pad = "  " * indent
    out = []
    for s in stmts:
        if s[0] == "assign":
            _, lhs, first, pairs = s
            expr = first + "".join(f" {op} {t}" for op, t in pairs)
            out.append(f"{pad}r{lhs} = {expr};")
        else:
            _, l, op, r, t, e = s
            out.append(f"{pad}if ({l} {op} {r}) {{")
            out.extend(render_stmts(t, indent + 1))
            if e:
                out.append(f"{pad}}} else {{")
                out.extend(render_stmts(e, indent + 1))
            out.append(f"{pad}}}")
    return out


def gen_hybrid(rng, cbits, nq=None):
    nq = nq or max(cbits, rng.randint(1, 3))
    qlines = []
    for q in range(nq):
        r = rng.random()
        if r < 0.4:
            qlines.append(f"h q[{q}];")
        elif r < 0.7:
            qlines.append(f"x q[{q}];")
        elif r < 0.85 and q + 1 < nq:
            qlines.append(f"cx q[{q}], q[{q + 1}];")
    for q in range(cbits):
        qlines.append(f"measure q[{q}] -> c[{q}];")
    stmts = gen_stmts(rng, cbits, depth=rng.randint(1, 3))
    body = "\n".join(render_stmts(stmts))
    src = ("OPENQASM 2.0;\ninclude \"qelib1.inc\";\n"
           f"qreg q[{nq}];\ncreg c[{cbits}];\n" + "\n".join(qlines) +
           f"\nclassical {{\n{body}\n}}")
    return src, stmts, qlines


# ======================================================================
# C/D. 校验
# ======================================================================
def check_case(src, stmts, qlines, rng):
    ops, asm = adapter.compile_hybrid(src)
    if not asm or not ops:
        return False, "空输出"

    # D. 量子部分语义等价：compile_hybrid 的 quantum_ops 应与输入量子行逐条等价
    def norm_q(line):
        line = line.rstrip(";").strip()
        return re.sub(r"\s+", " ", line)

    expect_ops = [norm_q(l) for l in qlines]
    # compile_hybrid 输出格式：'h q[0]' / 'cx q[0], q[1]'（含逗号空格）
    def norm_out(op):
        return re.sub(r"\s*,\s*", ", ", op).strip()

    got_ops = [norm_out(o) for o in ops]
    if got_ops != expect_ops:
        return False, f"量子操作序列不等价:\n  期望 {expect_ops}\n  实际 {got_ops}"

    # C. 穷举注入 2^n_cbit 组合
    cbits = len([1 for l in qlines if l.startswith("measure")])
    # 用输入里的 creg 数（上面 qlines 里 measure 条数）
    for mask in range(1 << cbits):
        ref = RefInterp(cbits)
        ref.inject({k: (mask >> k) & 1 for k in range(cbits)})
        ref.run(stmts)
        emu = TinyRISCVEmulator()
        emu.load_program(asm)
        for k in range(cbits):
            emu.set_register(f"x{10 + k}", (mask >> k) & 1)
        st = emu.execute()
        for i in range(1, 10):
            want = ref.r[i]
            got = st.get(f"x{i}", 0)
            if got != want:
                return False, (f"注入mask={mask:0{cbits}b}: x{i}={got} 期望 {want}\n"
                               f"  汇编:\n{asm}")
    return True, ""


def main():
    n_cases = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    rng = random.Random(seed)
    print("=" * 66)
    print(f"L3 随机 fuzz：{n_cases} 用例，种子 {seed}（穷举注入 + 量子部分等价）")
    print("=" * 66)
    fails = 0
    for idx in range(n_cases):
        cbits = rng.randint(1, 4)
        src, stmts, qlines = gen_hybrid(rng, cbits)
        ok, msg = check_case(src, stmts, qlines, rng)
        if not ok:
            fails += 1
            print(f"[FAIL] 用例#{idx} (cbits={cbits}): {msg}")
            print("---- 输入 ----")
            print(src)
            if fails >= 3:
                print("已到 3 个 FAIL，中止。")
                break
    print("=" * 66)
    total = n_cases
    print(f"结论: {total - fails}/{total} 用例全注入通过"
          + ("  -> ALL PASS" if fails == 0 else f"  -> {fails} FAIL"))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
