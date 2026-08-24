#!/usr/bin/env python3
"""L3 混合编译完整测试套件（public + 自建补充覆盖）。

官方公开集（evaluator.py --level l3）只有 1 题：public-branch。
本套件复用该题，并补充多分支 / 算术 / 多 cbit / 嵌套 if / 寄存器运算，
用官方 TinyRISCVEmulator 对每个用例做穷举注入验证（r1..r9 -> x1..x9,
c[k] -> x10+k）。

用法：
    python tests/l3_suite.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starter_kit"))
import adapter  # noqa: E402
from riscv_emulator import TinyRISCVEmulator  # noqa: E402


def hybrid(nq, nc, quantum_lines, classical_body):
    """拼一个 Hybrid-QASM 文本。"""
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";']
    lines.append(f"qreg q[{nq}];")

    lines.append(f"creg c[{nc}];")

    lines += quantum_lines
    lines.append(f"classical {{ {classical_body} }}")
    return "\n".join(lines) + "\n"


def run_case(desc, src, injections, checks):
    """编译并验证：injections = [{reg: val}...], checks = [lambda state: bool]"""
    ops, asm = adapter.compile_hybrid(src)
    ok_all = True
    for inj, check in zip(injections, checks):
        emu = TinyRISCVEmulator()
        emu.load_program(asm)
        for reg, val in inj.items():
            emu.set_register(reg, val)
        state = emu.execute()
        passed = check(state)
        ok_all = ok_all and passed
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {desc} 注入{inj} -> 终态 { {k: v for k, v in state.items() if v} }")
    print(f"  -> {desc}: {'ALL PASS' if ok_all else 'FAILED'}")
    return ok_all


def main():
    print("=" * 66)
    print("L3 混合编译完整测试清单")
    print("=" * 66)

    # ---------- A. 官方公开集（evaluator.py 同款） ----------
    print("\n[A. 官方公开集 public-branch]（evaluator.py --level l3 原题）")
    src = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
creg c[1];
measure q[0] -> c[0];
classical { if (c[0] == 1) { r1 = 7; } else { r1 = 3; } }"""
    all_ok = run_case("if(c[0]==1) r1=7 else r1=3",
                      src,
                      [{"x10": 0}, {"x10": 1}],
                      [lambda s: s.get("x1") == 3, lambda s: s.get("x1") == 7])

    # ---------- B. 自建补充：算术表达式 ----------
    print("\n[B. 自建补充：顺序赋值 + 算术]")
    src = hybrid(2, 2,
                 ["measure q[0] -> c[0];"],
                 "if (c[0] == 1) { r1 = 100; } else { r1 = 10; } r1 = r1 + 5;")
    all_ok &= run_case("if/else + r1=r1+5（期望 105/15）",
                       src,
                       [{"x10": 1}, {"x10": 0}],
                       [lambda s: s.get("x1") == 105, lambda s: s.get("x1") == 15])

    # ---------- C. 自建补充：多 cbit + 寄存器运算 ----------
    print("\n[C. 自建补充：多 cbit 条件 + 寄存器运算]")
    src = hybrid(2, 2,
                 ["measure q -> c;"],
                 "if (c[0] != 0) { r2 = r1 + 3; } else { r2 = r1 - 1; } r3 = r2 + r4;")
    all_ok &= run_case("c[0]!=0 分支 + r3=r2+r4",
                       src,
                       [{"x10": 1, "x1": 10, "x4": 100}, {"x10": 0, "x1": 10, "x4": 100}],
                       [lambda s: s.get("x2") == 13 and s.get("x3") == 113,
                        lambda s: s.get("x2") == 9 and s.get("x3") == 109])

    # ---------- D. 自建补充：c[1] 条件 ----------
    print("\n[D. 自建补充：c[1] 条件（x11）]")
    src = hybrid(2, 2,
                 ["measure q -> c;"],
                 "if (c[1] == 0) { r5 = 42; } else { r5 = 24; }")
    all_ok &= run_case("if(c[1]==0) r5=42 else 24",
                       src,
                       [{"x11": 0}, {"x11": 1}],
                       [lambda s: s.get("x5") == 42, lambda s: s.get("x5") == 24])

    # ---------- E. 自建补充：嵌套 if ----------
    print("\n[E. 自建补充：嵌套 if/else]")
    src = hybrid(2, 2,
                 ["measure q -> c;"],
                 "if (c[0] == 1) { if (c[1] == 0) { r1 = 1; } else { r1 = 2; } } else { r1 = 3; }")
    all_ok &= run_case("嵌套：外 c[0] 内 c[1]",
                       src,
                       [{"x10": 1, "x11": 0}, {"x10": 1, "x11": 1}, {"x10": 0, "x11": 0}],
                       [lambda s: s.get("x1") == 1, lambda s: s.get("x1") == 2,
                        lambda s: s.get("x1") == 3])

    # ---------- F. 自建补充：纯寄存器赋值（无分支） ----------
    print("\n[F. 自建补充：顺序寄存器赋值 + 加减]")
    src = hybrid(1, 1,
                 ["measure q[0] -> c[0];"],
                 "r1 = 20; r2 = r1 + 30; r3 = r2 - 5;")
    all_ok &= run_case("r1=20; r2=r1+30; r3=r2-5（期望 r3=45）",
                       src,
                       [{}],
                       [lambda s: s.get("x1") == 20 and s.get("x2") == 50 and s.get("x3") == 45])

    print("\n" + "=" * 66)
    print(f"L3 套件：{'ALL PASS' if all_ok else 'SOME FAILED'}（官方 1 题 + 自建 5 组 = 6 组覆盖）")
    print("=" * 66)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
