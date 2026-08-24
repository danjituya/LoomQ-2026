#!/usr/bin/env python3
"""Grover marker bug-fix self-check (穷举式).

验收要求：
  1. classify() 从 prompt 解析目标标记（标记<串>/target<串>/裸二进制串），
     解析不到默认 010；
  2. 动态构造 Grover-3 电路 + 动态期望分布，标记变 → 电路与峰值跟着变；
  3. 期望分布 = {标记: 0.94531, 其余各 0.00781}；
  4. "标记 010 / 110 / 001" 三个 prompt 的 fidelity 均 >= 0.97。

本脚本穷举全部 8 个 3 位标记 + 解析写法变体 + 默认回退，全部 PASS 才退出 0。
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starter_kit"))
from l2_oracle import classify, grover3_qasm, grover3_expected, oracle_fidelity  # noqa: E402

ALL_8 = [format(i, "03b") for i in range(8)]


def check(prompt, want, tag):
    """classify → 验证名称/峰值/期望分布/fidelity 四项。"""
    qasm, expected, name, kind = classify(prompt)
    if kind != "template":
        return False, f"kind={kind} (期望 template)"
    # 1) 名称里带正确的标记
    if want not in name:
        return False, f"名称 {name!r} 不含标记 {want}"
    # 2) 期望分布峰值就是标记
    peak = max(expected, key=expected.get)
    if peak != want:
        return False, f"峰值 {peak} != 期望 {want}"
    # 3) 期望数值：标记 0.94531，其余 0.00781
    if abs(expected[want] - 0.94531) > 1e-9:
        return False, f"P({want})={expected[want]} != 0.94531"
    bad = [k for k, v in expected.items() if k != want and abs(v - 0.00781) > 1e-9]
    if bad:
        return False, f"非标记分量异常: {bad} -> {expected}"
    # 4) 电路模拟 vs 期望分布 fidelity >= 0.97
    fid = oracle_fidelity(qasm, expected)
    if fid < 0.97:
        return False, f"fidelity={fid:.5f} < 0.97"
    return True, f"fidelity={fid:.5f}"


def main():
    print("=" * 74)
    print("Grover 标记修复自测：穷举 8 标记 + 写法变体 + 默认回退")
    print("=" * 74)
    cases = []
    # A) 验收三例（穷举式覆盖 010/110/001）
    for m in ("010", "110", "001"):
        cases.append((f"标记 {m}", m, f"验收·标记 {m}"))
    # B) 穷举全部 8 个 3 位标记（含 000/011/100/101/111）
    for m in ALL_8:
        cases.append((f"Grover 搜索，目标标记 {m}", m, f"穷举·标记 {m}"))
    # C) 解析写法变体
    cases += [
        ("target=101", "101", "变体·target="),
        ("target 011", "011", "变体·target 空格"),
        ("搜索 100", "100", "变体·裸串 搜索"),
        ("帮我用 Grover 算法找 111", "111", "变体·找"),
        ("Grover 搜索目标 001", "001", "变体·搜索目标"),
    ]
    # D) 默认回退（无任何标记信息）
    cases += [
        ("帮我写一个 Grover 搜索", "010", "默认·无标记"),
        ("Grover 算法搜索 8 个元素", "010", "默认·仅数字"),
    ]

    fails = 0
    for prompt, want, tag in cases:
        try:
            ok, msg = check(prompt, want, tag)
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        print(f"[{'PASS' if ok else 'FAIL'}] {tag:16s} {prompt!r:34s} -> {want} ({msg})")
        fails += 0 if ok else 1

    # E) 电路随标记变化的实证：110 与 001 的电路文本必须不同，且都不同于默认 010
    c010, c110, c001 = grover3_qasm("010"), grover3_qasm("110"), grover3_qasm("001")
    distinct = len({c010, c110, c001}) == 3
    print(f"[{'PASS' if distinct else 'FAIL'}] 电路动态性: 010/110/001 三电路互不相同 "
          f"({len(c010)}/{len(c110)}/{len(c001)} 字符)")
    # oracle 段的 X 门（第一个 ccz 之前）：只应含标记中为 '0' 的位（q 索引）
    def _oracle_x(qasm):
        out = []
        for l in qasm.splitlines():
            if l.startswith("ccx"):
                break
            m = re.match(r"^x q\[(\d)\];$", l)
            if m:
                out.append(m.group(1))
        return sorted(set(out))

    ox110 = _oracle_x(c110)  # "110" 只有 q[0] 是 0 -> ['0']
    ox001 = _oracle_x(c001)  # "001" 的 q[2],q[1] 是 0 -> ['1','2']
    oracle_ok = (ox110 == ["0"] and ox001 == ["1", "2"])
    print(f"[{'PASS' if oracle_ok else 'FAIL'}] oracle 位序: 110→x在q0, 001→x在q1,q2 "
          f"(实际 110:{ox110}, 001:{ox001})")
    fails += 0 if (distinct and oracle_ok) else 1

    print("=" * 74)
    total = len(cases) + 2
    print(f"结论: {total - fails}/{total} 项 PASS" + ("  -> ALL PASS" if fails == 0 else f"  -> {fails} 项 FAIL"))
    sys.exit(0 if fails == 0 else 1)


if __name__ == "__main__":
    main()
