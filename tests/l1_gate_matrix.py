#!/usr/bin/env python3
"""L1 白名单 12 门 × 后端全覆盖自动化测试.

对每个门构造一个"能暴露其行为"的小电路，用自写精确态矢量模拟器
(l2_oracle.simulate_statevector) 计算理论分布，然后在 braket / originq
两个后端各跑一次，比对 Hellinger Fidelity。输出 12门×后端 支持矩阵。

用法:
    python tests/l1_gate_matrix.py [--shots 8192]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, "starter_kit")

import adapter  # noqa: E402
import l2_oracle  # noqa: E402

SHOTS = 8192
FID_THRESHOLD = 0.97

# 每个门的验证电路：构造一个非平凡初态以暴露该门的真实行为。
# 期望分布由 l2_oracle 精确计算（不写死，避免手工错误）。
GATE_CIRCUITS = {
    "h":   "qreg q[1]; creg c[1]; h q[0];",
    "x":   "qreg q[1]; creg c[1]; x q[0];",
    "s":   "qreg q[1]; creg c[1]; h q[0]; s q[0];",
    "sdg": "qreg q[1]; creg c[1]; h q[0]; sdg q[0];",
    "t":   "qreg q[1]; creg c[1]; h q[0]; t q[0];",
    "tdg": "qreg q[1]; creg c[1]; h q[0]; tdg q[0];",
    "rz":  "qreg q[1]; creg c[1]; x q[0]; rz(0.6) q[0];",
    "ry":  "qreg q[1]; creg c[1]; ry(0.7) q[0];",
    "cx":  "qreg q[2]; creg c[2]; h q[0]; cx q[0], q[1];",
    "cu1": "qreg q[2]; creg c[2]; h q[0]; x q[1]; cu1(0.5) q[0], q[1];",
    "swap": "qreg q[2]; creg c[2]; x q[0]; swap q[0], q[1];",
    "ccx": "qreg q[3]; creg c[3]; x q[0]; x q[1]; ccx q[0], q[1], q[2];",
}


def full_qasm(body: str) -> str:
    return "OPENQASM 2.0;\ninclude \"qelib1.inc\";\n" + body + \
           "\nmeasure q -> c;\n"


def hellinger(p, q):
    states = set(p) | set(q)
    d = math.sqrt(sum((math.sqrt(p.get(s, 0.0)) - math.sqrt(q.get(s, 0.0))) ** 2
                      for s in states)) / math.sqrt(2.0)
    return max(0.0, min(1.0, 1.0 - d))


def run_and_fidelity(qasm, target, expected):
    try:
        res = adapter.run(qasm, target, SHOTS)
        counts = res["counts"]
        total = sum(counts.values())
        observed = {str(k): v / total for k, v in counts.items()}
        fid = hellinger(observed, expected)
        return fid, None
    except Exception as exc:  # noqa: BLE001
        return 0.0, f"{type(exc).__name__}: {exc}"


def main():
    global SHOTS
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, default=8192)
    args = parser.parse_args()
    SHOTS = args.shots

    targets = ["braket", "originq"]
    print(f"L1 白名单 12 门 × 后端全覆盖测试（shots={SHOTS}, 阈值={FID_THRESHOLD}）")
    print("=" * 74)
    print(f"{'门':<6}{'braket':<12}{'originq':<12}  状态")
    print("-" * 74)

    matrix = {g: {} for g in GATE_CIRCUITS}
    all_ok = True
    for gate, body in GATE_CIRCUITS.items():
        qasm = full_qasm(body)
        try:
            _, expected = l2_oracle.simulate_statevector(qasm)
        except Exception as exc:  # noqa: BLE001
            print(f"{gate:<6} oracle 失败: {exc}")
            all_ok = False
            continue
        row = []
        for target in targets:
            fid, err = run_and_fidelity(qasm, target, expected)
            ok = err is None and fid >= FID_THRESHOLD
            matrix[gate][target] = ok
            all_ok = all_ok and ok
            cell = f"{fid:.3f}" if err is None else "ERR"
            row.append(f"{cell:<12}")
        mark = "OK" if all(matrix[gate][t] for t in targets) else "FAIL"
        print(f"{gate:<6}{row[0]}{row[1]}  {mark}")
        if mark == "FAIL":
            # 打印期望分布帮助定位
            top = sorted(expected.items(), key=lambda kv: -kv[1])[:4]
            print(f"        期望分布: {dict(top)}")

    # all12.qasm 组合电路
    print("-" * 74)
    all12_path = os.path.join("starter_kit", "circuits", "all12.qasm")
    with open(all12_path, encoding="utf-8") as fh:
        all12 = fh.read()
    try:
        _, expected12 = l2_oracle.simulate_statevector(all12)
    except Exception as exc:  # noqa: BLE001
        print(f"all12 oracle 失败: {exc}")
        all_ok = False
    else:
        row = []
        for target in targets:
            fid, err = run_and_fidelity(all12, target, expected12)
            ok = err is None and fid >= FID_THRESHOLD
            all_ok = all_ok and ok
            row.append(f"{fid:.3f}" if err is None else "ERR")
        print(f"{'all12':<6}{row[0]:<12}{row[1]:<12}  组合电路（12门一起）")

    print("=" * 74)
    print("12门×后端 支持矩阵:")
    header = "        " + "".join(f"{t:<10}" for t in targets)
    print(header)
    for gate in GATE_CIRCUITS:
        cells = "".join(f"{'YES' if matrix[gate][t] else 'NO ':<10}" for t in targets)
        print(f"  {gate:<6}{cells}")
    print("=" * 74)
    print("结论:", "ALL PASS - 12 门在 2 后端全部验证通过" if all_ok else "存在失败，请检查对应门")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
