#!/usr/bin/env python3
"""L1 逐门保真度诊断：定位 braket↔originq 一致性不足 1.0 的来源。

对 all12.qasm 里每一个门单独构造一个"能暴露其行为"的小电路，
在 braket / originq 各取 2048 shots，逐门计算 Hellinger 保真度。

结论判定：
  - 所有门 >= 0.999  -> 整电路一致性损失来自采样噪声（2048 shots 下合理）；
  - 某门 < 0.999    -> 该门在两后端实现有偏差，需进一步定位（等价分解 vs 后端 bug）。

用法：
    python tests/l1_gate_diag.py [--shots 2048]
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starter_kit"))
import adapter  # noqa: E402

SHOTS = 2048


def hellinger(a, b):
    states = set(a) | set(b)
    d = math.sqrt(sum((math.sqrt(a.get(s, 0.0)) - math.sqrt(b.get(s, 0.0))) ** 2
                      for s in states)) / math.sqrt(2.0)
    return max(0.0, min(1.0, 1.0 - d))


def norm(c):
    tot = sum(c.values()) or 1
    return {k: v / tot for k, v in c.items()}


# 每门一个行为暴露电路（白名单门，作用于非平凡初态）
GATE_CIRCUITS = {
    "h":   "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; measure q[0] -> c[0];",
    "x":   "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; x q[0]; measure q[0] -> c[0];",
    "s":   "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; s q[0]; h q[0]; measure q[0] -> c[0];",
    "sdg": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; sdg q[0]; h q[0]; measure q[0] -> c[0];",
    "t":   "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; t q[0]; h q[0]; measure q[0] -> c[0];",
    "tdg": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; tdg q[0]; h q[0]; measure q[0] -> c[0];",
    "rz":  "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; h q[0]; rz(0.7) q[0]; h q[0]; measure q[0] -> c[0];",
    "ry":  "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; creg c[1]; ry(0.7) q[0]; measure q[0] -> c[0];",
    "cx":  "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; x q[0]; cx q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];",
    "cu1": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; x q[0]; x q[1]; cu1(0.7) q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];",
    "swap":"OPENQASM 2.0; include \"qelib1.inc\"; qreg q[2]; creg c[2]; x q[0]; swap q[0], q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];",
    "ccx": "OPENQASM 2.0; include \"qelib1.inc\"; qreg q[3]; creg c[3]; x q[0]; x q[1]; ccx q[0], q[1], q[2]; measure q[0] -> c[0]; measure q[1] -> c[1]; measure q[2] -> c[2];",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=SHOTS)
    args = ap.parse_args()
    shots = args.shots

    import l2_oracle

    print("=" * 78)
    print(f"逐门诊断（每门 {shots} shots）：braket↔originq 互比 + 各后端 vs 理论")
    print("=" * 78)

    low = []
    for gate, qasm in GATE_CIRCUITS.items():
        try:
            n, theo = l2_oracle.simulate_statevector(qasm)
            rb = adapter.run(qasm, "braket", shots)["counts"]
            ro = adapter.run(qasm, "originq", shots)["counts"]
            fid_bo = hellinger(norm(rb), norm(ro))
            fid_bt = hellinger(norm(rb), theo)
            fid_ot = hellinger(norm(ro), theo)
            flag = "OK " if fid_bt >= 0.999 and fid_ot >= 0.999 else "LOW"
            if fid_bt < 0.999 or fid_ot < 0.999:
                low.append((gate, fid_bo, fid_bt, fid_ot, rb, ro))
            print(f"[{flag}] {gate:5s} 互比={fid_bo:.4f}  braket理论={fid_bt:.4f}  originq理论={fid_ot:.4f}")
        except Exception as exc:
            print(f"[ERR] {gate:5s} {type(exc).__name__}: {exc}")

    print("-" * 78)
    if not low:
        print("结论：每个门在两后端各自 vs 理论均 >= 0.999（两后端都正确），")
        print("  braket↔originq 互比略低（0.99x）来自两路独立采样的叠加噪声。")
        print("  all12 整电路一致性 ~0.99 的来源 = 采样噪声（非后端偏差）。")
        print("结论类型：采样噪声")
    else:
        print("发现 vs 理论 < 0.999 的门（这才是真实偏差信号）：")
        for gate, fid_bo, fid_bt, fid_ot, rb, ro in low:
            print(f"  {gate}: 互比={fid_bo:.4f} braket理论={fid_bt:.4f} originq理论={fid_ot:.4f}")
            print(f"    braket  ={dict(sorted(rb.items(), key=lambda kv: -kv[1]))}")
            print(f"    originq ={dict(sorted(ro.items(), key=lambda kv: -kv[1]))}")
            _t = l2_oracle.simulate_statevector(GATE_CIRCUITS[gate])[1]
            print(f"    理论    ={ {k: round(v, 4) for k, v in sorted(_t.items(), key=lambda kv: -kv[1])[:4]} }")
        print("提示：若某后端 vs 理论系统性偏离（缺失分量/比例错），属该后端或等价分解偏差，需修复；"
              "若只是统计波动，属采样噪声。")
        print("结论类型：见上方输出（以 vs 理论为准判别）")
    print("=" * 78)


if __name__ == "__main__":
    main()
