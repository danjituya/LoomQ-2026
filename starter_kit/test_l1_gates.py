#!/usr/bin/env python3
"""L1 验证三件套之二：12 门全量 + 确定性单门 自测。

放到 starter_kit/ 目录下，与 adapter.py 同目录运行：
    python test_l1_gates.py
（需在能 import braket / pyqpanda 的环境，如官方 Docker 内）

它做三件事：
  1) 用 all12.qasm（含全部 12 个白名单门）在每个可用后端跑一次，必须不报错；
  2) 比对 braket 与 originq 的输出分布是否一致（同一电路 → 分布应相同）；
  3) 用几个确定性单门电路做断言（x→|1>, h→50/50, swap, ccx, ry 等）。
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapter import run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def hellinger(a, b):
    states = set(a) | set(b)
    d = math.sqrt(sum((math.sqrt(a.get(s, 0.0)) - math.sqrt(b.get(s, 0.0))) ** 2
                      for s in states)) / math.sqrt(2.0)
    return max(0.0, min(1.0, 1.0 - d))


def run_qasm(qasm_text, target, shots=1024):
    return run(qasm_text, target, shots)


def main():
    print("=" * 60)
    print("L1 全部 12 门 × 后端 验证")
    print("=" * 60)

    all12 = open(os.path.join(HERE, "all12.qasm"), encoding="utf-8").read()
    backends = ["braket", "originq"]  # spinq 未安装，单独检查优雅报错
    counts = {}

    for tgt in backends:
        try:
            # 8192 shots: the 0.99 cross-backend threshold is statistically
            # unreachable at 2048 shots for a 4-outcome distribution (~1% noise).
            r = run_qasm(all12, tgt, 8192)
            counts[tgt] = r["counts"]
            nonzero = {k: v for k, v in r["counts"].items() if v > 0}
            print(f"[OK]   {tgt:8s}: all12 运行成功，{len(nonzero)} 个非空结果")
        except Exception as exc:
            print(f"[FAIL] {tgt:8s}: {type(exc).__name__}: {exc}")
            counts[tgt] = None

    # 跨后端一致性
    if counts.get("braket") and counts.get("originq"):
        # hellinger() 需要概率分布，先把原始计数归一化
        def _norm(c):
            tot = sum(c.values()) or 1
            return {k: v / tot for k, v in c.items()}

        fid = hellinger(_norm(counts["braket"]), _norm(counts["originq"]))
        # 0.97 = 官方评测器通过阈值。两个独立采样之间的 Hellinger 噪声约 1-1.5%，
        # 0.99 在此 shot 数下统计上不可达；真实后端分歧（门转译错误）会低到 0.5 以下。
        status = "PASS" if fid >= 0.97 else "FAIL"
        print(f"[{status}] braket 与 originq 分布一致性 fidelity = {fid:.4f} (要求 >=0.97)")

    # ---- 确定性单门断言 ----
    print("-" * 60)
    print("确定性单门检查 (仅 braket，作为基线后端)")
    print("-" * 60)
    cases = [
        ("x → |1>", "qreg q[1]; creg c[1]; x q[0]; measure q[0] -> c[0];", {"1": 1.0}),
        ("h → 50/50", "qreg q[1]; creg c[1]; h q[0]; measure q[0] -> c[0];", {"0": 0.5, "1": 0.5}),
        ("ry(π/2) → 50/50",
         "qreg q[1]; creg c[1]; ry(1.570796326795) q[0]; measure q[0] -> c[0];",
         {"0": 0.5, "1": 0.5}),
        ("swap |10> → |01>",
         "qreg q[2]; creg c[2]; x q[0]; swap q[0],q[1]; measure q[0] -> c[0]; measure q[1] -> c[1];",
         # Contract: little-endian (Qiskit convention) -> key = c[1]c[0].
         # After x q[0]; swap: q0=|0>, q1=|1> -> c[0]=0, c[1]=1 -> "10".
         {"10": 1.0}),
        ("ccx |110> → |111>",
         "qreg q[3]; creg c[3]; x q[0]; x q[1]; ccx q[0],q[1],q[2]; "
         "measure q[0] -> c[0]; measure q[1] -> c[1]; measure q[2] -> c[2];",
         {"111": 1.0}),
        ("cu1(π/2) 相位 (|11>→|11> 带相位)",
         "qreg q[2]; creg c[2]; x q[0]; x q[1]; cu1(1.570796326795) q[0],q[1]; "
         "measure q[0] -> c[0]; measure q[1] -> c[1];",
         {"11": 1.0}),
    ]
    for name, qasm, expected in cases:
        try:
            r = run_qasm(qasm, "braket", 2048)
            got = r["counts"]
            ok = all(abs(got.get(k, 0) / 2048 - v) < 0.05 for k, v in expected.items())
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {name}: {got}")
        except Exception as exc:
            print(f"[FAIL] {name}: {type(exc).__name__}: {exc}")

    # ---- spinq 优雅报错检查 ----
    print("-" * 60)
    print("spinq 优雅报错检查")
    print("-" * 60)
    try:
        run_qasm(all12, "spinq", 1024)
        print("[FAIL] spinq 未报错（预期应优雅抛出 RuntimeError）")
    except RuntimeError as exc:
        print(f"[PASS] spinq 优雅报错: {exc}")
    except Exception as exc:
        print(f"[WARN] spinq 抛出非 RuntimeError: {type(exc).__name__}: {exc} "
              f"(建议改为 RuntimeError 以利上层捕获)")

    print("=" * 60)
    print("完成。任何 [FAIL] 都意味着该门/后端存在真实问题，必须修。")
    print("=" * 60)


if __name__ == "__main__":
    main()
