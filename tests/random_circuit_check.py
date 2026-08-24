#!/usr/bin/env python3
"""L1 随机电路保真度检查（补手册 8 电路中的 Random-Circuit x3 盲区）。

手册 L1 语义等价性覆盖 Bell / GHZ-3 / GHZ-5 / QFT-4 / Grover-3 /
Random-Circuit x3。前 5 类已有模板测试，随机电路此前未测——本脚本用
固定种子的随机 12 门电路（3 组不同种子/深度）验证：
  transpile(qasm, target) 后的电路 与 原电路 在自写精确模拟器上的
  Hellinger fidelity >= 0.97（与官方评测口径一致）。

用法： python tests/random_circuit_check.py
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "starter_kit"))
import adapter  # noqa: E402
from l2_oracle import simulate_statevector, hellinger_fidelity  # noqa: E402

GATES = ["h", "x", "s", "sdg", "t", "tdg", "rz", "ry", "cx", "cu1", "swap", "ccx"]
PARAM_GATES = {"rz", "ry", "cu1"}


def random_qasm(rng, nq, depth):
    lines = ["OPENQASM 2.0;", 'include "qelib1.inc";',
             f"qreg q[{nq}];", f"creg c[{nq}];"]
    for _ in range(depth):
        g = rng.choice(GATES)
        if g == "ccx":
            qs = rng.sample(range(nq), 3) if nq >= 3 else None
            if qs is None:
                continue
            lines.append(f"ccx q[{qs[0]}], q[{qs[1]}], q[{qs[2]}];")
        elif g in ("cx", "cu1", "swap"):
            qs = rng.sample(range(nq), 2) if nq >= 2 else None
            if qs is None:
                continue
            if g == "cu1":
                lines.append(f"cu1({rng.choice([0.5, 1.0, 1.5708, 3.1416])}) q[{qs[0]}], q[{qs[1]}];")
            else:
                lines.append(f"{g} q[{qs[0]}], q[{qs[1]}];")
        elif g in PARAM_GATES:
            q = rng.randrange(nq)
            lines.append(f"{g}({rng.choice([0.5, 1.0, 1.5708, 3.1416])}) q[{q}];")
        else:
            q = rng.randrange(nq)
            lines.append(f"{g} q[{q}];")
    for i in range(nq):
        lines.append(f"measure q[{i}] -> c[{i}];")
    return "\n".join(lines) + "\n"


def check(seed, nq, depth):
    rng = random.Random(seed)
    qasm = random_qasm(rng, nq, depth)
    _, ref = simulate_statevector(qasm)  # 参考分布（原电路，精确）
    results = {}
    for target in ("braket", "originq"):
        try:
            # 完整链路：transpile -> 后端模拟执行 -> counts（采样分布）
            r = adapter.run(qasm, target, 8192)
            tot = sum(r["counts"].values())
            dist = {k: v / tot for k, v in r["counts"].items()}
            fid = hellinger_fidelity(dist, ref)
            results[target] = ("OK", round(fid, 4))
        except Exception as exc:  # noqa: BLE001
            results[target] = ("ERR", str(exc))
    return results


def main():
    print("=" * 68)
    print("L1 Random-Circuit x3 保真度检查（固定种子，12 门白名单）")
    print("=" * 68)
    cases = [(7, 3, 12), (1234, 4, 18), (20260825, 5, 25)]
    all_ok = True
    for idx, (seed, nq, depth) in enumerate(cases, 1):
        res = check(seed, nq, depth)
        print(f"RC#{idx}: nq={nq} depth={depth} seed={seed}")
        for t, (st, v) in res.items():
            ok = st == "OK" and v >= 0.97
            all_ok = all_ok and ok
            print(f"  [{('PASS' if ok else 'FAIL')}] {t:9s} {st} fidelity={v}")
    print("=" * 68)
    print(f"结论: {'ALL PASS' if all_ok else 'HAS FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
