#!/usr/bin/env python3
"""L2 真实 API 自测脚本（改造方案要求的 4 项 + 12 类路由覆盖）。

用法（先设置环境变量）：
    $env:LOOMQ_LLM_API_KEY="sk-xxx"
    $env:LOOMQ_LLM_BASE_URL="https://api.deepseek.com"
    $env:LOOMQ_LLM_MODEL="deepseek-v4-flash"
    python tests/test_l2_real.py
"""
import os
import sys
import time

sys.path.insert(0, "starter_kit")

import l2_oracle  # noqa: E402

REQUIRED = ("LOOMQ_LLM_BASE_URL", "LOOMQ_LLM_API_KEY", "LOOMQ_LLM_MODEL")
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    print("缺少环境变量:", ", ".join(missing))
    print("请先设置 LOOMQ_LLM_* 三个变量（见文件头注释）")
    sys.exit(1)

import adapter  # noqa: E402


def verify(prompt, expected):
    import math

    t0 = time.time()
    try:
        reply = adapter.agent_chat(prompt)
    except Exception as exc:
        print(f"[ERROR] {prompt[:20]}: {type(exc).__name__}: {exc}")
        return False
    qasm = adapter._extract_qasm_block(reply)
    if not qasm:
        print(f"[FAIL] {prompt[:20]} -> 回复中没有 QASM")
        return False
    _, dist = l2_oracle.simulate_statevector(qasm)
    fid = l2_oracle.hellinger_fidelity(dist, expected)
    top = {k: f"{v*100:.1f}%" for k, v in sorted(dist.items(), key=lambda kv: -kv[1])[:4]
           if v > 0.005}
    ok = fid >= 0.97
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {prompt[:24]:<24} fidelity={fid:.4f} 分布={top} ({time.time()-t0:.0f}s)")
    if not ok:
        print(f"       回复尾部: {reply[-60:]!r}")
    return ok


def main():
    print("=" * 66)
    print("L2 真实 API 自测：4 项保真度要求 + 12 类路由覆盖")
    print("=" * 66)

    # 方案要求的 4 项保真度测试
    tests = [
        ("生成 3 比特 W 态", {"001": 1/3, "010": 1/3, "100": 1/3}),
        ("生成 3 比特 GHZ 态", {"000": 0.5, "111": 0.5}),
        ("生成 Bell 态", {"00": 0.5, "11": 0.5}),
        ("生成一个单比特叠加态", {"0": 0.5, "1": 0.5}),
    ]
    results = [verify(p, e) for p, e in tests]

    # 12 类路由覆盖（不调 LLM，只验证路由命中）
    print("\n--- 12 类意图路由覆盖（仅本地，不消耗 API） ---")
    routing = [
        "生成 2 比特 Bell 态", "生成 4 比特 GHZ 态", "生成 3 比特 W 态",
        "生成 5 比特均匀叠加", "隐形传态把 q0 状态传到 q2",
        "对 3 个比特做量子傅里叶变换", "用 Grover 搜索标记项",
        "Deutsch-Jozsa 判断函数类型", "量子加法器算 2+3",
        "把 q0 纠缠到 q1 并转 0.7 弧度", "生成一个随机量子线路",
    ]
    all_hit = True
    for prompt in routing:
        hit = l2_oracle.classify(prompt)
        if hit is None:
            ok, name = False, "未命中"
        elif hit[3] == "template":
            ok, name = True, hit[2]
        elif hit[3] == "structured":
            ok, name = True, "结构化合成(LLM JSON ops)"
        else:
            ok, name = False, hit[3]
        all_hit = all_hit and ok
        print(f"  [{'OK' if ok else 'MISS'}] {prompt[:24]:<24} -> {name}")

    print("\n" + "=" * 66)
    passed = sum(results)
    print(f"4 项保真度测试: {passed}/4 通过 | 路由覆盖: {'全部命中' if all_hit else '有遗漏'}")
    print("=" * 66)
    sys.exit(0 if passed == 4 and all_hit else 1)


if __name__ == "__main__":
    main()
