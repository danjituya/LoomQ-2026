"""L2 review P0 regression checks (from L2-评审报告, fixed in a7a101d).

Guards against two real scoring-losing regressions:
  P0-1 quantifier misparse: "给我一个 6 比特的均匀随机输出" used to route
       to a 1-qubit circuit (the 量词 "一个" matched the singles regex).
  P0-2 backend_select false positive: "用哪个门可以制造叠加态？" used to
       route to backend_select (teaching question -> backend pick = 0).

Also covers the related same-theme fixes: RNG intent routing, explicit
qubit count in the final fallback, and coin-verb singles (丢/抛/扔/掷硬币).

Run: python tests/l2_review_p0_check.py  (exit 0 = all pass)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "starter_kit"))
from l2_oracle import classify  # noqa: E402


def check(desc, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {desc}" + (f"  ({detail})" if detail else ""))
    return ok


def main():
    all_ok = True

    # ---- P0-1: quantifier misparse (review §4.1 reproduced cases) ----
    for q, want in (
        ("给我一个 6 比特的均匀随机输出", "6 比特"),
        ("给我做一个 7 比特的均匀叠加，全部测量", "7 比特"),
        ("用 5 个量子比特生成随机数", "5 比特"),
    ):
        r = classify(q)
        got = r[2] if r else None
        all_ok &= check(
            f"量词: {q!r} -> 应含 {want}",
            got is not None and want in got and r[3] == "template",
            f"got={got}",
        )

    # ---- P0-2: backend_select false positive (review §4.2 reproduced) ----
    r = classify("用哪个门可以制造叠加态？")
    all_ok &= check(
        "backend 误报: '用哪个门可以制造叠加态？' 不应路由到 backend_select",
        r is None or r[3] != "backend_select",
        f"kind={r[3] if r else None}",
    )

    # ---- 防误伤: real single-qubit / real backend prompts still work ----
    for q, want in (
        ("单个硬币，正反面各一半", "单比特"),
        ("随机出 1 或 0", "单比特"),
        ("丢硬币", "单比特"),
        ("抛一枚硬币", "单比特"),
    ):
        r = classify(q)
        got = r[2] if r else None
        all_ok &= check(
            f"单比特防误伤: {q!r} -> {want}",
            got is not None and want in got,
            f"got={got}",
        )
    for q in (
        "用哪个平台跑 20 比特",
        "用哪个模拟器",
        "帮我推荐一个 28 比特电路能跑的后端",
        "我想在真机上跑 10 个比特，选哪个平台？",
    ):
        r = classify(q)
        all_ok &= check(
            f"选后端防误伤: {q!r} -> backend_select",
            r is not None and r[3] == "backend_select",
            f"kind={r[3] if r else None}",
        )

    # ---- 常规回归: 模板族不受影响 ----
    for q, want in (
        ("生成一个 3 比特的 GHZ 态", "GHZ 态(3 比特)"),
        ("生成一个 2 比特的最大纠缠态，全测量", "Bell 态(2 比特)"),
        ("Generate a 3-qubit GHZ state and measure all qubits", "GHZ 态(3 比特)"),
    ):
        r = classify(q)
        got = r[2] if r else None
        all_ok &= check(
            f"常规: {q[:30]!r} -> {want}",
            got is not None and want in got,
            f"got={got}",
        )

    # ---- 同义词/音译覆盖（评审 P1：猫态/cat state 等 GHZ 别名缺失）----
    for q, want in (
        ("构造一个 3 量子比特的猫态并全测量", "GHZ 态(3 比特)"),
        ("create a 4-qubit cat state", "GHZ 态(4 比特)"),
        ("create a 4-qubit cat-state", "GHZ 态(4 比特)"),
        ("薛定谔猫态 3 比特", "GHZ 态(3 比特)"),
        ("薛定谔的猫，4 个比特", "GHZ 态(4 比特)"),
        ("绿伯格-霍恩-泽林格态 3 比特", "GHZ 态(3 比特)"),
        ("格罗弗搜索 3 比特标记 101", "Grover"),
        ("格洛弗算法找 110", "Grover"),
        ("多伊奇-约萨算法：平衡函数", "Deutsch"),
        ("德义奇问题：常数函数", "Deutsch"),
        ("量子传送 2 比特", "隐形传态"),
    ):
        r = classify(q)
        got = r[2] if r else None
        all_ok &= check(
            f"同义词: {q[:30]!r} -> {want}",
            got is not None and want in got,
            f"got={got}",
        )

    # 同义词防误伤: 教学/无关表述不得命中 GHZ
    for q in ("薛定谔方程是什么", "cat 是什么动物"):
        r = classify(q)
        got = r[2] if r else None
        all_ok &= check(
            f"同义词防误伤: {q[:30]!r} 不命中 GHZ",
            got is None or "GHZ" not in got,
            f"got={got}",
        )

    print()
    print("结论:", "ALL PASS" if all_ok else "SOME FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
