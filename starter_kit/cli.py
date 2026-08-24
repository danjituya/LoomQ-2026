#!/usr/bin/env python3
"""LoomQ Agent CLI - a human-friendly terminal entry for the L2 agent.

Lets a user with zero quantum background drive real quantum circuits using
plain language. Requires LOOMQ_LLM_* environment variables (see README).

Usage:
    export LOOMQ_LLM_BASE_URL=https://api.deepseek.com
    export LOOMQ_LLM_API_KEY=sk-xxx
    export LOOMQ_LLM_MODEL=deepseek-v4-flash
    python cli.py
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    try:
        from adapter import agent_chat, run
    except ImportError:
        from starter_kit.adapter import agent_chat, run

    missing = [
        name
        for name in ("LOOMQ_LLM_BASE_URL", "LOOMQ_LLM_API_KEY", "LOOMQ_LLM_MODEL")
        if not os.environ.get(name)
    ]
    if missing:
        print(
            "缺少 L2 模型配置，请先设置环境变量: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 1

    print("=" * 56)
    print("  LoomQ Agent · 用大白话指挥量子计算机")
    print("  试试输入：\n    - 生成一个 3 比特 GHZ 态\n"
          "    - 帮我修一段报错的量子代码\n"
          "    - 15 比特零排队选哪个平台")
    print("  输入 exit 或 quit 退出。")
    print("=" * 56)

    while True:
        try:
            prompt = input("\n你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if not prompt:
            continue
        if prompt.lower() in ("exit", "quit", "q"):
            print("再见！")
            break
        print("Agent > ", end="", flush=True)
        try:
            reply = agent_chat(prompt)
            print(reply)
        except Exception as exc:  # surface cleanly, never crash on one bad turn
            print(f"(调用出错: {type(exc).__name__}: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
