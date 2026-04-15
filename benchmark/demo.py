#!/usr/bin/env python3
"""
Roundly vs LangGraph — HITL Benchmark CLI

Usage:
    python demo.py                          # run all experiments
    python demo.py --exp 1                  # experiment 1 only
    python demo.py --exp 1 2               # experiments 1 and 2
    python demo.py --model gpt-4o          # use a specific model

Environment:
    OPENAI_API_KEY   (required)
    OPENAI_BASE_URL  (optional, for custom/local endpoints)
    OPENAI_MODEL     (optional, default: gpt-4o-mini)

Experiments:
    1 — Double Execution         (core)
    2 — Non-determinism (temp=0.7)
    3 — Parallel tool + interrupt  (Issue #6626)
    4 — Token deduplication billing
"""
import argparse
import asyncio
import os
import sys
import time


def _check_env():
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n❌  OPENAI_API_KEY is not set.")
        print("    Export it first:\n")
        print("        export OPENAI_API_KEY=sk-...\n")
        sys.exit(1)


def _banner():
    print("""
╔══════════════════════════════════════════════════════════════╗
║      Roundly vs LangGraph — HITL Double Execution Benchmark    ║
║                                                              ║
║  Paper: "Sub-Round Idempotent HITL for LLM Tool-Calling"    ║
║  Issue: github.com/langchain-ai/langgraph/issues/6208       ║
╚══════════════════════════════════════════════════════════════╝
""")


async def main():
    parser = argparse.ArgumentParser(
        description="Roundly vs LangGraph HITL benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--exp", nargs="+", default=["1", "2", "3", "4"],
        metavar="N",
        help="Which experiments to run (1 2 3 4). Default: all.",
    )
    parser.add_argument(
        "--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="LLM model to use (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENAI_BASE_URL", ""),
        help="Custom OpenAI-compatible base URL",
    )
    args = parser.parse_args()

    _check_env()
    _banner()

    cfg = {
        "api_key":  os.environ["OPENAI_API_KEY"],
        "base_url": args.base_url,
        "model":    args.model,
        "temperature": 0.0,
    }

    print(f"  Model:    {args.model}")
    print(f"  Base URL: {args.base_url or '(default OpenAI)'}")
    print(f"  Experiments: {args.exp}")
    print()

    # Import here so path resolution works when run from benchmark/ dir
    sys.path.insert(0, os.path.dirname(__file__))
    from experiments import run_all

    t0 = time.monotonic()
    await run_all(args.exp, cfg)
    elapsed = time.monotonic() - t0

    print(f"\n  Total runtime: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    asyncio.run(main())
