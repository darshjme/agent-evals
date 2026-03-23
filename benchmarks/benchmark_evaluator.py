"""
benchmarks/benchmark_evaluator.py

Performance benchmark for AgentEvaluator.

Runs the evaluator against 10, 50, and 100 EvalCase objects using
a zero-latency mock agent, then prints a clean results table showing
real wall-clock timing captured with time.perf_counter().

Usage
-----
    cd /path/to/agent-evals
    python3 benchmarks/benchmark_evaluator.py
"""

from __future__ import annotations

import sys
import os
import time

# Allow running from the repo root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_evals import AgentEvaluator, AgentResult, EvalCase
from agent_evals.metrics import ReasoningStep, ToolCall


# ---------------------------------------------------------------------------
# Mock agents
# ---------------------------------------------------------------------------

def fast_agent(prompt: str) -> AgentResult:
    """
    Instant mock agent — returns the expected answer immediately.
    No I/O, no sleep: pure CPU throughput benchmark.
    """
    return AgentResult(
        output=f"answer to: {prompt}",
        tool_calls=[
            ToolCall(name="search", arguments={"query": prompt}, result="ok"),
        ],
        reasoning_trace=[
            ReasoningStep(step_index=0, thought="Read the prompt", action="search"),
            ReasoningStep(step_index=1, thought="Formulate answer", action="respond"),
        ],
        steps_taken=2,
        estimated_cost_usd=0.0001,
        model="mock-fast",
    )


# ---------------------------------------------------------------------------
# EvalCase factory
# ---------------------------------------------------------------------------

def make_cases(n: int) -> list[EvalCase]:
    """Generate n EvalCase objects with varied prompts and expected outputs."""
    # Templates defined as static tuples (no f-string interpolation at definition time)
    TOPIC_TEMPLATES = [
        ("math",    "What is {idx} + {idx}?",                 "answer_{idx}"),
        ("capital", "What is the capital of country_{idx}?",  "City_{idx}"),
        ("code",    "Write a function for task_{idx}.",        None),
        ("science", "Explain phenomenon_{idx} briefly.",       None),
        ("history", "When did event_{idx} occur?",            "Year_{idx}"),
    ]
    cases = []
    for i in range(n):
        kind, prompt_tpl, expected_tpl = TOPIC_TEMPLATES[i % len(TOPIC_TEMPLATES)]
        cases.append(
            EvalCase(
                id=f"{kind}-{i:04d}",
                prompt=prompt_tpl.format(idx=i),
                expected_output=expected_tpl.format(idx=i) if expected_tpl else None,
                expected_tools=["search"],
                metadata={"category": kind, "index": i},
            )
        )
    return cases


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(case_counts: list[int], warmup: int = 1) -> list[dict]:
    """
    For each count in case_counts, instantiate a fresh AgentEvaluator,
    run .evaluate(), and record wall-clock elapsed time.

    A short warm-up pass (warmup evaluator runs on 5 cases) is performed
    before measurements begin to avoid cold-import skew.
    """
    evaluator = AgentEvaluator(agent_fn=fast_agent, max_steps=10, cost_ceiling=1.0)

    # Warm-up
    for _ in range(warmup):
        evaluator.evaluate(make_cases(5))

    results = []
    for n in case_counts:
        cases = make_cases(n)
        # Time only the evaluate() call
        t0 = time.perf_counter()
        report = evaluator.evaluate(cases)
        elapsed = time.perf_counter() - t0

        cases_per_sec = n / elapsed if elapsed > 0 else float("inf")
        avg_ms = (elapsed / n) * 1000 if n > 0 else 0.0

        results.append(
            {
                "cases": n,
                "elapsed_s": elapsed,
                "cases_per_sec": cases_per_sec,
                "avg_ms_per_case": avg_ms,
                "pass_rate": report.passed_cases / report.total_cases if report.total_cases else 0,
                "mean_overall": report.mean_overall,
            }
        )
        print(f"  ✓ {n:>4} cases done in {elapsed:.4f}s", flush=True)

    return results


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_table(rows: list[dict]) -> None:
    header = f"{'Cases':>6} | {'Time (s)':>8} | {'Cases/sec':>10} | {'Avg (ms/case)':>14}"
    sep    = "-" * 6 + "-+-" + "-" * 8 + "-+-" + "-" * 10 + "-+-" + "-" * 14
    print()
    print(header)
    print(sep)
    for r in rows:
        print(
            f"{r['cases']:>6} | "
            f"{r['elapsed_s']:>8.4f} | "
            f"{r['cases_per_sec']:>10.1f} | "
            f"{r['avg_ms_per_case']:>14.3f}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    case_counts = [10, 50, 100]

    print("=" * 56)
    print("  agent-evals  —  AgentEvaluator performance benchmark")
    print("=" * 56)
    print(f"\nPython {sys.version}")
    print(f"Benchmark case counts: {case_counts}\n")

    print("Running benchmarks...")
    rows = run_benchmark(case_counts)

    print_table(rows)

    # Summary line for the git commit message
    max_cps = max(r["cases_per_sec"] for r in rows)
    print(f"Peak throughput: {max_cps:,.0f} cases/sec")
    print(f"(rounded for commit: {round(max_cps / 1000):.0f}k cases/sec)\n")
