"""
Basic evaluation example — agent-evals

This example evaluates a mock agent across three realistic test cases:
  1. A simple factual query (task completion focus)
  2. A multi-step calculation requiring tool use
  3. A reasoning-heavy problem

Run with:
    cd /path/to/agent-evals
    python examples/basic_eval.py

No API keys needed — uses a mock agent to demonstrate the framework.
"""

from __future__ import annotations

import json
import sys
import os

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent_evals import (
    AgentEvaluator,
    AgentResult,
    EvalCase,
    ReasoningStep,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Mock agent
# ---------------------------------------------------------------------------
# In production, replace this with your real agent function.
# The function must accept a prompt string and return an AgentResult.

def mock_agent(prompt: str) -> AgentResult:
    """
    A deterministic mock agent for demonstration.

    Real agents would:
      - Call an LLM (OpenAI, Anthropic, Bedrock, etc.)
      - Execute tools (search, code interpreter, databases, APIs)
      - Return the final answer + full trace
    """

    prompt_lower = prompt.lower()

    # Case 1: Simple factual
    if "capital of france" in prompt_lower:
        return AgentResult(
            output="The capital of France is Paris.",
            reasoning_trace=[
                ReasoningStep(
                    step_index=0,
                    thought="The user is asking for a factual geographic fact.",
                    action="recall_knowledge",
                    observation="France's capital is Paris — a well-established fact.",
                ),
            ],
            tool_calls=[],
            steps_taken=1,
            estimated_cost_usd=0.0001,
            model="gpt-4o-mini",
        )

    # Case 2: Calculation with tool use
    if "compound interest" in prompt_lower:
        tool_result = _mock_calculator("1000 * (1 + 0.05) ** 10")
        return AgentResult(
            output=f"After 10 years at 5% annual compound interest, $1,000 grows to ${tool_result:.2f}.",
            reasoning_trace=[
                ReasoningStep(
                    step_index=0,
                    thought="I need to calculate compound interest: A = P(1 + r)^t",
                    action="call_tool:calculator",
                    observation=f"Calculator returned: {tool_result:.2f}",
                ),
                ReasoningStep(
                    step_index=1,
                    thought="The calculation is complete. Format the answer.",
                    action="format_response",
                    observation="Response formatted with dollar sign and two decimal places.",
                ),
            ],
            tool_calls=[
                ToolCall(
                    name="calculator",
                    arguments={"expression": "1000 * (1 + 0.05) ** 10"},
                    result=str(tool_result),
                    latency_ms=12.5,
                ),
            ],
            steps_taken=2,
            estimated_cost_usd=0.0003,
            model="gpt-4o-mini",
        )

    # Case 3: Reasoning — agent gets partial credit (misses one tool)
    if "summarize" in prompt_lower and "search" in prompt_lower:
        return AgentResult(
            output="Agent evaluation frameworks help test LLM agents by measuring task completion, reasoning quality, and tool use accuracy across multiple dimensions.",
            reasoning_trace=[
                ReasoningStep(
                    step_index=0,
                    thought="The user wants a summary of agent evaluation frameworks. I should search for recent developments.",
                    action="call_tool:web_search",
                    observation="Found articles on AWS Strands Evals, beam.ai agent testing research, and METR evals.",
                ),
                ReasoningStep(
                    step_index=1,
                    thought="I have enough context to synthesize a concise summary.",
                    action="synthesize",
                    observation="Summary drafted covering key dimensions of agent evaluation.",
                ),
            ],
            tool_calls=[
                ToolCall(
                    name="web_search",
                    arguments={"query": "LLM agent evaluation frameworks 2025 2026"},
                    result="AWS Strands Evals, beam.ai, METR...",
                    latency_ms=340.0,
                ),
                # Note: expected_tools also includes "summarizer" — agent skipped it
            ],
            steps_taken=2,
            estimated_cost_usd=0.0005,
            model="gpt-4o",
        )

    # Default fallback
    return AgentResult(
        output=f"I received your prompt: '{prompt[:60]}...' but could not determine the appropriate action.",
        steps_taken=1,
        estimated_cost_usd=0.0001,
        model="gpt-4o-mini",
    )


def _mock_calculator(expression: str) -> float:
    """Safe eval for simple math expressions (mock tool)."""
    # In production use a real sandboxed calculator tool
    allowed = set("0123456789 +-*/().**")
    if all(c in allowed for c in expression):
        return eval(expression)  # noqa: S307 — demo only
    raise ValueError(f"Unsafe expression: {expression}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    EvalCase(
        id="geo-001",
        prompt="What is the capital of France?",
        expected_output="Paris",
        expected_tools=[],                # No tools required
        metadata={"category": "factual", "difficulty": "easy"},
    ),
    EvalCase(
        id="math-001",
        prompt="Calculate the compound interest on $1,000 at 5% annual rate for 10 years.",
        expected_output="1628.89",        # Partial match — agent should include this number
        expected_tools=["calculator"],    # Must use calculator tool
        metadata={"category": "math", "difficulty": "medium"},
    ),
    EvalCase(
        id="research-001",
        prompt="Search and summarize the current state of LLM agent evaluation frameworks.",
        expected_output=None,             # Open-ended — no exact expected output
        expected_tools=["web_search", "summarizer"],  # Agent only uses web_search (partial credit)
        metadata={"category": "research", "difficulty": "hard"},
    ),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("agent-evals  —  basic_eval.py")
    print("=" * 60)
    print()

    evaluator = AgentEvaluator(
        agent_fn=mock_agent,
        max_steps=30,
        cost_ceiling=0.10,
        score_weights=(0.40, 0.30, 0.30),
        pass_threshold=0.70,
    )

    report = evaluator.evaluate(TEST_CASES)

    # Print summary
    print(report.summary())
    print()

    # Per-case breakdown
    print("Per-case breakdown:")
    print("-" * 60)
    for result in report.results:
        status = "✓ PASS" if result.overall_score >= report.pass_threshold else "✗ FAIL"
        guard_info = f"  [stopped: {result.stopped_by_guard}]" if result.stopped_by_guard else ""
        error_info = f"  [error: {result.error[:60]}]" if result.error else ""
        print(
            f"  {status}  {result.case_id:<15}"
            f"  task={result.task_completion_score:.2f}"
            f"  reason={result.reasoning_score:.2f}"
            f"  tools={result.tool_use_score:.2f}"
            f"  overall={result.overall_score:.2f}"
            f"  steps={result.steps_taken}"
            f"  cost=${result.estimated_cost_usd:.4f}"
            f"{guard_info}{error_info}"
        )
    print()

    # JSON export (useful for CI pipelines)
    report_dict = report.to_dict()
    print("JSON export (first 500 chars):")
    print(json.dumps(report_dict, indent=2)[:500] + "\n...")
    print()

    # Exit code for CI: non-zero if any case fails
    failures = [r for r in report.results if r.overall_score < report.pass_threshold]
    if failures:
        print(f"⚠  {len(failures)} case(s) below pass threshold ({report.pass_threshold})")
        sys.exit(1)
    else:
        print(f"✓  All {report.total_cases} cases passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
