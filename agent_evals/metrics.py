"""
Metrics and data models for agent evaluation.

EvalCase defines a single test scenario.
EvalReport aggregates results across all test cases.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class ToolCall:
    """Represents a single tool invocation during agent execution."""
    name: str
    arguments: dict[str, Any]
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class ReasoningStep:
    """A single step in the agent's reasoning trace."""
    step_index: int
    thought: str
    action: str | None = None
    observation: str | None = None


@dataclass
class EvalCase:
    """
    A single evaluation test case.

    Attributes:
        id:             Unique identifier for this test case.
        prompt:         The input prompt to send to the agent.
        expected_output: What the agent's final response should contain or match.
                         Can be a string (substring match), a callable (custom check),
                         or None if you only care about tool/reasoning scores.
        expected_tools: List of tool names the agent must call (order-insensitive).
                        Empty list means tool use is not evaluated for this case.
        max_tokens:     Budget hint — evaluation will flag if the agent exceeds this.
        metadata:       Arbitrary dict for tagging (difficulty, category, etc.).
    """
    id: str
    prompt: str
    expected_output: str | None = None
    expected_tools: list[str] = field(default_factory=list)
    max_tokens: int = 2048
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CaseResult:
    """
    Result for a single EvalCase.

    Scores are floats in [0.0, 1.0].
    """
    case_id: str
    prompt: str
    agent_output: str
    tool_calls: list[ToolCall]
    reasoning_trace: list[ReasoningStep]

    # Dimensional scores
    task_completion_score: float = 0.0   # Did the agent accomplish the goal?
    reasoning_score: float = 0.0         # Was the reasoning coherent and efficient?
    tool_use_score: float = 0.0          # Did the agent use the right tools correctly?

    # Aggregate
    overall_score: float = 0.0

    # Operational metadata
    steps_taken: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    stopped_by_guard: str | None = None  # e.g. "max_steps", "cost_ceiling", "loop_detected"
    error: str | None = None


@dataclass
class EvalReport:
    """
    Aggregated evaluation report across all test cases.
    """
    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    model: str = "unknown"

    results: list[CaseResult] = field(default_factory=list)

    # Aggregate stats (populated by EvalReport.finalize())
    mean_task_completion: float = 0.0
    mean_reasoning: float = 0.0
    mean_tool_use: float = 0.0
    mean_overall: float = 0.0

    total_cost_usd: float = 0.0
    total_cases: int = 0
    passed_cases: int = 0       # overall_score >= pass_threshold
    pass_threshold: float = 0.7

    def finalize(self) -> None:
        """Compute aggregate statistics from individual results."""
        if not self.results:
            return

        n = len(self.results)
        self.total_cases = n
        self.mean_task_completion = sum(r.task_completion_score for r in self.results) / n
        self.mean_reasoning = sum(r.reasoning_score for r in self.results) / n
        self.mean_tool_use = sum(r.tool_use_score for r in self.results) / n
        self.mean_overall = sum(r.overall_score for r in self.results) / n
        self.total_cost_usd = sum(r.estimated_cost_usd for r in self.results)
        self.passed_cases = sum(1 for r in self.results if r.overall_score >= self.pass_threshold)

    def summary(self) -> str:
        """Human-readable summary string."""
        lines = [
            f"EvalReport [{self.run_id}]  {self.timestamp}",
            f"  Model          : {self.model}",
            f"  Cases          : {self.total_cases}  (passed: {self.passed_cases}, threshold: {self.pass_threshold})",
            f"  Task completion: {self.mean_task_completion:.3f}",
            f"  Reasoning      : {self.mean_reasoning:.3f}",
            f"  Tool use       : {self.mean_tool_use:.3f}",
            f"  Overall        : {self.mean_overall:.3f}",
            f"  Total cost     : ${self.total_cost_usd:.4f}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serializable dict for JSON export."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "model": self.model,
            "aggregate": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "pass_threshold": self.pass_threshold,
                "mean_task_completion": self.mean_task_completion,
                "mean_reasoning": self.mean_reasoning,
                "mean_tool_use": self.mean_tool_use,
                "mean_overall": self.mean_overall,
                "total_cost_usd": self.total_cost_usd,
            },
            "results": [
                {
                    "case_id": r.case_id,
                    "task_completion_score": r.task_completion_score,
                    "reasoning_score": r.reasoning_score,
                    "tool_use_score": r.tool_use_score,
                    "overall_score": r.overall_score,
                    "steps_taken": r.steps_taken,
                    "estimated_cost_usd": r.estimated_cost_usd,
                    "latency_ms": r.latency_ms,
                    "stopped_by_guard": r.stopped_by_guard,
                    "error": r.error,
                }
                for r in self.results
            ],
        }
