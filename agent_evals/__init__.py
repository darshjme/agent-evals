"""
agent-evals — Production-grade LLM agent evaluation framework.

3-dimensional scoring: task completion · reasoning quality · tool use accuracy

Quick start:

    from agent_evals import AgentEvaluator, AgentResult, EvalCase

    def my_agent(prompt: str) -> AgentResult:
        # Call your agent here
        return AgentResult(output="42", steps_taken=1)

    evaluator = AgentEvaluator(agent_fn=my_agent)
    report = evaluator.evaluate([
        EvalCase(id="math-1", prompt="What is 6×7?", expected_output="42"),
    ])
    print(report.summary())
"""

from .evaluator import AgentEvaluator, AgentResult
from .guards import (
    CompositeGuard,
    CostCeilingGuard,
    Guard,
    GuardTripped,
    LoopDetectionGuard,
    MaxStepsGuard,
    default_guards,
)
from .metrics import CaseResult, EvalCase, EvalReport, ReasoningStep, ToolCall

__all__ = [
    # Evaluator
    "AgentEvaluator",
    "AgentResult",
    # Metrics / data models
    "EvalCase",
    "EvalReport",
    "CaseResult",
    "ToolCall",
    "ReasoningStep",
    # Guards
    "Guard",
    "GuardTripped",
    "MaxStepsGuard",
    "CostCeilingGuard",
    "LoopDetectionGuard",
    "CompositeGuard",
    "default_guards",
]

__version__ = "0.1.0"
