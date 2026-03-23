"""
Core evaluator for LLM agent evaluation.

AgentEvaluator runs a batch of EvalCase objects against an agent function,
applies production guards, and scores results across three dimensions:

  1. Task completion  — did the agent produce the expected output?
  2. Reasoning quality — was the reasoning trace coherent and efficient?
  3. Tool use accuracy — did the agent invoke the right tools?

Usage
-----
    from agent_evals import AgentEvaluator, EvalCase

    def my_agent(prompt: str) -> AgentResult:
        ...

    evaluator = AgentEvaluator(agent_fn=my_agent, max_steps=30, cost_ceiling=0.50)
    report = evaluator.evaluate([
        EvalCase(id="t1", prompt="What is 2+2?", expected_output="4"),
    ])
    print(report.summary())
"""

from __future__ import annotations

import time
import traceback
import uuid
from typing import Any, Callable

from .guards import CompositeGuard, GuardTripped, default_guards
from .metrics import CaseResult, EvalCase, EvalReport, ReasoningStep, ToolCall


# ---------------------------------------------------------------------------
# AgentResult — what your agent function must return
# ---------------------------------------------------------------------------

class AgentResult:
    """
    The return type your agent function must produce.

    Attributes
    ----------
    output        : The agent's final text response.
    tool_calls    : Every tool call made during the run (in order).
    reasoning_trace: The agent's internal reasoning steps.
    steps_taken   : How many steps the agent took.
    estimated_cost_usd: Dollar cost for this single run (best-effort estimate).
    model         : Which model/version was used.
    """

    def __init__(
        self,
        output: str,
        tool_calls: list[ToolCall] | None = None,
        reasoning_trace: list[ReasoningStep] | None = None,
        steps_taken: int = 1,
        estimated_cost_usd: float = 0.0,
        model: str = "unknown",
    ) -> None:
        self.output = output
        self.tool_calls: list[ToolCall] = tool_calls or []
        self.reasoning_trace: list[ReasoningStep] = reasoning_trace or []
        self.steps_taken = steps_taken
        self.estimated_cost_usd = estimated_cost_usd
        self.model = model


# ---------------------------------------------------------------------------
# AgentEvaluator
# ---------------------------------------------------------------------------

class AgentEvaluator:
    """
    Evaluate an agent function against a suite of EvalCase objects.

    Parameters
    ----------
    agent_fn      : Callable that accepts ``(prompt: str, **kwargs)`` and
                    returns an ``AgentResult``.
    max_steps     : Hard cap on reasoning steps per eval case.
    cost_ceiling  : Maximum spend (USD) per eval case before aborting.
    guard         : Override the default guard stack.  Provide your own
                    ``CompositeGuard`` for custom stop conditions.
    score_weights : Tuple (task_w, reasoning_w, tool_w) that must sum to 1.0.
                    Defaults to equal weighting (0.40, 0.30, 0.30).
    pass_threshold: Minimum overall score for a case to be considered passing.
    """

    def __init__(
        self,
        agent_fn: Callable[..., AgentResult],
        max_steps: int = 50,
        cost_ceiling: float = 1.0,
        guard: CompositeGuard | None = None,
        score_weights: tuple[float, float, float] = (0.40, 0.30, 0.30),
        pass_threshold: float = 0.70,
    ) -> None:
        if abs(sum(score_weights) - 1.0) > 1e-6:
            raise ValueError("score_weights must sum to 1.0")

        self.agent_fn = agent_fn
        self.max_steps = max_steps
        self.cost_ceiling = cost_ceiling
        self.guard = guard or default_guards(max_steps=max_steps, cost_ceiling=cost_ceiling)
        self.score_weights = score_weights
        self.pass_threshold = pass_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, test_cases: list[EvalCase]) -> EvalReport:
        """
        Run all test cases and return an EvalReport.

        Execution is sequential by default.  For parallel evaluation,
        wrap this method with asyncio or a ThreadPoolExecutor.
        """
        run_id = str(uuid.uuid4())[:8]
        report = EvalReport(run_id=run_id, pass_threshold=self.pass_threshold)

        for case in test_cases:
            result = self._run_case(case)
            report.results.append(result)

        report.finalize()

        # Infer model from first successful result
        for r in report.results:
            if not r.error:
                # model is on AgentResult, not CaseResult — look it up via agent_fn metadata if available
                break

        return report

    # ------------------------------------------------------------------
    # Private: run a single case
    # ------------------------------------------------------------------

    def _run_case(self, case: EvalCase) -> CaseResult:
        """Execute one eval case, applying guards and scoring."""
        self.guard.reset()
        t0 = time.monotonic()
        stopped_by: str | None = None
        error: str | None = None
        agent_result: AgentResult | None = None

        try:
            agent_result = self.agent_fn(case.prompt)

            # Post-run guard sweep: check each step retroactively.
            # (Real-time guard integration requires the agent to call
            # guard.check() internally — see examples/basic_eval.py.)
            for i in range(agent_result.steps_taken):
                action = self._action_key(agent_result, i)
                try:
                    self.guard(
                        step=i + 1,
                        action=action,
                        cost_so_far=agent_result.estimated_cost_usd,
                    )
                except GuardTripped as gt:
                    stopped_by = gt.guard_name
                    break

        except GuardTripped as gt:
            stopped_by = gt.guard_name
            error = str(gt)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"

        latency_ms = (time.monotonic() - t0) * 1000

        if agent_result is None:
            # Agent crashed before returning anything
            return CaseResult(
                case_id=case.id,
                prompt=case.prompt,
                agent_output="",
                tool_calls=[],
                reasoning_trace=[],
                steps_taken=0,
                estimated_cost_usd=0.0,
                latency_ms=latency_ms,
                stopped_by_guard=stopped_by,
                error=error,
            )

        # Score the three dimensions
        tc_score = self._score_task_completion(agent_result.output, case.expected_output)
        r_score = self._score_reasoning(agent_result.reasoning_trace, agent_result.steps_taken)
        tu_score = self._score_tool_use(agent_result.tool_calls, case.expected_tools)

        w_tc, w_r, w_tu = self.score_weights
        overall = w_tc * tc_score + w_r * r_score + w_tu * tu_score

        return CaseResult(
            case_id=case.id,
            prompt=case.prompt,
            agent_output=agent_result.output,
            tool_calls=agent_result.tool_calls,
            reasoning_trace=agent_result.reasoning_trace,
            task_completion_score=round(tc_score, 4),
            reasoning_score=round(r_score, 4),
            tool_use_score=round(tu_score, 4),
            overall_score=round(overall, 4),
            steps_taken=agent_result.steps_taken,
            estimated_cost_usd=agent_result.estimated_cost_usd,
            latency_ms=round(latency_ms, 2),
            stopped_by_guard=stopped_by,
            error=error,
        )

    # ------------------------------------------------------------------
    # Scoring methods
    # ------------------------------------------------------------------

    def _score_task_completion(self, result: str, expected: Any) -> float:
        """
        Score whether the agent produced the expected output.

        Scoring strategy (applied in order):
          - If expected is None:    return 1.0 (unconstrained task)
          - If expected is callable: return float(expected(result)) clamped to [0,1]
          - If expected is a str:   partial credit based on token overlap
        """
        if expected is None:
            return 1.0

        if callable(expected):
            try:
                score = float(expected(result))
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.0

        if isinstance(expected, str):
            # Exact match → full credit
            if expected.strip().lower() == result.strip().lower():
                return 1.0

            # Substring match → 0.85 credit (the agent answered correctly but verbosely)
            if expected.strip().lower() in result.strip().lower():
                return 0.85

            # Token-level Jaccard similarity for partial credit
            expected_tokens = set(expected.lower().split())
            result_tokens = set(result.lower().split())
            if not expected_tokens:
                return 1.0
            intersection = expected_tokens & result_tokens
            union = expected_tokens | result_tokens
            jaccard = len(intersection) / len(union) if union else 0.0
            return round(jaccard, 4)

        return 0.0

    def _score_reasoning(self, trace: list[ReasoningStep], steps_taken: int) -> float:
        """
        Score the quality of the agent's reasoning trace.

        Heuristics (each contributes equally):
          1. Non-empty trace                  (+0.25)
          2. Each step has a non-empty thought (+0.25 if all steps do)
          3. Efficiency: no wasted steps       (+0.25 based on step count vs max)
          4. Progress: each step has an action (+0.25 if majority do)

        This is intentionally simple.  Replace with an LLM-as-judge call
        for production systems that require semantic quality assessment.
        """
        if not trace:
            # No trace provided — can't evaluate, give neutral score
            return 0.5

        score = 0.0

        # 1. Trace exists
        score += 0.25

        # 2. All thoughts are non-empty
        if all(step.thought.strip() for step in trace):
            score += 0.25

        # 3. Efficiency (fewer steps relative to cap = better)
        efficiency = 1.0 - (steps_taken / self.max_steps) if self.max_steps > 0 else 0.0
        score += 0.25 * max(0.0, efficiency)

        # 4. Most steps have an action (not just thinking, actually doing)
        steps_with_action = sum(1 for s in trace if s.action and s.action.strip())
        if trace:
            action_ratio = steps_with_action / len(trace)
            score += 0.25 * action_ratio

        return round(min(score, 1.0), 4)

    def _score_tool_use(self, tool_calls: list[ToolCall], expected_tools: list[str]) -> float:
        """
        Score tool use accuracy against expected_tools.

        If expected_tools is empty, return 1.0 (no tool requirement).

        Scoring:
          - Recall: fraction of expected tools that were actually called.
          - Precision: fraction of called tools that were expected.
          - F1 of recall and precision.
          - Penalise failed tool calls (those with .error set).
        """
        if not expected_tools:
            return 1.0

        actual_tool_names = [tc.name for tc in tool_calls]
        expected_set = set(expected_tools)
        actual_set = set(actual_tool_names)

        # Recall
        recall = len(expected_set & actual_set) / len(expected_set)

        # Precision
        precision = len(expected_set & actual_set) / len(actual_set) if actual_set else 0.0

        # F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        # Error penalty: each tool call with an error deducts 0.1 (capped at 0.5)
        error_count = sum(1 for tc in tool_calls if tc.error)
        penalty = min(0.5, error_count * 0.1)

        return round(max(0.0, f1 - penalty), 4)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _action_key(result: AgentResult, step_index: int) -> str:
        """Build a normalized action string for loop detection."""
        if step_index < len(result.reasoning_trace):
            step = result.reasoning_trace[step_index]
            action = step.action or ""
            return f"{action[:120]}"
        if step_index < len(result.tool_calls):
            tc = result.tool_calls[step_index]
            args_repr = str(tc.arguments)[:80]
            return f"{tc.name}:{args_repr}"
        return f"step_{step_index}"
