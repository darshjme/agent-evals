"""
Unit tests for agent_evals — evaluator, metrics, and guards.

All tests are self-contained; no external LLM calls required.
Uses only mock/synthetic data.
"""

from __future__ import annotations

import pytest

from agent_evals.evaluator import AgentEvaluator, AgentResult
from agent_evals.guards import (
    CostCeilingGuard,
    GuardTripped,
    LoopDetectionGuard,
    MaxStepsGuard,
    CompositeGuard,
)
from agent_evals.metrics import (
    CaseResult,
    EvalCase,
    EvalReport,
    ReasoningStep,
    ToolCall,
)


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_evaluator(**kwargs) -> AgentEvaluator:
    """Return an evaluator with a no-op agent (overridden per test)."""
    def _noop(prompt: str) -> AgentResult:
        return AgentResult(output="noop")

    defaults = dict(max_steps=10, cost_ceiling=1.0)
    defaults.update(kwargs)
    return AgentEvaluator(agent_fn=_noop, **defaults)


def _simple_trace(n: int) -> list[ReasoningStep]:
    """Return n synthetic reasoning steps, each with thought + action."""
    return [
        ReasoningStep(step_index=i, thought=f"Thinking step {i}", action=f"action_{i}")
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1. Task completion scoring
# ---------------------------------------------------------------------------

class TestTaskCompletionScoring:
    """Tests for AgentEvaluator._score_task_completion."""

    def setup_method(self):
        self.ev = _make_evaluator()

    def test_exact_match_returns_1(self):
        score = self.ev._score_task_completion("Paris", "Paris")
        assert score == 1.0, f"Expected 1.0 for exact match, got {score}"

    def test_exact_match_case_insensitive(self):
        score = self.ev._score_task_completion("PARIS", "paris")
        assert score == 1.0

    def test_no_match_returns_low_score(self):
        score = self.ev._score_task_completion("London", "Paris")
        # No token overlap → Jaccard = 0.0
        assert score == 0.0, f"Expected 0.0 for no match, got {score}"

    def test_none_expected_returns_1(self):
        score = self.ev._score_task_completion("anything", None)
        assert score == 1.0

    def test_substring_match_returns_0_85(self):
        score = self.ev._score_task_completion("The capital of France is Paris!", "Paris")
        assert score == 0.85, f"Substring match should return 0.85, got {score}"

    def test_callable_expected_perfect(self):
        """Callable expected returning 1.0 → score is 1.0."""
        score = self.ev._score_task_completion("output", lambda x: 1.0)
        assert score == 1.0

    def test_callable_expected_zero(self):
        """Callable expected returning 0.0 → score is 0.0."""
        score = self.ev._score_task_completion("output", lambda x: 0.0)
        assert score == 0.0

    def test_partial_token_overlap(self):
        """Partial token match produces Jaccard score between 0 and 1."""
        score = self.ev._score_task_completion("dog cat bird", "dog cat fish elephant")
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# 2. Reasoning scoring
# ---------------------------------------------------------------------------

class TestReasoningScoring:
    """Tests for AgentEvaluator._score_reasoning — fewer steps = higher score."""

    def setup_method(self):
        # max_steps=10 so efficiency is meaningful
        self.ev = _make_evaluator(max_steps=10)

    def test_fewer_steps_higher_score(self):
        """1 step should score higher than 8 steps (same trace length)."""
        trace = _simple_trace(3)
        score_few = self.ev._score_reasoning(trace, steps_taken=1)
        score_many = self.ev._score_reasoning(trace, steps_taken=8)
        assert score_few > score_many, (
            f"Fewer steps should yield higher score: {score_few} vs {score_many}"
        )

    def test_empty_trace_returns_neutral(self):
        """Empty trace → neutral 0.5 (can't evaluate)."""
        score = self.ev._score_reasoning([], steps_taken=0)
        assert score == 0.5

    def test_full_trace_with_actions_scores_high(self):
        """Good trace: all thoughts filled, all actions present, few steps."""
        trace = _simple_trace(2)
        score = self.ev._score_reasoning(trace, steps_taken=1)
        assert score >= 0.75, f"High-quality trace should score ≥ 0.75, got {score}"

    def test_trace_without_actions_scores_lower(self):
        """Trace where no step has an action should score lower."""
        trace = [
            ReasoningStep(step_index=i, thought=f"Thought {i}", action=None)
            for i in range(3)
        ]
        trace_with_actions = _simple_trace(3)
        score_no_actions = self.ev._score_reasoning(trace, steps_taken=3)
        score_with_actions = self.ev._score_reasoning(trace_with_actions, steps_taken=3)
        assert score_with_actions > score_no_actions

    def test_max_steps_reached_penalises_efficiency(self):
        """steps_taken == max_steps → efficiency = 0 → lower score."""
        trace = _simple_trace(2)
        score_max = self.ev._score_reasoning(trace, steps_taken=10)  # max_steps=10
        score_min = self.ev._score_reasoning(trace, steps_taken=1)
        assert score_min > score_max


# ---------------------------------------------------------------------------
# 3. Tool use scoring
# ---------------------------------------------------------------------------

class TestToolUseScoring:
    """Tests for AgentEvaluator._score_tool_use."""

    def setup_method(self):
        self.ev = _make_evaluator()

    def _make_tool_calls(self, names: list[str]) -> list[ToolCall]:
        return [ToolCall(name=n, arguments={}) for n in names]

    def test_correct_tools_returns_1(self):
        calls = self._make_tool_calls(["search", "calculator"])
        score = self.ev._score_tool_use(calls, expected_tools=["search", "calculator"])
        assert score == 1.0

    def test_wrong_tools_returns_0(self):
        calls = self._make_tool_calls(["wrong_tool"])
        score = self.ev._score_tool_use(calls, expected_tools=["search"])
        assert score == 0.0, f"Wrong tools should return 0.0, got {score}"

    def test_no_expected_tools_returns_1(self):
        """If no tools are required, score is always 1.0."""
        calls = self._make_tool_calls(["anything"])
        score = self.ev._score_tool_use(calls, expected_tools=[])
        assert score == 1.0

    def test_partial_tools_returns_partial_score(self):
        """Called only half the required tools → score between 0 and 1."""
        calls = self._make_tool_calls(["search"])
        score = self.ev._score_tool_use(calls, expected_tools=["search", "calculator"])
        assert 0.0 < score < 1.0

    def test_error_on_tool_call_penalises_score(self):
        """Tool calls with errors reduce score."""
        calls = [
            ToolCall(name="search", arguments={}, error="timeout"),
            ToolCall(name="calculator", arguments={}, error=None),
        ]
        score_with_error = self.ev._score_tool_use(
            calls, expected_tools=["search", "calculator"]
        )
        calls_no_error = [ToolCall(name="search", arguments={}), ToolCall(name="calculator", arguments={})]
        score_no_error = self.ev._score_tool_use(
            calls_no_error, expected_tools=["search", "calculator"]
        )
        assert score_no_error > score_with_error


# ---------------------------------------------------------------------------
# 4. MaxStepsGuard
# ---------------------------------------------------------------------------

class TestMaxStepsGuard:
    """Agent stops when step exceeds max_steps."""

    def test_within_limit_does_not_raise(self):
        guard = MaxStepsGuard(max_steps=5)
        for step in range(1, 6):  # steps 1-5 inclusive
            guard(step=step, action="act", cost_so_far=0.0)  # should not raise

    def test_exceeds_limit_raises(self):
        guard = MaxStepsGuard(max_steps=5)
        with pytest.raises(GuardTripped) as exc_info:
            guard(step=6, action="act", cost_so_far=0.0)
        assert exc_info.value.guard_name == "max_steps"
        assert "6 > 5" in str(exc_info.value)

    def test_exactly_at_limit_does_not_raise(self):
        guard = MaxStepsGuard(max_steps=5)
        guard(step=5, action="act", cost_so_far=0.0)  # should be fine

    def test_invalid_max_steps_raises_value_error(self):
        with pytest.raises(ValueError):
            MaxStepsGuard(max_steps=0)

    def test_evaluator_records_stopped_by_guard(self):
        """End-to-end: evaluator records stopped_by_guard when steps exceed max."""
        def over_budget_agent(prompt: str) -> AgentResult:
            return AgentResult(
                output="I went on forever",
                reasoning_trace=_simple_trace(3),
                steps_taken=15,  # > max_steps=5
            )

        ev = AgentEvaluator(agent_fn=over_budget_agent, max_steps=5)
        case = EvalCase(id="steps-test", prompt="run long")
        report = ev.evaluate([case])
        result = report.results[0]
        assert result.stopped_by_guard == "max_steps", (
            f"Expected stopped_by_guard='max_steps', got {result.stopped_by_guard!r}"
        )


# ---------------------------------------------------------------------------
# 5. CostCeilingGuard
# ---------------------------------------------------------------------------

class TestCostCeilingGuard:
    """Agent stops when accumulated cost exceeds ceiling."""

    def test_below_ceiling_does_not_raise(self):
        guard = CostCeilingGuard(ceiling_usd=1.0)
        guard(step=1, action="act", cost_so_far=0.99)  # should not raise

    def test_above_ceiling_raises(self):
        guard = CostCeilingGuard(ceiling_usd=0.50)
        with pytest.raises(GuardTripped) as exc_info:
            guard(step=1, action="act", cost_so_far=0.55)
        assert exc_info.value.guard_name == "cost_ceiling"

    def test_exactly_at_ceiling_does_not_raise(self):
        guard = CostCeilingGuard(ceiling_usd=1.0)
        guard(step=1, action="act", cost_so_far=1.0)  # exactly at ceiling → no raise

    def test_invalid_ceiling_raises_value_error(self):
        with pytest.raises(ValueError):
            CostCeilingGuard(ceiling_usd=0.0)
        with pytest.raises(ValueError):
            CostCeilingGuard(ceiling_usd=-1.0)

    def test_evaluator_records_stopped_by_cost(self):
        """End-to-end: evaluator records stopped_by_guard='cost_ceiling'."""
        def expensive_agent(prompt: str) -> AgentResult:
            return AgentResult(
                output="I am expensive",
                reasoning_trace=_simple_trace(2),
                steps_taken=2,
                estimated_cost_usd=5.0,  # > ceiling 1.0
            )

        ev = AgentEvaluator(agent_fn=expensive_agent, max_steps=50, cost_ceiling=1.0)
        case = EvalCase(id="cost-test", prompt="spend a lot")
        report = ev.evaluate([case])
        result = report.results[0]
        assert result.stopped_by_guard == "cost_ceiling", (
            f"Expected 'cost_ceiling', got {result.stopped_by_guard!r}"
        )


# ---------------------------------------------------------------------------
# 6. LoopDetectionGuard
# ---------------------------------------------------------------------------

class TestLoopDetectionGuard:
    """Agent stops when repeating the same action more than max_repeats times."""

    def test_no_loop_does_not_raise(self):
        guard = LoopDetectionGuard(max_repeats=3)
        for i in range(3):
            guard(step=i + 1, action="search:python docs", cost_so_far=0.0)

    def test_loop_raises_on_fourth_call(self):
        guard = LoopDetectionGuard(max_repeats=3)
        for i in range(3):
            guard(step=i + 1, action="search:python docs", cost_so_far=0.0)
        with pytest.raises(GuardTripped) as exc_info:
            guard(step=4, action="search:python docs", cost_so_far=0.0)
        assert exc_info.value.guard_name == "loop_detected"

    def test_different_actions_do_not_trigger_loop(self):
        guard = LoopDetectionGuard(max_repeats=2)
        guard(step=1, action="search:A", cost_so_far=0.0)
        guard(step=2, action="search:B", cost_so_far=0.0)
        guard(step=3, action="search:C", cost_so_far=0.0)
        # None repeated > max_repeats → should not raise

    def test_reset_clears_state(self):
        guard = LoopDetectionGuard(max_repeats=2)
        for i in range(2):
            guard(step=i + 1, action="same_action", cost_so_far=0.0)
        guard.reset()
        # After reset, counter cleared — should not raise for the first 2 again
        guard(step=1, action="same_action", cost_so_far=0.0)
        guard(step=2, action="same_action", cost_so_far=0.0)

    def test_invalid_max_repeats_raises(self):
        with pytest.raises(ValueError):
            LoopDetectionGuard(max_repeats=0)

    def test_evaluator_loop_detection_end_to_end(self):
        """End-to-end: evaluator detects loop via repeated actions in trace."""
        repeated_action = "search:same query"

        def looping_agent(prompt: str) -> AgentResult:
            # 6 steps, all using the same action — triggers loop after max_repeats
            trace = [
                ReasoningStep(step_index=i, thought=f"step {i}", action=repeated_action)
                for i in range(6)
            ]
            return AgentResult(
                output="looping forever",
                reasoning_trace=trace,
                steps_taken=6,
            )

        # max_repeats=4 (from default_guards inside evaluator)
        ev = AgentEvaluator(agent_fn=looping_agent, max_steps=50, cost_ceiling=10.0)
        case = EvalCase(id="loop-test", prompt="loop")
        report = ev.evaluate([case])
        result = report.results[0]
        assert result.stopped_by_guard == "loop_detected", (
            f"Expected 'loop_detected', got {result.stopped_by_guard!r}"
        )


# ---------------------------------------------------------------------------
# 7. EvalCase creation
# ---------------------------------------------------------------------------

class TestEvalCaseCreation:
    """Valid EvalCase instantiation and field access."""

    def test_minimal_eval_case(self):
        case = EvalCase(id="c1", prompt="What is 2+2?")
        assert case.id == "c1"
        assert case.prompt == "What is 2+2?"
        assert case.expected_output is None
        assert case.expected_tools == []
        assert case.max_tokens == 2048
        assert case.metadata == {}

    def test_full_eval_case(self):
        case = EvalCase(
            id="c2",
            prompt="Search for Python docs",
            expected_output="python.org",
            expected_tools=["search", "browser"],
            max_tokens=1024,
            metadata={"difficulty": "easy", "category": "search"},
        )
        assert case.expected_output == "python.org"
        assert set(case.expected_tools) == {"search", "browser"}
        assert case.max_tokens == 1024
        assert case.metadata["difficulty"] == "easy"

    def test_eval_case_with_callable_expected(self):
        checker = lambda output: "python" in output.lower()
        case = EvalCase(id="c3", prompt="Tell me about Python", expected_output=checker)
        assert callable(case.expected_output)

    def test_eval_cases_are_independent(self):
        """Mutable defaults (lists/dicts) must not be shared between instances."""
        c1 = EvalCase(id="x", prompt="a")
        c2 = EvalCase(id="y", prompt="b")
        c1.expected_tools.append("tool_a")
        assert "tool_a" not in c2.expected_tools


# ---------------------------------------------------------------------------
# 8. EvalReport summary
# ---------------------------------------------------------------------------

class TestEvalReportSummary:
    """EvalReport aggregates scores correctly after finalize()."""

    def _make_result(self, case_id: str, tc: float, r: float, tu: float, overall: float,
                      cost: float = 0.0) -> CaseResult:
        return CaseResult(
            case_id=case_id,
            prompt="test",
            agent_output="output",
            tool_calls=[],
            reasoning_trace=[],
            task_completion_score=tc,
            reasoning_score=r,
            tool_use_score=tu,
            overall_score=overall,
            estimated_cost_usd=cost,
        )

    def test_finalize_computes_means(self):
        report = EvalReport(run_id="r1", model="gpt-test")
        report.results = [
            self._make_result("c1", 1.0, 0.8, 1.0, 0.94),
            self._make_result("c2", 0.5, 0.6, 0.0, 0.37),
        ]
        report.finalize()

        assert report.total_cases == 2
        assert abs(report.mean_task_completion - 0.75) < 1e-6
        assert abs(report.mean_reasoning - 0.70) < 1e-6
        assert abs(report.mean_tool_use - 0.50) < 1e-6
        assert abs(report.mean_overall - 0.655) < 1e-6

    def test_finalize_counts_passed_cases(self):
        report = EvalReport(run_id="r2", pass_threshold=0.7)
        report.results = [
            self._make_result("pass1", 1.0, 1.0, 1.0, 1.0),
            self._make_result("fail1", 0.0, 0.0, 0.0, 0.0),
            self._make_result("pass2", 0.8, 0.8, 0.8, 0.8),
        ]
        report.finalize()
        assert report.passed_cases == 2
        assert report.total_cases == 3

    def test_finalize_total_cost(self):
        report = EvalReport(run_id="r3")
        report.results = [
            self._make_result("c1", 1.0, 1.0, 1.0, 1.0, cost=0.10),
            self._make_result("c2", 1.0, 1.0, 1.0, 1.0, cost=0.25),
        ]
        report.finalize()
        assert abs(report.total_cost_usd - 0.35) < 1e-9

    def test_finalize_empty_results(self):
        """finalize() on empty results should not crash."""
        report = EvalReport(run_id="r4")
        report.finalize()
        assert report.total_cases == 0
        assert report.mean_overall == 0.0

    def test_summary_string_contains_key_fields(self):
        report = EvalReport(run_id="test-run", model="claude-test")
        report.results = [
            self._make_result("c1", 1.0, 1.0, 1.0, 1.0, cost=0.05),
        ]
        report.finalize()
        summary = report.summary()
        assert "test-run" in summary
        assert "claude-test" in summary
        assert "1.000" in summary  # mean scores
        assert "0.0500" in summary  # total cost

    def test_to_dict_structure(self):
        report = EvalReport(run_id="dict-test", model="m1")
        report.results = [self._make_result("c1", 0.9, 0.8, 0.7, 0.8)]
        report.finalize()
        d = report.to_dict()
        assert d["run_id"] == "dict-test"
        assert "aggregate" in d
        assert d["aggregate"]["total_cases"] == 1
        assert len(d["results"]) == 1
        assert d["results"][0]["case_id"] == "c1"

    def test_full_evaluator_pipeline_integration(self):
        """End-to-end: evaluator runs cases and report aggregates correctly."""
        def perfect_agent(prompt: str) -> AgentResult:
            return AgentResult(
                output="4",
                tool_calls=[ToolCall(name="calculator", arguments={"expr": "2+2"})],
                reasoning_trace=_simple_trace(2),
                steps_taken=2,
                estimated_cost_usd=0.001,
            )

        ev = AgentEvaluator(agent_fn=perfect_agent, max_steps=20, cost_ceiling=1.0)
        cases = [
            EvalCase(id="math1", prompt="What is 2+2?", expected_output="4",
                     expected_tools=["calculator"]),
            EvalCase(id="math2", prompt="What is 3+3?", expected_output="6",
                     expected_tools=["calculator"]),
        ]
        report = ev.evaluate(cases)

        assert report.total_cases == 2
        assert report.mean_task_completion > 0.0
        assert report.mean_tool_use == 1.0  # calculator is expected and called
        assert report.total_cost_usd == pytest.approx(0.002, abs=1e-9)
