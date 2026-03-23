"""
Production stop-condition guards for agent evaluation runs.

Guards are callables that receive the current agent state and raise
GuardTripped when a safety condition is violated.  The evaluator
checks guards after every step, before proceeding to the next one.

Available guards:
  - MaxStepsGuard      — hard cap on reasoning/tool-call steps
  - CostCeilingGuard   — abort when accumulated spend exceeds a dollar limit
  - LoopDetectionGuard — abort when the agent repeats the same action
  - CompositeGuard     — chain multiple guards together
"""

from __future__ import annotations

from collections import Counter
from typing import Protocol


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class GuardTripped(Exception):
    """Raised by a guard when the agent must be stopped."""

    def __init__(self, guard_name: str, reason: str) -> None:
        self.guard_name = guard_name
        self.reason = reason
        super().__init__(f"[{guard_name}] {reason}")


# ---------------------------------------------------------------------------
# Guard protocol — anything implementing __call__ with this signature works
# ---------------------------------------------------------------------------

class Guard(Protocol):
    """
    A callable that inspects the current step and raises GuardTripped
    if the run must stop.

    Parameters
    ----------
    step        : 1-based index of the step just completed
    action      : string description of what the agent did in this step
    cost_so_far : accumulated cost in USD for this run
    """

    def __call__(self, step: int, action: str, cost_so_far: float) -> None:
        ...


# ---------------------------------------------------------------------------
# Concrete guards
# ---------------------------------------------------------------------------

class MaxStepsGuard:
    """
    Abort when the agent exceeds ``max_steps`` reasoning/tool-call steps.

    Why this matters: Agents can enter planning loops where they keep
    re-evaluating the same sub-goal.  A hard step cap prevents runaway
    token spend and gives you a clean failure signal.

    Example
    -------
    >>> guard = MaxStepsGuard(max_steps=20)
    >>> guard(step=21, action="...", cost_so_far=0.0)
    GuardTripped: [MaxStepsGuard] Step limit reached: 21 > 20
    """

    name = "max_steps"

    def __init__(self, max_steps: int = 50) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.max_steps = max_steps

    def __call__(self, step: int, action: str, cost_so_far: float) -> None:
        if step > self.max_steps:
            raise GuardTripped(
                self.name,
                f"Step limit reached: {step} > {self.max_steps}",
            )


class CostCeilingGuard:
    """
    Abort when the accumulated cost for a single run exceeds ``ceiling_usd``.

    Why this matters: A misbehaving agent can exhaust your API budget in
    minutes.  This guard gives you a financial circuit-breaker per eval run.

    Example
    -------
    >>> guard = CostCeilingGuard(ceiling_usd=0.50)
    >>> guard(step=5, action="...", cost_so_far=0.55)
    GuardTripped: [CostCeilingGuard] Cost ceiling exceeded: $0.5500 > $0.5000
    """

    name = "cost_ceiling"

    def __init__(self, ceiling_usd: float = 1.0) -> None:
        if ceiling_usd <= 0:
            raise ValueError("ceiling_usd must be > 0")
        self.ceiling_usd = ceiling_usd

    def __call__(self, step: int, action: str, cost_so_far: float) -> None:
        if cost_so_far > self.ceiling_usd:
            raise GuardTripped(
                self.name,
                f"Cost ceiling exceeded: ${cost_so_far:.4f} > ${self.ceiling_usd:.4f}",
            )


class LoopDetectionGuard:
    """
    Abort when the agent repeats the same action more than ``max_repeats`` times.

    Why this matters: Agents that fail to make progress often cycle through
    the same tool call repeatedly (e.g., re-fetching a URL after getting a
    404).  This guard detects the pattern and exits cleanly.

    ``action`` should be a normalized string representation of what the agent
    did — typically "<tool_name>:<first 80 chars of args>".  The evaluator is
    responsible for constructing this string; the guard just counts repeats.

    Example
    -------
    >>> guard = LoopDetectionGuard(max_repeats=3)
    >>> for i in range(4):
    ...     guard(step=i+1, action="search:python docs", cost_so_far=0.0)
    GuardTripped: [LoopDetectionGuard] Repeated action 4× (max 3): 'search:python docs'
    """

    name = "loop_detected"

    def __init__(self, max_repeats: int = 3) -> None:
        if max_repeats < 1:
            raise ValueError("max_repeats must be >= 1")
        self.max_repeats = max_repeats
        self._counts: Counter[str] = Counter()

    def reset(self) -> None:
        """Clear state between eval cases."""
        self._counts.clear()

    def __call__(self, step: int, action: str, cost_so_far: float) -> None:
        self._counts[action] += 1
        count = self._counts[action]
        if count > self.max_repeats:
            raise GuardTripped(
                self.name,
                f"Repeated action {count}× (max {self.max_repeats}): {action!r}",
            )


class CompositeGuard:
    """
    Chain multiple guards.  The first one to trip wins.

    Example
    -------
    >>> guard = CompositeGuard([
    ...     MaxStepsGuard(max_steps=30),
    ...     CostCeilingGuard(ceiling_usd=0.25),
    ...     LoopDetectionGuard(max_repeats=3),
    ... ])
    """

    def __init__(self, guards: list[Guard]) -> None:
        self.guards = guards

    def reset(self) -> None:
        for g in self.guards:
            if hasattr(g, "reset"):
                g.reset()

    def __call__(self, step: int, action: str, cost_so_far: float) -> None:
        for guard in self.guards:
            guard(step=step, action=action, cost_so_far=cost_so_far)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def default_guards(max_steps: int = 50, cost_ceiling: float = 1.0) -> CompositeGuard:
    """Return a sane default guard stack for production use."""
    return CompositeGuard([
        MaxStepsGuard(max_steps=max_steps),
        CostCeilingGuard(ceiling_usd=cost_ceiling),
        LoopDetectionGuard(max_repeats=4),
    ])
