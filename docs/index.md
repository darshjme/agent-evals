# agent-evals

**Production-grade LLM agent evaluation framework.**  
3-dimensional scoring: task completion · reasoning quality · tool use accuracy.

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![PyPI](https://img.shields.io/badge/pypi-coming%20soon-orange)](https://pypi.org/project/agent-evals/)

---

## The problem with testing agents

Traditional unit tests work because functions are deterministic. Given the same input, `add(2, 3)` always returns `5`. You assert equality, ship, done.

Agents break this contract. An agent asked "book me a flight to Paris" might succeed by calling a travel API directly, or by web searching for prices first, or by asking a clarifying question, or by failing gracefully with an explanation. **All four can be correct.** None can be validated with `assert output == expected`.

What's missing is a structured, multi-dimensional framework that scores agents across what actually matters in production: **did it complete the task**, **did it reason efficiently**, and **did it use its tools correctly**?

That's what `agent-evals` provides.

---

## Install

```bash
pip install agent-evals
```

Or install from source:

```bash
git clone https://github.com/darshjme/agent-evals
cd agent-evals
pip install -e .
```

---

## Quickstart

```python
from agent_evals import AgentEvaluator, EvalCase

def my_agent(prompt: str) -> dict:
    # Your agent logic here
    return {
        "output": "Paris is the capital of France.",
        "trace": [{"thought": "User asked a geography question", "action": "answer"}],
        "tools_called": [],
        "cost_usd": 0.001,
    }

evaluator = AgentEvaluator(agent_fn=my_agent)

cases = [
    EvalCase(
        id="geo-01",
        prompt="What is the capital of France?",
        expected_output="Paris",
        expected_tools=[],
    ),
]

report = evaluator.evaluate(cases)
report.print_summary()
```

---

## The 3-dimensional model

```
                    ┌─────────────────────────────┐
                    │       AGENT EVAL SCORE       │
                    │                              │
                    │  40% Task Completion         │
                    │  30% Reasoning Quality       │
                    │  30% Tool Use Accuracy       │
                    └─────────────────────────────┘
```

- **Task Completion (40%)** — Did the agent accomplish the goal?
- **Reasoning Quality (30%)** — Was the internal reasoning coherent and efficient?
- **Tool Use Accuracy (30%)** — Did the agent call the right tools without errors?

See the [API Reference](api/evaluator.md) for full details on each dimension.

---

## Related projects

- **[act-guard-patterns](https://github.com/darshjme/act-guard-patterns)** — Stop-condition guards for ReAct agents.
- **[llm-router](https://github.com/darshjme/llm-router)** — Semantic task routing to specialist agents.

All three repos by [@darshjme](https://github.com/darshjme).
