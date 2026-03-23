<p align="center">
  <img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iODAwIiBoZWlnaHQ9IjE2MCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48ZGVmcz48bGluZWFyR3JhZGllbnQgaWQ9ImJnIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj48c3RvcCBvZmZzZXQ9IjAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMGQxMTE3Ii8+PHN0b3Agb2Zmc2V0PSIxMDAlIiBzdHlsZT0ic3RvcC1jb2xvcjojMTYxYjIyIi8+PC9saW5lYXJHcmFkaWVudD48L2RlZnM+PHJlY3Qgd2lkdGg9IjgwMCIgaGVpZ2h0PSIxNjAiIGZpbGw9InVybCgjYmcpIi8+PHRleHQgeD0iNjAiIHk9Ijc4IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjQ4IiBmb250LXdlaWdodD0iYm9sZCIgZmlsbD0iI2U2ZWRmMyI+dmVyZGljdDwvdGV4dD48dGV4dCB4PSI2MCIgeT0iMTE0IiBmb250LWZhbWlseT0ibW9ub3NwYWNlIiBmb250LXNpemU9IjE2IiBmaWxsPSIjOGI5NDllIj5TY29yZSB5b3VyIGFnZW50cy4gS25vdyB3aGF0IGJyZWFrcy48L3RleHQ+PHBvbHlnb24gcG9pbnRzPSI3MjAsMzAgNzYwLDgwIDc0MCw4MCA3NDAsMTMwIDcwMCwxMzAgNzAwLDgwIDY4MCw4MCIgZmlsbD0iIzNmYjk1MCIgb3BhY2l0eT0iMC45Ii8+PC9zdmc+" alt="verdict" width="800"/>
</p>

<p align="center">
  <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"/></a>
  <img src="https://img.shields.io/badge/tests-45%20passing-brightgreen" alt="Tests"/>
  <img src="https://img.shields.io/badge/throughput-44k%20evals%2Fsec-blue" alt="Throughput"/>
  <img src="https://img.shields.io/badge/pypi-coming%20soon-orange" alt="PyPI"/>
</p>

<p align="center"><b>Score your agents across 3 dimensions — without spinning up another LLM to do it.</b></p>

---

## The Problem with Testing Agents

Traditional unit tests work because functions are deterministic. `add(2, 3)` always returns `5`.

Agents break this contract. An agent asked "book me a flight to Paris" might succeed by calling a travel API, searching for prices first, asking a clarifying question, or failing gracefully. **All four can be correct.** None can be validated with `assert output == expected`.

The field has tried:
- **Exact match** — too brittle. Any rephrasing fails.
- **Regex / substring** — misses semantic correctness entirely.
- **Human eval** — gold standard, but slow and doesn't scale to CI/CD.
- **LLM-as-judge** — powerful, but expensive and non-deterministic itself.

What's missing: a structured, multi-dimensional framework that scores what actually matters in production.

---

## Architecture

```mermaid
flowchart LR
    A[AgentRun] --> B[verdict Scorer]
    B --> C[TaskScore\n0.0–1.0]
    B --> D[ReasoningScore\n0.0–1.0]
    B --> E[ToolScore\n0.0–1.0]
    C --> F[AggregateScore]
    D --> F
    E --> F
    F --> G{≥ threshold?}
    G -->|yes| H[✅ PASS]
    G -->|no| I[❌ FAIL]
```

---

## Quick Start

```bash
git clone https://github.com/darshjme/verdict
cd verdict && pip install -e .
```

```python
from agent_evals import AgentEvaluator, EvalCase

def my_agent(task: str) -> str:
    # your LLM agent here
    return call_llm(task)

evaluator = AgentEvaluator(my_agent, max_steps=50, cost_ceiling=1.0)

report = evaluator.evaluate([
    EvalCase(input="What is 2+2?", expected="4"),
    EvalCase(input="Summarise the README", expected="production agent evaluation"),
    EvalCase(input="Fix the auth bug", expected="returns 200 on valid token"),
])

print(f"Pass rate: {report.pass_rate:.0%}")         # Pass rate: 87%
print(f"Avg task score: {report.avg_task_score:.2f}")
print(f"Avg reasoning: {report.avg_reasoning_score:.2f}")
print(f"Avg tool use: {report.avg_tool_score:.2f}")
```

---

## Three Scoring Dimensions

| Dimension | What It Measures | How |
|-----------|-----------------|-----|
| **Task Completion** | Did the agent actually accomplish the goal? | Semantic match against expected outcome |
| **Reasoning Quality** | Was the chain-of-thought coherent and efficient? | Step analysis — no circular logic, no dead ends |
| **Tool Use Accuracy** | Were the right tools called with the right args? | Tool call inspection against expected tool usage |

Each dimension returns a `float` between `0.0` and `1.0`. The aggregate is a weighted mean.

---

## Benchmark Pipeline

```mermaid
flowchart TD
    A[EvalCase batch\n44k cases] --> B[AgentEvaluator]
    B --> C{Parallel workers}
    C --> D[Worker 1]
    C --> E[Worker 2]
    C --> F[Worker N]
    D --> G[ScoreResult]
    E --> G
    F --> G
    G --> H[EvalReport\n44k evals/sec on commodity HW]
```

**44,000 eval cases/sec** on a standard CPU — no GPU required.

---

## API Reference

| Class | Purpose |
|-------|---------|
| `AgentEvaluator(agent_fn, max_steps, cost_ceiling)` | Main evaluator — wraps your agent |
| `EvalCase(input, expected, tags=[])` | A single test case |
| `ScoreResult` | Per-case scores: task, reasoning, tool, aggregate |
| `EvalReport` | Aggregate report: pass_rate, avg scores, failed cases |
| `DimensionScorer` | Base class — extend to add custom scoring dimensions |

---

## Part of Arsenal

```
verdict · sentinel · herald · engram · arsenal
```

| Repo | Purpose |
|------|---------|
| [verdict](https://github.com/darshjme/verdict) | ← you are here |
| [sentinel](https://github.com/darshjme/sentinel) | ReAct guard patterns — stop runaway agents |
| [herald](https://github.com/darshjme/herald) | Semantic task router — dispatch to specialists |
| [engram](https://github.com/darshjme/engram) | Agent memory — short-term + episodic recall |
| [arsenal](https://github.com/darshjme/arsenal) | Meta-hub — the full pipeline |

---

## License

MIT © [Darshankumar Joshi](https://github.com/darshjme) · Built as part of the [Arsenal](https://github.com/darshjme/arsenal) toolkit.
