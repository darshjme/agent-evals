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

The field has tried a few approaches:

- **Exact match** — too brittle. Any rephrasing fails.
- **Regex / substring** — better, but misses semantic correctness entirely.
- **Human eval** — gold standard, but $$$, slow, and doesn't scale to CI/CD.
- **LLM-as-judge** — powerful, but expensive and non-deterministic itself.

What's missing is a structured, multi-dimensional framework that scores agents across what actually matters in production: **did it complete the task**, **did it reason efficiently**, and **did it use its tools correctly**?

That's what `agent-evals` provides.

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

### 1. Task Completion (40%)

Did the agent accomplish the goal stated in the prompt?

Scoring is tiered:
- **1.0** — exact match or callable validator returns 1.0
- **0.85** — expected answer is a substring of the output (correct but verbose)
- **0.0–0.7** — Jaccard token similarity between expected and actual output
- **1.0** — if `expected_output=None`, the task is open-ended; full credit always

For semantic tasks (summaries, analysis), pass a callable validator:
```python
EvalCase(
    id="summary-01",
    prompt="Summarize the Strands Evals announcement",
    expected_output=lambda output: 0.9 if "agent" in output.lower() else 0.3,
)
```

### 2. Reasoning Quality (30%)

Was the agent's internal reasoning coherent, non-redundant, and efficient?

This dimension evaluates the **trace**, not the output. Four heuristics:
- Trace exists (not empty)
- Every step has a non-empty thought
- Steps taken relative to the configured max (fewer = better)
- Majority of steps have an associated action (thinking + doing)

For teams that want semantic reasoning evaluation, swap in an LLM-as-judge call in `_score_reasoning()`. The interface is designed to make this easy.

### 3. Tool Use Accuracy (30%)

Did the agent call the tools it was supposed to call, without errors?

Scored via F1 of precision/recall against `expected_tools`, with a penalty per failed tool call (e.g., API errors, malformed arguments):

```python
EvalCase(
    id="search-01",
    prompt="Find recent papers on agent evaluation",
    expected_tools=["web_search", "pdf_reader"],
)
```

If `expected_tools=[]`, this dimension always scores 1.0.

---

## Quick start

### Install

```bash
pip install agent-evals          # when published to PyPI
# or
git clone https://github.com/darshjme/agent-evals
cd agent-evals
pip install -e .
```

### Evaluate your agent

```python
from agent_evals import AgentEvaluator, AgentResult, EvalCase, ToolCall, ReasoningStep

# Step 1: Wrap your agent in a function that returns AgentResult
def my_agent(prompt: str) -> AgentResult:
    # Replace with your real agent call
    # (OpenAI Agents SDK, LangGraph, Strands, CrewAI, etc.)
    response = call_your_agent(prompt)
    return AgentResult(
        output=response.final_answer,
        tool_calls=[ToolCall(name=tc.name, arguments=tc.args) for tc in response.tool_calls],
        reasoning_trace=[
            ReasoningStep(step_index=i, thought=s.thought, action=s.action)
            for i, s in enumerate(response.steps)
        ],
        steps_taken=len(response.steps),
        estimated_cost_usd=response.usage.total_cost,
    )

# Step 2: Define your test suite
test_cases = [
    EvalCase(
        id="flight-search",
        prompt="Find the cheapest flight from Mumbai to London next Friday",
        expected_output=lambda r: 0.9 if "£" in r or "GBP" in r else 0.2,
        expected_tools=["flight_search", "price_compare"],
        metadata={"category": "travel", "difficulty": "medium"},
    ),
    EvalCase(
        id="code-review",
        prompt="Review this Python function for security issues: def login(user, pwd): ...",
        expected_output="SQL injection",   # must mention this
        expected_tools=["code_analyzer"],
        metadata={"category": "security", "difficulty": "hard"},
    ),
]

# Step 3: Run the evaluation
evaluator = AgentEvaluator(
    agent_fn=my_agent,
    max_steps=30,          # abort after 30 steps
    cost_ceiling=0.50,     # abort if single run exceeds $0.50
    pass_threshold=0.70,   # cases scoring < 0.70 are failures
)

report = evaluator.evaluate(test_cases)
print(report.summary())

# Step 4: Export for CI/CD
import json
with open("eval_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

### Run the built-in example

```bash
python examples/basic_eval.py
```

Expected output:
```
EvalReport [abc12345]  2026-03-23T...Z
  Model          : unknown
  Cases          : 3  (passed: 2, threshold: 0.7)
  Task completion: 0.7083
  Reasoning      : 0.6458
  Tool use       : 0.7500
  Overall        : 0.7000
  Total cost     : $0.0009
```

---

## Architecture

```
agent-evals/
├── agent_evals/
│   ├── __init__.py      — Public API surface
│   ├── evaluator.py     — AgentEvaluator, AgentResult, scoring logic
│   ├── metrics.py       — EvalCase, EvalReport, CaseResult, ToolCall, ReasoningStep
│   └── guards.py        — MaxStepsGuard, CostCeilingGuard, LoopDetectionGuard
└── examples/
    └── basic_eval.py    — Working demo with mock agent
```

**Data flow:**

```
EvalCase[] ──▶ AgentEvaluator.evaluate()
                    │
                    ├── agent_fn(prompt) ──▶ AgentResult
                    │        ↑
                    │    [your agent here]
                    │
                    ├── CompositeGuard.check() [after each step]
                    │
                    ├── _score_task_completion()
                    ├── _score_reasoning()
                    └── _score_tool_use()
                              │
                              ▼
                         CaseResult[]
                              │
                              ▼
                         EvalReport.finalize()
```

---

## Production guards

Agents in production need circuit breakers. `agent-evals` ships three:

```python
from agent_evals import MaxStepsGuard, CostCeilingGuard, LoopDetectionGuard, CompositeGuard

guard = CompositeGuard([
    MaxStepsGuard(max_steps=25),        # hard step limit
    CostCeilingGuard(ceiling_usd=0.25), # financial circuit-breaker
    LoopDetectionGuard(max_repeats=3),  # detect stuck-in-loop agents
])

evaluator = AgentEvaluator(agent_fn=my_agent, guard=guard)
```

When a guard trips, the case is marked `stopped_by_guard` in the report — distinguishing "failed" from "aborted" is important for debugging.

---

## Real-world examples

### Customer support agent
```python
EvalCase(
    id="refund-policy",
    prompt="I bought this 6 weeks ago and it's broken. Can I get a refund?",
    expected_output=lambda r: 1.0 if any(w in r.lower() for w in ["policy", "30-day", "manager"]) else 0.3,
    expected_tools=["order_lookup", "policy_retriever"],
    metadata={"category": "support", "sla_seconds": 5},
)
```

### Code generation agent
```python
EvalCase(
    id="rest-api",
    prompt="Write a FastAPI endpoint that accepts a JSON body and returns it reversed",
    expected_output="@app.post",   # must include a POST decorator
    expected_tools=["code_writer", "syntax_checker"],
)
```

### Research agent
```python
EvalCase(
    id="aws-strands",
    prompt="What did AWS announce for agent evaluation in March 2026?",
    expected_output=lambda r: 0.9 if "strands" in r.lower() else 0.1,
    expected_tools=["web_search"],
    metadata={"requires_internet": True},
)
```

---

## Prior art and related work

- **[AWS Strands Evals](https://aws.amazon.com/blogs/aws/)** (March 2026) — AWS's native evaluation harness for Strands Agents, focused on task-level success metrics inside the Bedrock ecosystem.
- **[METR](https://metr.org)** — Autonomous replication/capability evaluations for frontier models.
- **[AgentBench](https://llmbench.ai/agent)** — Multi-environment benchmark for LLM-as-agent performance.
- **beam.ai research** — Published work on evaluating agents across tool accuracy and goal completion in production environments.
- **[OpenAI Evals](https://github.com/openai/evals)** — Template-based eval framework for OpenAI models; strong on unit-test-style checks, weaker on multi-step agent traces.

`agent-evals` sits in the gap: it's framework-agnostic, works with any agent (LangGraph, CrewAI, Strands, custom), and emphasises the **three-dimensional** production signal rather than binary pass/fail.

---

## Roadmap

- [ ] `async_evaluate()` — parallel case execution
- [ ] LLM-as-judge integration for semantic scoring
- [ ] HTML/Markdown report generation
- [ ] Pytest plugin (`pytest-agent-evals`)
- [ ] Weights & Biases / MLflow logging callbacks
- [ ] Drift detection across eval runs

---

## Contributing

Issues and PRs welcome. Please open an issue first for significant changes.

```bash
git clone https://github.com/darshjme/agent-evals
cd agent-evals
pip install -e ".[dev]"
pytest
```

---

## License

MIT © [Darshankumar Joshi](https://github.com/darshjme)

---

## Related Projects

Building production-grade agent systems? These companion repos complete the stack:

- **[react-guard-patterns](https://github.com/darshjme/react-guard-patterns)** — Stop-condition guards for ReAct agents. Prevent the infinite loops you'd otherwise have to debug.
- **[llm-router](https://github.com/darshjme/llm-router)** — Semantic task routing. Dispatch to specialist agents before you even need to evaluate them.

All three repos by [@darshjme](https://github.com/darshjme).
