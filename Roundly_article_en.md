# There's a Time Bomb in LangGraph's HITL — Here's How We Defused It
## 👉 Replay-safe HITL for LLM agents

> **TL;DR**: LangGraph's `interrupt()` has a trade-off you need to understand: after a Human-in-the-Loop approval, tools that already executed will **execute again**. This isn't a bug — it's the natural consequence of the graph execution model (checkpoint = node, resume = replay). LangGraph is optimized for deterministic, replayable DAGs, not for side-effectful agent loops. If that tool is `restart_service` or `send_alert`, your service gets restarted twice and your on-call engineer gets paged twice. This behavior is documented in LangGraph GitHub Issues [#6208](https://github.com/langchain-ai/langgraph/issues/6208), [#6533](https://github.com/langchain-ai/langgraph/issues/6533), and [#6626](https://github.com/langchain-ai/langgraph/issues/6626). This post describes how we hit it — and how we built a 200-line loop-level fix.

---

## Background

We're building an AI agent system for automated IT incident response: the agent analyzes alerts, traces root causes, and when needed, executes remediation actions (SSH service restarts, cloud API mutations, database writes).

The core workflow is the classic [ReAct loop](https://arxiv.org/abs/2210.03629) [1]:

```python
while True:
    response = llm.invoke(messages)       # LLM reasons, selects tools
    if not response.tool_calls:
        return response.content           # final answer
    for tc in response.tool_calls:
        result = execute(tc)              # tool executes
        messages.append(ToolMessage(result))
```

**The problem**: some tools have irreversible side effects. Before executing `restart_service` or `modify_firewall_rule`, we need human approval.

LangGraph's answer is `interrupt()` — pause inside a tool, wait for human confirmation, then continue via `Command(resume=...)`.

We wired it up. Then we found the problem.

---

## Why This Happens: LangGraph's Execution Model

LangGraph's design philosophy is rooted in [Pregel](https://research.google/pubs/pub37252/) — a graph computation model built for deterministic execution. Its checkpoint granularity is the **node**, and its resume semantics are **replay**:

> checkpoint = state at node entry  
> resume = re-execute from node entry

This model is a great fit for deterministic DAGs (think MapReduce-style data pipelines): nodes are pure functions, same input guarantees same output, replay is side-effect-free.

**The problem**: an LLM agent's tool-calling loop is not a pure function. Tools have side effects. LLMs are non-deterministic. When you insert `interrupt()` into such a loop, Pregel's replay semantics become a landmine.

Here's exactly what `interrupt()` does:

1. Serializes graph state to a checkpointer (SQLite / Redis / PostgreSQL) at the node boundary
2. Suspends execution
3. On `resume`, **re-enters the node from its entry point**

The key phrase: **re-enters the node from its entry point**.

Given this workflow:

```
Round 1: LLM calls check_system_metrics → executes → result in messages
Round 2: LLM calls restart_service → interrupt() fires → awaiting approval
Round 2: Approved → resume()
         ↓
         LangGraph: re-enter tool_node from the top
         ↓
         Round 1's check_system_metrics executes AGAIN
         ↓
         Round 2's restart_service executes AGAIN
```

This is the **Double Execution Problem**.

The parallel tool scenario is even more treacherous — when the LLM calls multiple tools in the same round:

```
LLM decides to call in parallel: [fetch_db_stats, fetch_cache_stats]
tool_node executes tool[0] (fetch_db_stats) → interrupt()
Approved → resume
↓
tool_node re-enters from the top
↓
tool[0] (fetch_db_stats) executes AGAIN   ← ❌ double execution
tool[1] (fetch_cache_stats) executes
```

If `tool[0]` is `send_pagerduty_alert`, your on-call engineer gets paged **twice** at 3am.

We didn't guess this. Users have repeatedly reported this behavior on LangGraph's official Issue Tracker:

- [**Issue #6208**](https://github.com/langchain-ai/langgraph/issues/6208): *"Do not re-execute interrupted node on resume"*
- [**Issue #6533**](https://github.com/langchain-ai/langgraph/issues/6533): *"ToolNode misroutes resume values to wrong tool"*
- [**Issue #6626**](https://github.com/langchain-ai/langgraph/issues/6626): *"Parallel interrupt ID collision"*

The existence of these issues tells the story: this trade-off surprises users — it isn't the expected behavior. LangGraph optimizes for graph replayability, but provides no out-of-the-box protection for side-effectful tool loops.

---

## We Reproduced It with a Benchmark

To put this on solid ground, we built a runnable benchmark: [`demo.py`](benchmark/demo.py), comparing Roundly and LangGraph across 4 experiment scenarios.

The core instrument is an `ExecutionCounter` that tracks actual tool invocations as ground truth:

```python
class ExecutionCounter:
    def record(self, tool_name: str, round_num: int, framework: str):
        self.calls[tool_name] = self.calls.get(tool_name, 0) + 1
        self.log.append({"tool": tool_name, "round": round_num, "framework": framework})
```

### Experiment 3: Parallel Tools + interrupt (the clearest reproduction)

Scenario: the LLM decides to call `fetch_db_stats` and `fetch_cache_stats` in parallel. HITL policy requires approval before execution.

**LangGraph's behavior** ([`langgraph_runner.py`](benchmark/langgraph_runner.py), `interrupt_mode="inside"`):

```
tool_node executes:
  Step 1: fetch_db_stats → executes ✓
  Step 2: interrupt() — awaiting approval...
  (approved)
  tool_node re-enters from the top:
  Step 1: fetch_db_stats → executes AGAIN ← ❌ DOUBLE EXECUTION
  Step 2: fetch_cache_stats → executes ✓
```

**Measured results**:

```
┌─────────────────────────────────────────────────────────────────┐
│  Tool Execution Count after interrupt → resume                  │
├───────────────────────────────┬─────────────┬──────────────────┤
│  Tool                         │ Roundly     │ LangGraph        │
├───────────────────────────────┼─────────────┼──────────────────┤
│  fetch_db_stats               │ ✅ 1x       │ ❌ 2x ← RE-EXECUTED! │
│  fetch_cache_stats            │ ✅ 1x       │ ✅ 1x            │
└───────────────────────────────┴─────────────┴──────────────────┘

🔥 DOUBLE EXECUTION CONFIRMED  (Issue #6208 / #6533)
   Tool re-executed: ['fetch_db_stats']

   Real-world impact:
   • fetch_db_stats = read-only → harmless but wasteful
   • If tool[0] were restart_service or send_alert:
     → Production service restarted TWICE
     → PagerDuty alert fired TWICE → on-call flooded
     → DB transaction executed TWICE → data corruption
   This is not a bug — it is an INCIDENT.
```

---

## LangGraph's "Safe Mode" Isn't a Real Solution Either

LangGraph offers a second HITL mode: `interrupt_before=["tool_node"]` — pause at the node boundary rather than inside the node. This avoids double execution, but introduces a different problem:

**Every entry into `tool_node` requires approval.**

If your workflow has 3 rounds:
- Round 1: `check_system_metrics` (read-only, no approval needed)
- Round 2: `read_logs` (read-only, no approval needed)
- Round 3: `restart_service` (dangerous, approval required)

`interrupt_before` mode demands approval on all 3 rounds. The operator clicks "Approve" 3 times — including two completely unnecessary confirmations.

Experiment 1 results (three-way comparison):

```
┌─────────────────────────────────────────────────────────────────┐
│  Tool Execution Count (ground truth via ExecutionCounter)       │
├───────────────────────┬────────────┬────────────┬──────────────┤
│  Framework            │ metrics    │ restart    │ Approvals    │
├───────────────────────┼────────────┼────────────┼──────────────┤
│  Roundly              │ ✅ 1x      │ ✅ 1x      │ 1 (selective)│
│  LangGraph [safe]     │ ✅ 1x      │ ✅ 1x      │ ~2 (every)   │
│  LangGraph [inside]   │ ✅ 1x      │ ❌ 2x 🔥  │ 1 (replays!) │
└───────────────────────┴────────────┴────────────┴──────────────┘
```

**The verdict**: for side-effectful agent loops, LangGraph gives you two options — double execution, or over-interrupting the operator. Roundly is the third path: operate inside the loop, bypass graph replay semantics entirely.

---

## Non-Determinism Makes It Worse

Someone might argue: use `temperature=0`, the LLM always picks the same tools, replay is harmless.

Wrong.

When `temperature > 0` (as is common in production), LangGraph's node replay **re-invokes the LLM**. In Experiment 2, we ran 3 times at `temperature=0.7`:

```
Run 1: ['check_system_metrics', 'check_system_metrics', 'read_logs', 'restart_service']
                                 ↑ duplicate caused by replay

Run 2: ['check_system_metrics', 'read_logs', 'restart_service']  ← different sequence!
Run 3: ['read_logs', 'check_system_metrics', 'restart_service']  ← different again!
```

This means:

> **The same alert. The same approval flow. Run 1 restarts nginx. Run 2 might restart postgres.**

This isn't non-determinism. This is **unpredictable production state mutation**.

---

## Roundly: Pause at Round Boundaries, Not Node Boundaries

Roundly's core idea is straightforward:

**Don't pause at the node boundary. Pause at the round boundary.**

Specifically: pause after the LLM selects tools but before any tool executes. Serialize the complete message chain (including all prior tool results). On resume, continue from `round_num + 1` — never look back.

```python
# Before executing any tool, check whether human approval is needed
if pre_tool_check and pre_tool_check(round_num, tool_calls):
    messages.append(response)    # AIMessage (with tool_calls) enters the chain
    raise PreToolApprovalPause(
        tool_calls=tool_calls,
        messages_serialized=_serialize_messages(messages),  # full freeze
        round_num=round_num,
        total_tokens=total_tokens,
        ...
    )
```

On resume:

```python
# Deserialize the complete message chain
messages = _deserialize_messages(frozen["messages_serialized"])
start_round = frozen["round_num"] + 1    # ← continue here, never revisit prior rounds

# Execute the approved pending tools (they have never run)
for tc in frozen["pending_tool_calls"]:
    result = await tool_executor(tc["name"], tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

# Continue from round N+1
for round_num in range(start_round, MAX_ROUNDS + 1):
    response = await llm_with_tools.ainvoke(messages)
    ...
```

**Why does this structurally eliminate double execution?**

The key is `messages_serialized`. That list already contains every `ToolMessage` result from all prior rounds. On resume, the loop starts at `start_round` and never passes through earlier rounds — their results are already in the message chain. The LLM reads them directly. There is no reason to re-execute anything.

---

## The Serialization Detail That Makes It Work

The mechanism depends on **complete, correct serialization of LangChain message objects** — especially `tool_call_id` bindings:

```python
def _serialize_messages(msgs) -> list[dict]:
    for m in msgs:
        if isinstance(m, AIMessage):
            d = {"role": "ai", "content": m.content or ""}
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls   # preserve tool_call_id
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "content": m.content,
                "tool_call_id": m.tool_call_id   # AIMessage ↔ ToolMessage binding
            })
```

Why does `tool_call_id` matter so much? The OpenAI API validates that every `ToolMessage` references a valid `tool_call_id` from a preceding `AIMessage`. Break that binding and the first LLM call after resume throws an API error.

The frozen state is a self-contained JSON blob — any process can deserialize it and resume:

```json
{
  "messages_serialized": [
    {"role": "system", "content": "..."},
    {"role": "human",  "content": "..."},
    {"role": "ai",     "content": "...", "tool_calls": [{"name": "check_system_metrics", "id": "call_xyz", "args": {...}}]},
    {"role": "tool",   "content": "CPU: 94%, Memory: 78%", "tool_call_id": "call_xyz"},
    {"role": "ai",     "content": "...", "tool_calls": [{"name": "restart_service", "id": "call_abc", "args": {...}}]}
  ],
  "round_num": 2,
  "pending_tool_calls": [{"name": "restart_service", "args": {"host": "prod-web-01", "service": "nginx"}, "id": "call_abc"}],
  "total_tokens": 1847
}
```

Note: `check_system_metrics`'s result (`"CPU: 94%, Memory: 78%"`) is already present as a `ToolMessage` in the chain. There is no reason to call it again on resume.

---

## Token Double-Billing

This is an easy-to-miss side effect.

When LangGraph resumes, it re-enters `agent_node` and calls the LLM again. The tokens from that call get added to the running total — but they include tokens already counted at pause time (because the message chain is reconstructed from scratch).

Result: **the same reasoning gets billed twice**.

Roundly's fix is a deduplication formula:

```
T_billed = T_pause + max(0, T_resume_cumulative - T_pause)
```

Experiment 4 measurements:

```
┌──────────────────────────────┬──────────────┬──────────────────┐
│  Metric                      │ LangGraph    │ Roundly          │
├──────────────────────────────┼──────────────┼──────────────────┤
│  Tokens billed               │ 1996         │ 1982             │
│  Deduplication applied       │ ❌ No        │ ✅ Yes           │
│  Billing inflation           │ +14 tok      │ 0                │
└──────────────────────────────┴──────────────┴──────────────────┘

Scale projection:
  100 incidents/day    → +1,400 extra tokens/day
  3,000 incidents/month → +42,000 extra tokens/month
  ≈ +$0.01/month wasted (at $0.20/1M)
```

Small per cycle, but this is waste on **every HITL pause/resume cycle**, scaling linearly with workflow depth.

---

## Framework Comparison

| Feature | LangGraph interrupt() | LangGraph interrupt_before | OpenAI Agents SDK | Roundly |
|---|---|---|---|---|
| Double execution | ❌ Present | ✅ None | ⚠️ Requires 2nd LLM call | ✅ None |
| Granularity | Node level | Node level | Tool level | Round level |
| Selective approval | ❌ Difficult | ❌ Every round | ✅ Per-tool | ✅ Configurable |
| Cross-process persistence | ✅ Checkpointer | ✅ Checkpointer | ⚠️ OpenAI-dependent | ✅ Any store |
| Token deduplication | ❌ None | ✅ Not needed (no replay) | ❌ None | ✅ Built-in formula |
| Provider independence | ❌ LangGraph-bound | ❌ LangGraph-bound | ❌ OpenAI-bound | ✅ Any LLM |
| Parallel tool correctness | ❌ Issue #6533 | ✅ No issue | ⚠️ Unverified | ✅ Atomic pause |

[AEGIS](https://arxiv.org/abs/2603.12621) [2] and [AgentSpec](https://arxiv.org/abs/2503.18666) [3] are related academic work providing security interception layers and declarative policy enforcement. Both focus on *interception*, not *idempotent resume*. [Restate](https://www.restate.dev/blog/building-durable-agents-with-vercel-and-restate) [4] provides durable async promises — the closest existing approach to Roundly's durability guarantees — but requires separate infrastructure and isn't natively integrated into any AI framework.

---

## Production Notes

Roundly runs in production handling thousands of IT alert approval workflows daily, covering:

- SSH command execution approval (`ssh_exec_write`-class tools)
- Cloud API mutation authorization (AWS/Azure modify operations)
- Knowledge base write confirmation

Pause duration ranges from seconds (Web UI approval) to hours (waiting for an on-call engineer). Throughout the wait, the worker process is idle — no blocked threads, no polling, no wasted resources.

One implementation detail worth calling out: when writing an `ApprovalRequest` to PostgreSQL at pause time, you must open a fresh `asyncpg` connection rather than reusing the pipeline's SQLAlchemy session — that session holds an open transaction on the `PipelineExecution` row, and sharing it causes `"another operation is in progress"` errors. This is the kind of production-critical detail that framework-level HITL implementations typically don't handle.

---

## Limitations

Roundly isn't a silver bullet:

**Idempotency on process crash**: if the process crashes after approval is granted but before the approved tool executes, Roundly will retry the tool call on restart. For non-idempotent tools, this can still cause unintended re-execution. The planned mitigation is writing a `tool_execution_started` tombstone to the database before each tool call.

**Message chain growth**: long workflows produce large serialized chains. We don't truncate `messages_serialized` — doing so would break `tool_call_id` bindings and cause API errors on the first LLM call after resume.

**Multi-agent coordination**: the current implementation handles a single agent node pause at a time. Coordinating simultaneous approval requests from multiple parallel agent nodes is a planned extension.

---

## Code

Full runnable benchmark:

```
benchmark/
├── demo.py            — CLI entry point
├── roundly_core.py    — Roundly core (~230 lines)
├── langgraph_runner.py — LangGraph dual-mode runner
└── experiments.py     — 4 comparison experiments
```

To run:

```bash
cd benchmark/
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # or any OpenAI-compatible model

python demo.py --exp 3   # start with the most striking double execution experiment
python demo.py           # run all 4 experiments
```

[`roundly_core.py`](benchmark/roundly_core.py) has no LangGraph dependency and can be integrated into any LangChain-based agent.

---

## Conclusion

LangGraph is a well-designed framework — it does an excellent job of optimizing for deterministic, replayable graph execution. That's exactly what it was built for.

But **side-effectful agent loops are not deterministic graphs**. When you use `interrupt()` inside a tool-calling loop that triggers real-world side effects, Pregel's replay semantics become a liability: checkpoint granularity is the node, resume replays from the node entry, tools that already ran must run again.

Roundly doesn't modify LangGraph or fight its design — it operates inside the loop and shifts the pause boundary down to the **round level**: serialize the complete message chain after the LLM selects tools, before any tool executes. Resume from round `N+1`. All prior round results are already in the chain; the loop never revisits them. Double execution is structurally eliminated.

**Two models, two use cases:**

| | LangGraph interrupt | Roundly |
|---|---|---|
| Best for | Deterministic DAGs, pure-function nodes | Side-effectful tool loops |
| Checkpoint granularity | Node | Round (LLM inference boundary) |
| Resume semantics | Replay from node entry | Continue from round N+1 |
| Side-effect safety | Requires extra protection | Structural guarantee |

If you're building HITL into a LangGraph agent, start with `interrupt_before=["tool_node"]` to sidestep the most obvious risk. If you need selective per-tool approval or correctness guarantees with parallel tool calls, Roundly's round-level approach is worth a look.

---

## References

[1] Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629

[2] He et al. (2026). *AEGIS: A Framework-Agnostic Pre-Execution Security Firewall for LLM Tool Calling*. arXiv:2603.12621

[3] Zhang et al. (2025). *AgentSpec: Runtime Enforcement for LLM Agents Using a Declarative Policy Language*. arXiv:2503.18666

[4] Restate, Inc. (2025). *Building Durable Agents with Vercel AI and Restate*. https://www.restate.dev/blog/building-durable-agents-with-vercel-and-restate

[5] LangGraph Issue #6208: *Do not re-execute interrupted node on resume*. https://github.com/langchain-ai/langgraph/issues/6208

[6] LangGraph Issue #6533: *ToolNode misroutes resume values to wrong tool*. https://github.com/langchain-ai/langgraph/issues/6533

[7] LangGraph Issue #6626: *Parallel interrupt ID collision*. https://github.com/langchain-ai/langgraph/issues/6626

---

*Code: [`benchmark/`](benchmark/)*
