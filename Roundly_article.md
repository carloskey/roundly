# 我在生产环境中发现了 LangGraph 的一个定时炸弹
## 👉 Replay-safe HITL for LLM agents

> **TL;DR**：LangGraph 的 `interrupt()` 有一个你必须知道的 trade-off：Human-in-the-Loop 审批通过后，已经执行过的工具会**再次执行**。这不是 bug，是 graph execution 模型（checkpoint = node，resume = replay）的自然结果——LangGraph 为确定性可重放的 DAG 优化，而不是为 side-effectful agent loop 优化。如果那个工具是 `restart_service` 或 `send_alert`，你的服务会被重启两次，你的 on-call 会被呼叫两次。这是 LangGraph GitHub Issues #6208、#6533、#6626 记录的已知行为。本文描述我们是如何踩到它的，以及我们用 200 行 Python 在 loop 层面绕过了它。

---

## 背景

我们在构建一个 AI Agent 系统，用于自动处理 IT 告警、分析根因，在需要时执行修复操作（SSH 重启服务、调云 API、写数据库）。

核心工作流是经典的 [ReAct 循环](https://arxiv.org/abs/2210.03629) [1]：

```python
while True:
    response = llm.invoke(messages)       # LLM 推理，选择工具
    if not response.tool_calls:
        return response.content           # 最终答案
    for tc in response.tool_calls:
        result = execute(tc)              # 工具执行
        messages.append(ToolMessage(result))
```

**问题**：有些工具有不可逆的副作用。在执行 `restart_service` 或 `modify_firewall_rule` 之前，我们需要人工审批。

LangGraph 的解决方案是 `interrupt()`——在工具执行过程中暂停，等待人工确认后通过 `Command(resume=...)` 继续。

我们接入了。然后我们发现了这个问题。

---

## 为什么会这样：LangGraph 的执行模型

LangGraph 的设计哲学来自 [Pregel](https://research.google/pubs/pub37252/)——一个为确定性图计算设计的执行模型。它的 checkpoint 粒度是 **node**，resume 语义是 **replay**：

> checkpoint = 节点入口状态  
> resume = 从节点入口重新执行

这个模型非常适合确定性 DAG（比如 MapReduce 风格的数据处理管道）：节点是纯函数，相同输入保证相同输出，replay 无副作用。

**问题在于**：LLM agent 的 tool-calling loop 不是纯函数。工具有副作用，LLM 有非确定性。当你把 `interrupt()` 塞进这样的 loop，Pregel 的 replay 语义就变成了地雷。

`interrupt()` 的工作原理是：

1. 在节点边界把图状态序列化到 checkpointer（SQLite / Redis / PostgreSQL）
2. 挂起执行
3. 收到 `resume` 信号后，**从节点入口重新执行**

关键词：**从节点入口重新执行**。

假设你的工作流是这样的：

```
Round 1: LLM 调用 check_system_metrics → 执行 → 结果写入 messages
Round 2: LLM 调用 restart_service → interrupt() 触发 → 等待审批
Round 2: 审批通过 → resume()
         ↓
         LangGraph: 重新进入 tool_node，从头执行
         ↓
         Round 1 的 check_system_metrics 再次执行
         ↓
         Round 2 的 restart_service 也再次执行
```

这就是 **Double Execution Problem**。

更危险的是并行工具调用场景——LLM 在同一个 round 里同时调用多个工具：

```
LLM 决定并行调用: [fetch_db_stats, fetch_cache_stats]
tool_node 执行 tool[0] (fetch_db_stats) → interrupt()
审批通过 → resume
↓
tool_node 从头重新进入
↓
tool[0] (fetch_db_stats) 再次执行   ← ❌ 双重执行
tool[1] (fetch_cache_stats) 执行
```

如果 `tool[0]` 是 `send_pagerduty_alert`，你的 on-call 工程师会在凌晨 3 点被呼叫**两次**。

这不是我们猜测的。用户在 LangGraph 官方 Issue Tracker 上已经反复报告这一行为：

- [**Issue #6208**](https://github.com/langchain-ai/langgraph/issues/6208)：*"Do not re-execute interrupted node on resume"*
- [**Issue #6533**](https://github.com/langchain-ai/langgraph/issues/6533)：*"ToolNode misroutes resume values to wrong tool"*
- [**Issue #6626**](https://github.com/langchain-ai/langgraph/issues/6626)：*"Parallel interrupt ID collision"*

这些 issue 的存在本身说明：这个 trade-off 对许多用户来说是意外，而非预期。LangGraph 优化了 graph replayability，但没有针对 side-effectful tool loop 提供开箱即用的保护。

---

## 我们用 benchmark 把它重现了

为了让这个问题有铁证，我们写了一个可运行的 benchmark：[`demo.py`](benchmark/demo.py)，对比 Roundly 和 LangGraph 在 4 个实验场景下的行为。

核心是一个 `ExecutionCounter`，精确记录每个工具实际被调用的次数：

```python
class ExecutionCounter:
    def record(self, tool_name: str, round_num: int, framework: str):
        self.calls[tool_name] = self.calls.get(tool_name, 0) + 1
        self.log.append({"tool": tool_name, "round": round_num, "framework": framework})
```

### Experiment 3：并行工具 + interrupt（最直接的复现）

场景：LLM 决定并行调用 `fetch_db_stats` 和 `fetch_cache_stats`。HITL 策略要求执行前审批。

**LangGraph 的行为**（[`langgraph_runner.py`](benchmark/langgraph_runner.py)，`interrupt_mode="inside"`）：

```
tool_node 执行:
  Step 1: fetch_db_stats → 执行 ✓
  Step 2: interrupt() — 等待审批...
  (审批通过)
  tool_node 从头重进入:
  Step 1: fetch_db_stats → 再次执行 ← ❌ DOUBLE EXECUTION
  Step 2: fetch_cache_stats → 执行 ✓
```

**实验结果**：

```
┌─────────────────────────────────────────────────────────────────┐
│  Tool Execution Count after interrupt → resume                  │
├───────────────────────────────┬─────────────┬──────────────────┤
│  Tool                         │ Roundly       │ LangGraph        │
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

## LangGraph 的"安全模式"也不是真正的解法

LangGraph 有另一种 HITL 模式：`interrupt_before=["tool_node"]`，在节点边界暂停，而不是在节点内部。这避免了 Double Execution——但引入了另一个问题：

**每次进入 tool_node 都需要审批**。

如果你的工作流有 3 轮：
- Round 1：`check_system_metrics`（只读，不需要审批）
- Round 2：`read_logs`（只读，不需要审批）  
- Round 3：`restart_service`（危险，需要审批）

`interrupt_before` 模式会在 3 个 round 都要求审批。操作员需要点击 3 次"批准"——包括两次纯粹多余的确认。

这是 Experiment 1 的结果（三路对比）：

```
┌─────────────────────────────────────────────────────────────────┐
│  Tool Execution Count (ground truth via ExecutionCounter)       │
├───────────────────────┬────────────┬────────────┬──────────────┤
│  Framework            │ metrics    │ restart    │ Approvals    │
├───────────────────────┼────────────┼────────────┼──────────────┤
│  Roundly                │ ✅ 1x      │ ✅ 1x      │ 1 (selective)│
│  LangGraph [safe]     │ ✅ 1x      │ ✅ 1x      │ ~2 (every)   │
│  LangGraph [inside]   │ ✅ 1x      │ ❌ 2x 🔥  │ 1 (replays!) │
└───────────────────────┴────────────┴────────────┴──────────────┘
```

**结论**：在 side-effectful agent loop 里，LangGraph 给你两个选项——双重执行，或者过度打扰操作员。Roundly 是第三条路：在 loop 内部操作，完全绕开 graph replay 语义。

---

## 非确定性让问题更难捉摸

有人可能会说：`temperature=0`，LLM 总是选相同的工具，replay 也没关系。

这是错的。

当 `temperature > 0`（生产系统通常如此），LangGraph 的 node replay 会**重新调用 LLM**。在 Experiment 2 中，我们用 `temperature=0.7` 跑了 3 次：

```
Run 1: ['check_system_metrics', 'check_system_metrics', 'read_logs', 'restart_service']
                                 ↑ 这是 replay 导致的重复

Run 2: ['check_system_metrics', 'read_logs', 'restart_service']  ← 不同序列!
Run 3: ['read_logs', 'check_system_metrics', 'restart_service']  ← 又不同!
```

这意味着：

> **同一个告警，同一个审批流程，Run 1 重启了 nginx，Run 2 可能重启了 postgres。**

这不是 non-determinism——这是**不可预测的系统状态变更**。

---

## Roundly：在 Round 边界暂停，而不是节点边界

Roundly 的核心思路很简单：

**不要在节点边界暂停。在 round 边界暂停。**

更具体地说：在 LLM 选好工具之后、任何工具执行之前暂停。把完整的 message chain（包含所有历史 tool 结果）序列化存储。Resume 时，从 `round_num + 1` 继续——永远不回头。

```python
# 在工具执行之前，检查是否需要人工审批
if pre_tool_check and pre_tool_check(round_num, tool_calls):
    messages.append(response)    # AIMessage (含 tool_calls) 进入 chain
    raise PreToolApprovalPause(
        tool_calls=tool_calls,
        messages_serialized=_serialize_messages(messages),  # 完整冻结
        round_num=round_num,
        total_tokens=total_tokens,
        ...
    )
```

Resume 时：

```python
# 反序列化完整的 message chain
messages = _deserialize_messages(frozen["messages_serialized"])
start_round = frozen["round_num"] + 1    # ← 从这里继续，永不回头

# 执行审批通过的工具（它们从未执行过）
for tc in frozen["pending_tool_calls"]:
    result = await tool_executor(tc["name"], tc["args"])
    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

# 从 N+1 轮继续
for round_num in range(start_round, MAX_ROUNDS + 1):
    response = await llm_with_tools.ainvoke(messages)
    ...
```

**为什么这从结构上消除了 Double Execution？**

关键在于 `messages_serialized`。这个 list 里已经包含了历史所有 round 的 `ToolMessage` 结果。Resume 时，loop 从 `start_round` 开始，根本不会经过之前的 round——那些 round 的结果已经在 message chain 里了，LLM 可以直接读取，不需要重新执行。

---

## Message Chain 序列化的细节

这个机制能工作的关键在于**完整、正确地序列化 LangChain message 对象**——特别是 `tool_call_id` 绑定：

```python
def _serialize_messages(msgs) -> list[dict]:
    for m in msgs:
        if isinstance(m, AIMessage):
            d = {"role": "ai", "content": m.content or ""}
            if m.tool_calls:
                d["tool_calls"] = m.tool_calls   # 保留 tool_call_id
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({
                "role": "tool",
                "content": m.content,
                "tool_call_id": m.tool_call_id   # AIMessage ↔ ToolMessage 绑定
            })
```

为什么 `tool_call_id` 这么重要？OpenAI API 验证每个 `ToolMessage` 必须引用一个有效的 `tool_call_id`——来自对应的 `AIMessage`。如果这个绑定断裂，Resume 后第一次 LLM 调用就会报 API 错误。

冻结状态存储为自包含的 JSON，可以被任意进程反序列化继续执行：

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

注意：`check_system_metrics` 的结果（`"CPU: 94%, Memory: 78%"`）已经作为 `ToolMessage` 存在于 chain 里了。Resume 时没有任何理由再次调用它。

---

## Token 双重计费问题

这是一个容易被忽视的副作用。

当 LangGraph resume 时，它重新进入 agent_node，再次调用 LLM。这次 LLM 调用产生的 token 被加到总计数里——但其中包含了 pause 之前已经计费过的 token（因为 message chain 是从头重建的）。

结果：**同一段推理被计费两次**。

Roundly 的解法是一个去重公式：

```
T_billed = T_pause + max(0, T_resume_cumulative - T_pause)
```

Experiment 4 的实际测量：

```
┌──────────────────────────────┬──────────────┬──────────────────┐
│  Metric                      │ LangGraph    │ Roundly            │
├──────────────────────────────┼──────────────┼──────────────────┤
│  Tokens billed               │ 1996         │ 1982             │
│  Deduplication applied       │ ❌ No        │ ✅ Yes           │
│  Billing inflation           │ +14 tok      │ 0                │
└──────────────────────────────┴──────────────┴──────────────────┘

Scale projection:
  100 incidents/day   → +1,400 extra tokens/day
  3000 incidents/month → +42,000 extra tokens/month
  ≈ +$0.01/month wasted (at $0.20/1M)
```

单次看起来很小，但这是**每一次 HITL pause/resume cycle** 的浪费，而且随着工作流层数增加线性增长。

---

## 与其他框架的对比

| 特性 | LangGraph interrupt() | LangGraph interrupt_before | OpenAI Agents SDK | Roundly |
|------|----------------------|--------------------------|-------------------|-------|
| Double Execution | ❌ 存在 | ✅ 无 | ⚠️ 需二次 LLM 调用 | ✅ 无 |
| 粒度 | Node 级别 | Node 级别 | Tool 级别 | Round 级别 |
| 选择性审批 | ❌ 困难 | ❌ 每次都审 | ✅ Per-tool | ✅ 可自定义 |
| 跨进程持久化 | ✅ Checkpointer | ✅ Checkpointer | ⚠️ 依赖 OpenAI | ✅ 任意存储 |
| Token 去重 | ❌ 无 | ✅ 无需（无 replay） | ❌ 无 | ✅ 内置公式 |
| Provider 无关 | ❌ LangGraph 绑定 | ❌ LangGraph 绑定 | ❌ OpenAI 绑定 | ✅ 任意 LLM |
| 并行工具正确性 | ❌ Issue #6533 | ✅ 无问题 | ⚠️ 未验证 | ✅ 原子暂停 |

[AEGIS](https://arxiv.org/abs/2603.12621) [2] 和 [AgentSpec](https://arxiv.org/abs/2503.18666) [3] 是学术界两个相关工作，分别提供安全拦截层和声明式策略执行。它们关注的是"拦截"，不关注"幂等恢复"。[Restate](https://www.restate.dev/blog/building-durable-agents-with-vercel-and-restate) [4] 提供持久化 async promise，是目前最接近 Roundly 持久性保证的方案，但需要额外基础设施，且未集成到 AI 框架中。

---

## 生产环境部署

Roundly 目前运行在生产环境中，每天处理数千个 IT 告警的审批流程，涵盖：

- SSH 命令执行审批（`ssh_exec_write` 类工具）
- 云 API 变更授权（AWS/Azure 修改操作）
- 知识库写入确认

暂停时长可以是几秒（Web UI 审批）到几小时（等待 on-call 工程师）。整个等待期间，Celery worker 是空闲的——没有阻塞线程，没有轮询，没有资源浪费。

一个值得记录的实现细节：当 `_handle_pre_tool_pause()` 写入 `ApprovalRequest` 到 PostgreSQL 时，必须开一个独立的 `asyncpg` 连接，而不能复用 pipeline 的 SQLAlchemy session——那个 session 正在持有 `PipelineExecution` 行的事务，复用会导致 `"another operation is in progress"` 错误。这类细节是框架级 HITL 实现通常不会处理的。

---

## 局限性

Roundly 不是万能的：

**进程崩溃后的幂等性**：如果进程在审批通过后、工具执行前崩溃，Roundly 会在重启后重试该工具调用。对于非幂等工具，这仍然可能造成重复执行。计划的缓解方案是在每次工具调用前写入 `tool_execution_started` tombstone。

**Message chain 增长**：长工作流会产生较大的序列化 chain。我们没有截断 `messages_serialized`——截断会破坏 `tool_call_id` 绑定，导致 API 错误。

**多 agent 协调**：当前实现处理单个 agent node 的暂停。多个并行 agent node 同时等待审批的场景是计划中的扩展。

---

## 代码

完整的 benchmark 代码（可直接运行）：

```
benchmark/
├── demo.py           — CLI 入口
├── roundly_core.py     — Roundly 核心实现（~230 行）
├── langgraph_runner.py — LangGraph 双模式 runner
└── experiments.py    — 4 个对比实验
```

运行方式：

```bash
cd benchmark/
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o-mini   # 或任何 OpenAI 兼容模型

python demo.py --exp 3   # 直接看最震撼的 Double Execution 实验
python demo.py           # 运行全部 4 个实验
```

[`roundly_core.py`](benchmark/roundly_core.py) 的核心逻辑不依赖 LangGraph，可以独立集成到任何基于 LangChain 的 agent 中。

---

## 结论

LangGraph 是一个优秀的框架——它为确定性、可重放的 graph 执行做了很好的优化。这正是它的设计目标。

但 **side-effectful agent loop 不是确定性 graph**。当你在一个会触发真实世界副作用的 tool-calling loop 里使用 `interrupt()`，Pregel 的 replay 语义就成了负担：checkpoint 粒度是 node，resume 从 node 入口重跑，已执行的工具必然再执行一次。

Roundly 不修改 LangGraph，不对抗它的设计——它在 loop 内部操作，把暂停边界下移到 **round 层面**：LLM 选定工具后、任何工具执行前序列化完整 message chain，Resume 从 `N+1` round 继续。历史 round 的所有结果已在 chain 里，loop 永远不会回到它们。Double Execution 被从结构上排除。

**两个模型，两个适用场景：**

| | LangGraph interrupt | Roundly |
|---|---|---|
| 适合 | 确定性 DAG，纯函数节点 | Side-effectful tool loop |
| Checkpoint 粒度 | Node | Round（LLM inference 边界） |
| Resume 语义 | Replay from node entry | Continue from round N+1 |
| 副作用安全 | 需额外保护 | 结构性保证 |

如果你在用 LangGraph 构建需要 HITL 的 AI agent，先用 `interrupt_before=["tool_node"]` 规避最明显的风险；如果你需要精细的选择性审批或并行工具场景下的正确性，可以考虑 Roundly 的 round-level 方案。

---

## 参考文献

[1] Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models*. arXiv:2210.03629

[2] He et al. (2026). *AEGIS: A Framework-Agnostic Pre-Execution Security Firewall for LLM Tool Calling*. arXiv:2603.12621

[3] Zhang et al. (2025). *AgentSpec: Runtime Enforcement for LLM Agents Using a Declarative Policy Language*. arXiv:2503.18666

[4] Restate, Inc. (2025). *Building Durable Agents with Vercel AI and Restate*. https://www.restate.dev/blog/building-durable-agents-with-vercel-and-restate

[5] LangGraph Issue #6208: *Do not re-execute interrupted node on resume*. https://github.com/langchain-ai/langgraph/issues/6208

[6] LangGraph Issue #6533: *ToolNode misroutes resume values to wrong tool*. https://github.com/langchain-ai/langgraph/issues/6533

[7] LangGraph Issue #6626: *Parallel interrupt ID collision*. https://github.com/langchain-ai/langgraph/issues/6626

---

*代码：[`benchmark/`](benchmark/)*
