"""
LangGraph runner — reproduces the Double Execution Problem using
LangGraph's interrupt() INSIDE the ToolNode (after some tools have run).

This is the scenario from Issue #6208 / #6533:
  - Tool node starts, executes tool[0] (e.g. check_system_metrics)
  - Then calls interrupt() to ask for approval before executing tool[1]
  - On resume(), LangGraph replays tool_node from its entry point
  - tool[0] EXECUTES AGAIN — double execution confirmed

We also provide the safer interrupt_before=["tool_node"] variant for comparison,
which avoids double execution but has the "per-tool-batch interrupt" overhead.

References:
  - https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/
  - Issue #6208: nodes replay from entry on resume
  - Issue #6533: tool node double execution on resume
"""
from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from typing_extensions import TypedDict

from roundly_core import ExecutionCounter


class AgentState(TypedDict, total=False):
    messages: list
    final_answer: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    tool_logs: list


def build_langgraph_agent(
    system_prompt: str,
    tools_schema: list[dict],
    tool_executor: Callable,
    *,
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    interrupt_mode: str = "inside",   # "inside" = interrupt() after first tool (Double Exec!)
                                       # "before" = interrupt_before=["tool_node"] (safe)
    interrupt_before_tools: bool = True,   # kept for backwards compat, use interrupt_mode
    counter: ExecutionCounter | None = None,
):
    """Build a LangGraph agent that uses interrupt() for HITL.

    Architecture:
        START → agent_node → tool_node (loops) → END

    interrupt_mode="inside":
        interrupt() is called AFTER executing the first tool in tool_node.
        On resume, tool_node replays from entry — first tool EXECUTES AGAIN.
        This is the Double Execution Problem (Issue #6208).

    interrupt_mode="before":
        compile(interrupt_before=["tool_node"]) interrupts at node boundary.
        On resume, tool_node runs fresh — no prior tools to re-execute.
        Avoids double execution but requires separate interrupt per tool batch.
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from langgraph.checkpoint.memory import MemorySaver

    # backwards compat: interrupt_before_tools=False → no interrupt at all
    if not interrupt_before_tools:
        interrupt_mode = "none"

    init_kw: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_key:
        init_kw["api_key"] = api_key
    if base_url:
        init_kw["base_url"] = base_url

    llm = ChatOpenAI(**init_kw).bind_tools(tools_schema)

    # ── Agent node: calls LLM ────────────────────────────────────────────────
    async def agent_node(state: AgentState) -> AgentState:
        msgs = state.get("messages", [])
        response = await llm.ainvoke(msgs)

        total = state.get("total_tokens", 0)
        prompt = state.get("prompt_tokens", 0)
        compl = state.get("completion_tokens", 0)
        if response.usage_metadata:
            total += response.usage_metadata.get("total_tokens", 0)
            prompt += response.usage_metadata.get("input_tokens", 0)
            compl += response.usage_metadata.get("output_tokens", 0)

        return {
            **state,
            "messages": msgs + [response],
            "total_tokens": total,
            "prompt_tokens": prompt,
            "completion_tokens": compl,
        }

    # ── Tool node (INSIDE mode): execute first tool, THEN interrupt ──────────
    # This is the Double Execution scenario.
    # When resumed, LangGraph replays tool_node from the top → first tool runs again!
    async def tool_node_inside_interrupt(state: AgentState) -> AgentState:
        msgs = state.get("messages", [])
        last_ai = msgs[-1]
        tool_calls = getattr(last_ai, "tool_calls", []) or []
        tool_logs = list(state.get("tool_logs") or [])
        new_msgs = []

        for i, tc in enumerate(tool_calls):
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc.get("id", "")

            # Execute tool i
            if counter:
                counter.record(name, len(tool_logs) + 1, "LangGraph-inside")
            print(f"  [LangGraph-inside] ToolNode: executing {name}({json.dumps(args)})")
            result = await tool_executor(name, args)
            tool_logs.append({"tool": name, "result": result})
            new_msgs.append(ToolMessage(content=result, tool_call_id=tc_id))

            # After each tool (except the last), interrupt() for approval.
            # On resume, this entire tool_node will replay from the top.
            if i < len(tool_calls) - 1:
                remaining = [t["name"] for t in tool_calls[i+1:]]
                print(f"  [LangGraph-inside] interrupt() — next tools: {remaining}")
                print(f"  [LangGraph-inside] ⚠️  On resume, this node REPLAYS from top!")
                interrupt({
                    "executed": name,
                    "pending": remaining,
                    "message": "Approve remaining tool execution?",
                })

        return {**state, "messages": msgs + new_msgs, "tool_logs": tool_logs}

    # ── Tool node (BEFORE mode): standard, interrupt handled at graph level ───
    async def tool_node_safe(state: AgentState) -> AgentState:
        msgs = state.get("messages", [])
        last_ai = msgs[-1]
        tool_calls = getattr(last_ai, "tool_calls", []) or []
        tool_logs = list(state.get("tool_logs") or [])
        new_msgs = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc.get("id", "")

            if counter:
                counter.record(name, len(tool_logs) + 1, "LangGraph-before")
            print(f"  [LangGraph] ToolNode: executing {name}({json.dumps(args)})")
            result = await tool_executor(name, args)
            tool_logs.append({"tool": name, "result": result})
            new_msgs.append(ToolMessage(content=result, tool_call_id=tc_id))

        return {**state, "messages": msgs + new_msgs, "tool_logs": tool_logs}

    # ── Routing ──────────────────────────────────────────────────────────────
    def route_after_agent(state: AgentState) -> str:
        msgs = state.get("messages", [])
        last = msgs[-1] if msgs else None
        if last and getattr(last, "tool_calls", None):
            return "tool_node"
        return END

    # ── Build graph ──────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("agent_node", agent_node)

    if interrupt_mode == "inside":
        builder.add_node("tool_node", tool_node_inside_interrupt)
        compile_kwargs = {}   # No interrupt_before — interrupt() is inside the node
    else:
        builder.add_node("tool_node", tool_node_safe)
        compile_kwargs = {"interrupt_before": ["tool_node"]}

    builder.set_entry_point("agent_node")
    builder.add_conditional_edges("agent_node", route_after_agent,
                                  {"tool_node": "tool_node", END: END})
    builder.add_edge("tool_node", "agent_node")

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer, **compile_kwargs)


async def run_langgraph_with_interrupt(
    graph,
    system_prompt: str,
    user_message: str,
    thread_id: str = "bench-thread-1",
) -> dict:
    """Run the graph, hit the interrupt, then resume (auto-approve)."""
    config = {"configurable": {"thread_id": thread_id}}
    init_state: AgentState = {
        "messages": [SystemMessage(content=system_prompt),
                     HumanMessage(content=user_message)],
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "tool_logs": [],
    }

    # ── Phase 1: run until interrupt ─────────────────────────────────────────
    print("\n  [LangGraph] Phase 1: running until interrupt...")
    result = None
    async for chunk in graph.astream(init_state, config=config):
        result = chunk

    # ── Phase 2: resume (simulate human approval) ────────────────────────────
    print("  [LangGraph] Interrupt hit. Resuming (auto-approve)...")
    async for chunk in graph.astream(
        Command(resume="approved"), config=config
    ):
        result = chunk

    # Drain remaining execution (may hit another interrupt or finish)
    state = graph.get_state(config)
    while state.next:
        print(f"  [LangGraph] Another interrupt at: {state.next} — resuming...")
        async for chunk in graph.astream(Command(resume="approved"), config=config):
            result = chunk
        state = graph.get_state(config)

    final_state = graph.get_state(config).values
    return {
        "total_tokens": final_state.get("total_tokens", 0),
        "prompt_tokens": final_state.get("prompt_tokens", 0),
        "completion_tokens": final_state.get("completion_tokens", 0),
        "tool_logs": final_state.get("tool_logs", []),
        "messages": final_state.get("messages", []),
    }
