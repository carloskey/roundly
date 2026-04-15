"""
Roundly Core — Self-contained implementation of the Round-boundary Idempotent
Human-in-the-Loop mechanism.

No LangGraph dependency. Works with any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ── Execution counter (shared across both frameworks for fair comparison) ────

class ExecutionCounter:
    """Tracks how many times each tool was actually invoked."""
    def __init__(self):
        self.calls: dict[str, int] = {}
        self.log: list[dict] = []

    def record(self, tool_name: str, round_num: int, framework: str):
        self.calls[tool_name] = self.calls.get(tool_name, 0) + 1
        self.log.append({"tool": tool_name, "round": round_num, "framework": framework})

    def reset(self):
        self.calls.clear()
        self.log.clear()


# ── PreToolApprovalPause exception ──────────────────────────────────────────

class PreToolApprovalPause(Exception):
    """Raised inside the tool-calling loop when human approval is needed.
    Carries the full frozen message chain so resume is idempotent.
    """
    def __init__(
        self,
        tool_calls: list[dict],
        messages_serialized: list[dict],
        round_num: int,
        total_tokens: int,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tool_logs: list[dict] = None,
    ):
        self.tool_calls = tool_calls
        self.messages_serialized = messages_serialized
        self.round_num = round_num
        self.total_tokens = total_tokens
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.tool_logs = tool_logs or []


# ── Message serialization ────────────────────────────────────────────────────

def _serialize_messages(msgs) -> list[dict]:
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    out = []
    for m in msgs:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            out.append({"role": "human", "content": m.content})
        elif isinstance(m, AIMessage):
            d: dict[str, Any] = {"role": "ai", "content": m.content or ""}
            if getattr(m, "tool_calls", None):
                d["tool_calls"] = m.tool_calls
            out.append(d)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content,
                        "tool_call_id": m.tool_call_id})
    return out


def _deserialize_messages(data: list[dict]):
    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
    msgs = []
    for d in data:
        role = d["role"]
        if role == "system":
            msgs.append(SystemMessage(content=d["content"]))
        elif role == "human":
            msgs.append(HumanMessage(content=d["content"]))
        elif role == "ai":
            msg = AIMessage(content=d.get("content", ""))
            if d.get("tool_calls"):
                msg.tool_calls = d["tool_calls"]
            msgs.append(msg)
        elif role == "tool":
            msgs.append(ToolMessage(content=d["content"],
                                    tool_call_id=d.get("tool_call_id", "")))
    return msgs


# ── Roundly tool-calling loop ────────────────────────────────────────────────

MAX_ROUNDS = 10


async def roundly_call_with_tools(
    system_prompt: str,
    user_message: str,
    tools_schema: list[dict],
    tool_executor: Callable,             # async (name, args) -> str
    *,
    model: str = "gpt-4o-mini",
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    pre_tool_check: Callable[[int, list[dict]], bool] | None = None,
    resume_from: dict | None = None,
    counter: ExecutionCounter | None = None,
    framework_label: str = "Roundly",
) -> tuple[str, dict]:
    """Run a tool-calling loop with optional Roundly pause/resume.

    Returns (final_text, result_dict).
    result_dict keys: total_tokens, prompt_tokens, completion_tokens, tool_logs,
                      paused (bool), frozen_state (dict if paused).
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

    init_kw: dict[str, Any] = {"model": model, "temperature": temperature}
    if api_key:
        init_kw["api_key"] = api_key
    if base_url:
        init_kw["base_url"] = base_url

    llm = ChatOpenAI(**init_kw).bind_tools(tools_schema)

    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    tool_logs: list[dict] = []

    # ── Resume path ──────────────────────────────────────────────────────────
    if resume_from:
        messages = _deserialize_messages(resume_from["messages_serialized"])
        start_round = resume_from["round_num"] + 1      # ← KEY: skip past interrupted round
        total_tokens = resume_from.get("total_tokens", 0)
        prompt_tokens = resume_from.get("prompt_tokens", 0)
        completion_tokens = resume_from.get("completion_tokens", 0)
        tool_logs = list(resume_from.get("tool_logs", []))

        print(f"\n  [Roundly] Deserializing frozen message chain "
              f"({len(messages)} messages, round_num={resume_from['round_num']})")
        print(f"  [Roundly] Resuming from round {start_round} "
              f"(skipping rounds 1–{resume_from['round_num']})")

        # Execute approved pending tools
        for tc in resume_from.get("pending_tool_calls", []):
            name = tc["name"]
            args = tc.get("args", tc.get("arguments", {}))
            tc_id = tc.get("id", "")
            if counter:
                counter.record(name, resume_from["round_num"], framework_label)
            print(f"  [Roundly] Executing approved tool: {name}({json.dumps(args)})")
            result = await tool_executor(name, args)
            tool_logs.append({"tool": name, "round": resume_from["round_num"],
                               "result": result, "executed_by": "resume"})
            messages.append(ToolMessage(content=result, tool_call_id=tc_id))
    else:
        messages = [SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)]
        start_round = 1

    # ── Main loop ────────────────────────────────────────────────────────────
    for round_num in range(start_round, MAX_ROUNDS + 1):
        response = await llm.ainvoke(messages)

        if response.usage_metadata:
            rt = response.usage_metadata.get("total_tokens", 0)
            total_tokens += rt
            prompt_tokens += response.usage_metadata.get("input_tokens", 0)
            completion_tokens += response.usage_metadata.get("output_tokens", 0)

        tool_calls = getattr(response, "tool_calls", []) or []

        if not tool_calls:
            return response.content, {
                "total_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "tool_logs": tool_logs,
                "paused": False,
            }

        # ── Pre-tool check (Roundly pause) ──────────────────────────────────
        if pre_tool_check and pre_tool_check(round_num, tool_calls):
            messages.append(response)
            print(f"\n  [Roundly] ⏸  PAUSE at round {round_num} — "
                  f"tools: {[tc['name'] for tc in tool_calls]}")
            print(f"  [Roundly] Serializing {len(messages)} messages to frozen state")
            raise PreToolApprovalPause(
                tool_calls=tool_calls,
                messages_serialized=_serialize_messages(messages),
                round_num=round_num,
                total_tokens=total_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tool_logs=tool_logs,
            )

        # ── Execute tools ────────────────────────────────────────────────────
        messages.append(response)
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc.get("id", "")
            if counter:
                counter.record(name, round_num, framework_label)
            print(f"  [Roundly] Round {round_num}: executing {name}({json.dumps(args)})")
            result = await tool_executor(name, args)
            tool_logs.append({"tool": name, "round": round_num, "result": result})
            from langchain_core.messages import ToolMessage as TM
            messages.append(TM(content=result, tool_call_id=tc_id))

    return "Max rounds reached", {
        "total_tokens": total_tokens, "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens, "tool_logs": tool_logs,
        "paused": False,
    }
