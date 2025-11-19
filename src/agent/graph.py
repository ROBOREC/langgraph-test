"""LangGraph two-node chatbot, NO external dependencies.
Uses langgraph.prebuilt.chat_model (built-in wrapper)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from langgraph.prebuilt import chat_model   # <-- built-in model wrapper


# ---------- Context ----------

class Context(TypedDict, total=False):
    my_text: str


# ---------- State ----------

@dataclass
class State:
    messages: List[Dict[str, str]] = field(default_factory=list)
    number_handled: bool = False


# ---------- Built-in LLM (NO external deps) ----------

# This uses the OPENAI_API_KEY automatically
llm = chat_model("gpt-4.1-mini")   # ← You can change model here


# ---------- NODE 1: detect number ----------

async def check_number(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    messages = state.messages
    if not messages:
        return {}

    last = messages[-1]
    if last.get("role") != "user":
        return {}

    text = str(last.get("content", "")).strip()

    try:
        num = float(text)
    except ValueError:
        return {}  # not a number → go to LLM

    result = num + 1

    suffix = (runtime.context or {}).get("my_text", "")
    final_text = f"{result}" + (f" {suffix}" if suffix else "")

    return {
        "messages": messages + [
            {"role": "assistant", "content": final_text}
        ],
        "number_handled": True,
    }


# ---------- NODE 2: LLM fallback ----------

async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    messages = list(state.messages)

    response = await llm.ainvoke(messages)
    base_text = response.content

    suffix = (runtime.context or {}).get("my_text", "")
    final_text = base_text + (f" {suffix}" if suffix else "")

    return {
        "messages": state.messages + [
            {"role": "assistant", "content": final_text}
        ]
    }


# ---------- Router ----------

def router(state: State):
    if state.number_handled:
        return "__end__"
    return "call_model"


# ---------- Graph ----------

graph = (
    StateGraph(State, context_schema=Context)
    .add_node("check_number", check_number)
    .add_node("call_model", call_model)
    .add_edge("__start__", "check_number")
    .add_conditional_edges("check_number", router, ["__end__", "call_model"])
    .compile(name="Chatbot Graph")
)
