"""LangGraph two-node chatbot:
1) check_number  → if user input is a number, reply with number+1
2) call_model    → otherwise call OpenAI LLM via langchain-openai

Includes: my_text context → appended to final assistant message.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI


# ---------- Context: uses my_text ----------

class Context(TypedDict, total=False):
    """You can set 'my_text' in your deployment config.
    It will be appended to every final assistant message.
    """
    my_text: str


# ---------- State ----------

@dataclass
class State:
    messages: List[Dict[str, str]] = field(default_factory=list)
    number_handled: bool = False


# ---------- LLM client (langchain-openai wrapper around OpenAI) ----------

llm = ChatOpenAI(
    model="gpt-5-mini",  # change model name here if you want
    temperature=0
)


# ---------- NODE 1: detect number ----------

async def check_number(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    messages = state.messages
    if not messages:
        return {}

    last = messages[-1]
    if last.get("role") != "user":
        return {}

    text = str(last.get("content", "")).strip()

    # Try to parse number
    try:
        num = float(text)
    except ValueError:
        return {}  # not a number → go to LLM

    result = num + 1

    # Add configurable suffix (my_text)
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
    # Use the existing history from state
    messages = list(state.messages)

    # Call LLM with full history
    response = await llm.ainvoke(messages)
    base_text = response.content

    # Add configurable suffix (my_text)
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
