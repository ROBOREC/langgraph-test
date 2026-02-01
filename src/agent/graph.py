from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI  # <-- REQUIRED: Real model driver

# ---------- State ----------

@dataclass
class State:
    messages: List[Dict[str, str]] = field(default_factory=list)
    number_handled: bool = False

# ---------- LLM Setup ----------

# You must have langchain-openai installed and OPENAI_API_KEY in your env
# "gpt-4.1-mini" is not valid; assumed "gpt-4o-mini"
llm = ChatOpenAI(model="gpt-4o-mini") 

# ---------- NODE 1: detect number ----------

async def check_number(state: State, config: RunnableConfig) -> Dict[str, Any]:
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
        return {}  # not a number -> go to LLM

    result = num + 1

    # Access config/context safely
    configurable = config.get("configurable", {})
    suffix = configurable.get("my_text", "")
    
    final_text = f"{result}" + (f" {suffix}" if suffix else "")

    return {
        "messages": messages + [
            {"role": "assistant", "content": final_text}
        ],
        "number_handled": True,
    }

# ---------- NODE 2: LLM fallback ----------

async def call_model(state: State, config: RunnableConfig) -> Dict[str, Any]:
    messages = list(state.messages)

    # Invoke the real LangChain model
    response = await llm.ainvoke(messages)
    base_text = response.content

    configurable = config.get("configurable", {})
    suffix = configurable.get("my_text", "")
    
    final_text = str(base_text) + (f" {suffix}" if suffix else "")

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
    StateGraph(State)
    .add_node("check_number", check_number)
    .add_node("call_model", call_model)
    .add_edge("__start__", "check_number")
    .add_conditional_edges("check_number", router, {
        "__end__": END, 
        "call_model": "call_model"
    })
    .compile()
)
