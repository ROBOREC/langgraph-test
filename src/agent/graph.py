"""LangGraph single-node graph template turned into a simple chatbot.

It:
- accepts a list of chat messages in the state
- sends full history to OpenAI
- appends the assistant's reply to the history
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

from openai import AsyncOpenAI


# ---------- Context (optional, configurable from deployment) ----------

class Context(TypedDict, total=False):
    """Context parameters for the agent.

    You can set these when creating assistants OR when invoking the graph.
    For example you can pass a system prefix or mode flag.
    """

    my_configurable_param: str  # e.g. "teacher-mode", "debug", etc.


# ---------- State definition (this is what the graph receives & returns) ----------

@dataclass
class State:
    """Chat state: full history of messages.

    Each message is a dict like:
      {"role": "user" | "assistant" | "system", "content": "text"}

    When you invoke the graph, you pass:
      {"messages": [{"role": "user", "content": "Hello"}]}
    """

    messages: List[Dict[str, str]] = field(default_factory=list)


# ---------- OpenAI client setup (newest models live here) ----------

client = AsyncOpenAI()  # reads OPENAI_API_KEY from env


# ---------- Node logic: call OpenAI with message history ----------

async def call_model(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    """Single node of the graph.

    - Reads message history from state
    - Optionally uses runtime.context["my_configurable_param"]
    - Calls OpenAI with the full history
    - Appends assistant reply to state
    """
    messages = list(state.messages)  # copy

    # Optional: contextual system prompt based on config
    cfg = (runtime.context or {}).get("my_configurable_param")
    if cfg:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    f"Special mode: {cfg}."
                ),
            },
            *messages,
        ]

    # Call the latest OpenAI chat model (you can switch to gpt-4.1, o3, etc.)
    completion = await client.chat.completions.create(
        model="gpt-5-mini",  # <-- change model name here if you want
        messages=messages,
    )

    assistant_message = completion.choices[0].message

    # Convert SDK message object to plain dict and append to history
    new_messages = state.messages + [
        {
            "role": assistant_message.role,
            "content": assistant_message.content,
        }
    ]

    return {"messages": new_messages}


# ---------- Graph definition ----------

graph = (
    StateGraph(State, context_schema=Context)
    .add_node(call_model)                 # node name auto = "call_model"
    .add_edge("__start__", "call_model")  # START -> node
    .compile(name="Chatbot Graph")
)
