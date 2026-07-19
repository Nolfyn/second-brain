"""
LangGraph implementation of the agentic retrieval loop.

Functionally equivalent to agent.py's Agent, expressed as a LangGraph
state machine:

    agent (LLM decides) --tool_calls?--> tools (execute) --> agent
           |
           no tool calls / round cap reached
           v
          END  -> streaming synthesis (outside the graph)

Selected via AGENT_MODE=graph (default remains the classic agent).
"""

import json
import os
from collections.abc import Generator
from typing import Annotated, TypedDict

import ollama
from langgraph.graph import StateGraph, END

from brain.search.retriever import Retriever
from brain.search.synthesiser import synthesise, _model_options
from brain.search.language import language_instruction, detect_language
from brain.search.web_search import web_search as do_web_search

_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Search the user's personal notes and documents. "
            "Pass the user's complete question as the query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for current external information.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

_SYSTEM = """You are a personal knowledge assistant with two tools:
search_documents (the user's notes) and web_search (the live internet).

- Call search_documents with the user's COMPLETE question. Do not rephrase it.
- The text returned IS the document content. Read it, including Russian text.
- After getting results, answer comprehensively. Do not call more tools unless
  the results are empty or you genuinely need the web.
- Never invent facts not in the results. Respond in the user's language.
"""


def _merge_lists(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict):
    messages: list  # ollama chat messages
    original_query: str
    use_web: bool
    chunks: Annotated[list, _merge_lists]  # accumulated retrieval
    web_results: Annotated[list, _merge_lists]
    events: Annotated[list, _merge_lists]  # UI events to emit
    calls_made: list  # dedup keys
    rounds: int


class GraphAgent:
    def __init__(self, retriever: Retriever | None = None):
        self._retriever = retriever or Retriever()
        self._graph = self._build()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", self._tools_node)
        g.set_entry_point("agent")
        g.add_conditional_edges(
            "agent",
            self._route,
            {"tools": "tools", "end": END},
        )
        g.add_edge("tools", "agent")
        return g.compile()

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------

    def _agent_node(self, state: AgentState) -> dict:
        tools = (
            _TOOLS
            if state["use_web"]
            else [t for t in _TOOLS if t["function"]["name"] != "web_search"]
        )
        resp = ollama.chat(
            model=_MODEL,
            messages=state["messages"],
            tools=tools,
            options=_model_options() or None,
        )
        msg = resp["message"]
        return {
            "messages": state["messages"] + [msg],
            "rounds": state["rounds"] + 1,
        }

    def _tools_node(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        new_messages = list(state["messages"])
        new_calls = list(state["calls_made"])
        chunks_delta: list = []
        web_delta: list = []
        events_delta: list = []

        for call in last.get("tool_calls", []):
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            key = f"{name}:{json.dumps(args, sort_keys=True)}"

            if key in new_calls:
                new_messages.append(
                    {"role": "tool", "content": '{"note":"already searched"}'}
                )
                continue
            new_calls.append(key)
            events_delta.append(
                {"type": "tool_call", "tool": name, "arguments": args}
            )

            if name == "search_documents":
                # ALWAYS the original query -- ported behaviour.
                results = self._retriever.retrieve(state["original_query"])
                chunks_delta.extend(
                    {
                        "file_path": r.file_path,
                        "text": r.text,
                        "score": r.score,
                        "chunk_index": r.chunk_index,
                    }
                    for r in results
                )
                payload = {
                    "found": len(results),
                    "note": "Answer using these results now.",
                }
            elif name == "web_search":
                web = do_web_search(args.get("query", state["original_query"]))
                web_delta.extend(web)
                payload = {"results": web}
            else:
                payload = {"error": "unknown tool"}

            events_delta.append(
                {"type": "tool_result", "tool": name, "result": payload}
            )
            new_messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(payload, ensure_ascii=False),
                }
            )

        lang = detect_language(state["original_query"])
        new_messages.append(
            {
                "role": "user",
                "content": f"Answer in {lang} using the results above: "
                f"{state['original_query']}",
            }
        )
        return {
            "messages": new_messages,
            "calls_made": new_calls,
            "chunks": chunks_delta,
            "web_results": web_delta,
            "events": events_delta,
        }

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(self, state: AgentState) -> str:
        if state["rounds"] >= 4:  # ported round cap
            return "end"
        last = state["messages"][-1]
        if last.get("tool_calls"):
            return "tools"
        return "end"

    # ------------------------------------------------------------------
    # Public API -- same contract as Agent.run()
    # ------------------------------------------------------------------

    def run(
        self, query: str, use_web: bool = False
    ) -> Generator[dict, None, None]:
        if not use_web:
            # Fast path, identical to classic agent: no graph needed.
            results = self._retriever.retrieve(query)
            yield {
                "type": "tool_call",
                "tool": "search_documents",
                "arguments": {"query": query},
            }
            yield {
                "type": "tool_result",
                "tool": "search_documents",
                "result": {"found": len(results)},
            }
            chunks = [
                {
                    "file_path": r.file_path,
                    "text": r.text,
                    "score": r.score,
                    "chunk_index": r.chunk_index,
                }
                for r in results
            ]
            for token in synthesise(query, chunks):
                yield {"type": "token", "content": token}
            return

        system = language_instruction(query) + "\n\n" + _SYSTEM
        initial: AgentState = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            "original_query": query,
            "use_web": use_web,
            "chunks": [],
            "web_results": [],
            "events": [],
            "calls_made": [],
            "rounds": 0,
        }

        final = self._graph.invoke(initial)

        # Emit the tool events the graph accumulated, in order.
        yield from final["events"]

        # Final answer: if the model produced direct content with no pending
        # tool calls, stream it; otherwise synthesise from accumulated results.
        last = final["messages"][-1]
        content = (
            last.get("content") if last.get("role") == "assistant" else None
        )
        if content and not last.get("tool_calls"):
            yield {"type": "token", "content": content}
        else:
            for token in synthesise(
                query, final["chunks"], final["web_results"] or None
            ):
                yield {"type": "token", "content": token}
