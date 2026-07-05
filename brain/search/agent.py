"""
Agent: agentic retrieval.

For most queries, the Retriever already handles everything correctly
in one call. The agent's value is specifically for multi-step discovery.
"""

import json
import os
from collections.abc import Generator

import ollama

from brain.search.retriever import Retriever
from brain.search.synthesiser import synthesise, _model_options
from brain.search.language import language_instruction, detect_language
from brain.search.web_search import web_search as do_web_search

_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Simplified single tool -- the model calls it with what it wants to find
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the user's personal documents. Pass the COMPLETE original "
                "user question as the query -- do not rephrase or summarise it. "
                "The search engine handles keywords, file names, years, and topics automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The complete question to search for.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for current information not in the user's documents.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

_SYSTEM = """You are a personal knowledge assistant.

To answer questions, call search_documents with the COMPLETE original question.
Do NOT rephrase, shorten, or summarise the question -- pass it exactly as the user wrote it.
The search engine is smart and handles keywords, file names, years automatically.

After getting search results:
- The excerpts ARE the document content. Read them carefully.
- Russian text in excerpts is valid content -- read and use it.
- Give a detailed, comprehensive answer based on ALL the excerpts.
- Do not call more tools unless the results are clearly empty.
- Respond in the same language the user asked in.
"""


class Agent:
    def __init__(self, retriever: Retriever | None = None):
        self._retriever = retriever or Retriever()

    def run(
        self, query: str, use_web: bool = False
    ) -> Generator[dict, None, None]:
        lang = detect_language(query)

        # FAST PATH: if the Retriever can handle this in one call
        # (has keyword/file/year signals), skip the tool loop entirely.
        # This is more reliable than letting the model call the tool.
        has_signal = self._has_clear_signal(query)
        if has_signal and not use_web:
            yield from self._direct_retrieval(query)
            return

        # AGENT PATH: for multi-step or web+local queries
        yield from self._agent_loop(query, lang, use_web)

    def _has_clear_signal(self, query: str) -> bool:
        """Return True if the Retriever has a deterministic retrieval path."""
        import re

        _YEAR = re.compile(r"\b(19[5-9]\d|20[0-3]\d)\b")
        _EXT = re.compile(r"[\w\- ]*?\.(docx|doc|pdf)", re.IGNORECASE)
        _URL = re.compile(r"https?://\S+")
        _DOM = re.compile(
            r"\b[a-z0-9-]+\.(?:com|org|net|ru|io)\b", re.IGNORECASE
        )
        _QUOT = re.compile(r'"[^"]{2,}"|\'[^\']{2,}\'')
        _BARE = re.compile(
            r"\b(?:from|on|at)\s+([a-z][a-z0-9-]{4,})\b", re.IGNORECASE
        )
        _NOT = {
            "from",
            "have",
            "read",
            "articles",
            "were",
            "where",
            "what",
            "that",
            "this",
            "with",
            "been",
            "more",
            "when",
            "them",
            "they",
            "here",
            "there",
            "about",
            "after",
            "back",
            "only",
            "some",
            "also",
            "just",
            "your",
            "doing",
        }
        _REFL = {
            "conclusions",
            "conclusion",
            "reflections",
            "thoughts",
            "summary",
            "итоги",
            "выводы",
            "мысли",
        }

        if _YEAR.search(query):
            return True
        if _EXT.search(query):
            return True
        if _URL.search(query):
            return True
        if _DOM.search(query):
            return True
        if _QUOT.search(query):
            return True
        if any(t in query.lower() for t in _REFL):
            return True
        for m in _BARE.finditer(query):
            if m.group(1).lower() not in _NOT:
                return True
        return False

    def _direct_retrieval(self, query: str) -> Generator[dict, None, None]:
        """Retrieve and synthesise in one shot, bypassing the tool loop."""
        results = self._retriever.retrieve(query)
        fname = (
            results[0].file_path.split("\\")[-1] if results else "documents"
        )
        yield {
            "type": "tool_call",
            "tool": "search_documents",
            "arguments": {"query": query},
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
        yield {
            "type": "tool_result",
            "tool": "search_documents",
            "result": {"found": len(results)},
        }

        for token in synthesise(query, chunks):
            yield {"type": "token", "content": token}

    def _agent_loop(
        self, query: str, lang: str, use_web: bool
    ) -> Generator[dict, None, None]:
        """Full tool-calling loop for multi-step queries."""
        system = language_instruction(query) + "\n\n" + _SYSTEM
        tools = (
            _TOOLS
            if use_web
            else [t for t in _TOOLS if t["function"]["name"] != "web_search"]
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ]

        all_chunks: list[dict] = []
        all_web: list[dict] = []
        calls_made: set[str] = set()

        for _ in range(4):
            response = ollama.chat(
                model=_MODEL,
                messages=messages,
                tools=tools,
                options=_model_options() or None,
            )

            message = response["message"]
            tool_calls = message.get("tool_calls")

            if not tool_calls:
                content = message.get("content", "")
                if content:
                    yield {"type": "token", "content": content}
                else:
                    yield from self._synthesise_chunks(
                        query, all_chunks, all_web
                    )
                return

            messages.append(message)

            for call in tool_calls:
                name = call["function"]["name"]
                args = call["function"]["arguments"]
                call_key = f"{name}::{json.dumps(args, sort_keys=True)}"

                if call_key in calls_made:
                    messages.append(
                        {
                            "role": "tool",
                            "content": '{"note": "Already searched. Use previous results to answer."}',
                        }
                    )
                    continue
                calls_made.add(call_key)

                yield {"type": "tool_call", "tool": name, "arguments": args}

                if name == "search_documents":
                    # Always use original query -- model tends to rephrase badly
                    search_q = query
                    results = self._retriever.retrieve(search_q)
                    chunks = [
                        {
                            "file_path": r.file_path,
                            "text": r.text,
                            "score": r.score,
                            "chunk_index": r.chunk_index,
                        }
                        for r in results
                    ]
                    all_chunks.extend(chunks)
                    payload = {
                        "found": len(results),
                        "note": "Results retrieved. Now answer the user's question using ALL excerpts.",
                        "excerpts": [
                            {
                                "file": r.file_path.split("\\")[-1],
                                "preview": r.text[:200],
                            }
                            for r in results[:8]
                        ],
                    }

                elif name == "web_search":
                    web = do_web_search(args.get("query", query))
                    all_web.extend(web)
                    payload = {"results": web}

                else:
                    payload = {"error": f"Unknown: {name}"}

                yield {"type": "tool_result", "tool": name, "result": payload}
                messages.append(
                    {
                        "role": "tool",
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Using ALL the excerpts above, give a comprehensive answer in {lang}. "
                        f"Question: {query}"
                    ),
                }
            )

        yield from self._synthesise_chunks(query, all_chunks, all_web)

    def _synthesise_chunks(
        self, query: str, chunks: list[dict], web: list[dict]
    ) -> Generator[dict, None, None]:
        for token in synthesise(query, chunks, web or None):
            yield {"type": "token", "content": token}
