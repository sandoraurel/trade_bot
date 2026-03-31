from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Dict, List

from .client import ChatModelClient
from .knowledge import LocalKnowledgeBase
from .tools import ToolContext, ToolRegistry, build_default_tool_registry


@dataclass
class AssistantResponse:
    answer: str
    retrieved_context: List[Dict[str, Any]]
    tool_results: List[Dict[str, Any]]


class BotOperatorAssistant:
    """
    RAG + tool-calling assistant for operator-facing questions.

    - Parametric memory: model client
    - Non-parametric memory: local knowledge base
    - Grounding: explicit tool results from the live bot
    """

    def __init__(
        self,
        bot: Any,
        knowledge_dir: str,
        model_client: ChatModelClient | None = None,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.bot = bot
        self.knowledge = LocalKnowledgeBase(knowledge_dir)
        self.model_client = model_client or ChatModelClient()
        self.tools = tool_registry or build_default_tool_registry()

    def answer(self, question: str, top_k: int = 4) -> AssistantResponse:
        retrieved = self.knowledge.search(question, top_k=top_k)
        context = ToolContext(bot=self.bot)
        planned_tools = self._plan_tools(question)
        tool_results: List[Dict[str, Any]] = []

        for tool_name, arguments in planned_tools:
            try:
                result = self.tools.call(tool_name, context, arguments)
                tool_results.append({"tool": tool_name, "arguments": arguments, "result": result})
            except Exception as exc:
                tool_results.append({"tool": tool_name, "arguments": arguments, "error": str(exc)})

        if self.model_client.is_enabled():
            answer = self._model_answer(question, retrieved, tool_results)
        else:
            answer = self._grounded_fallback(question, retrieved, tool_results)

        return AssistantResponse(
            answer=answer,
            retrieved_context=[
                {
                    "doc_id": chunk.document.doc_id,
                    "title": chunk.document.title,
                    "source": chunk.document.source,
                    "score": round(chunk.score, 3),
                    "snippet": chunk.snippet,
                }
                for chunk in retrieved
            ],
            tool_results=tool_results,
        )

    def _plan_tools(self, question: str) -> List[tuple[str, Dict[str, Any]]]:
        lower = question.lower()
        tool_calls: List[tuple[str, Dict[str, Any]]] = [("get_runtime_snapshot", {})]

        for symbol in self.bot.config.symbols:
            if symbol.lower() in lower or symbol.split("/")[0].lower() in lower:
                tool_calls.append(("get_symbol_snapshot", {"symbol": symbol, "timeframe": "15m"}))
                break

        if "position" in lower or "open trade" in lower:
            tool_calls.append(("list_open_positions", {}))

        if "risk" in lower or "halt" in lower or "reconciliation" in lower or "drift" in lower:
            tool_calls.append(("get_risk_state", {}))

        if "news" in lower or "event" in lower or "listing" in lower or "delisting" in lower:
            tool_calls.append(("get_event_research", {}))

        if "readiness" in lower or "promote" in lower or "promotion" in lower or "canary" in lower or "shadow mode" in lower:
            tool_calls.append(("get_readiness_report", {}))

        return tool_calls

    def _model_answer(self, question: str, retrieved: List[Any], tool_results: List[Dict[str, Any]]) -> str:
        knowledge_block = "\n\n".join(
            f"[{idx + 1}] {chunk.document.title}\nSource: {chunk.document.source}\nSnippet: {chunk.snippet}"
            for idx, chunk in enumerate(retrieved)
        ) or "No matching local documents found."

        tool_block = json.dumps(tool_results, ensure_ascii=True, indent=2)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an operator assistant for a trading bot. "
                    "Answer only from retrieved knowledge and tool results. "
                    "If grounded evidence is missing, say that explicitly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved knowledge:\n{knowledge_block}\n\n"
                    f"Tool results:\n{tool_block}"
                ),
            },
        ]
        return self.model_client.complete(messages, temperature=0.1)

    def _grounded_fallback(self, question: str, retrieved: List[Any], tool_results: List[Dict[str, Any]]) -> str:
        lines = [
            f"Question: {question}",
            "Grounded answer only. Model API is not configured, so this response is built from local retrieval and tool outputs.",
        ]

        if retrieved:
            lines.append("Retrieved knowledge:")
            for chunk in retrieved:
                lines.append(f"- {chunk.document.title}: {chunk.snippet}")

        if tool_results:
            lines.append("Tool results:")
            for result in tool_results:
                lines.append(f"- {result['tool']}: {json.dumps(result.get('result', result.get('error')), ensure_ascii=True)}")

        if not retrieved and not tool_results:
            lines.append("No grounded context was found.")

        return "\n".join(lines)


def ensure_default_knowledge_base(base_dir: str) -> str:
    knowledge_dir = os.path.join(base_dir, "knowledge")
    os.makedirs(knowledge_dir, exist_ok=True)

    primer_path = os.path.join(knowledge_dir, "bot_runtime.md")
    if not os.path.exists(primer_path):
        with open(primer_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# Bot Runtime Knowledge\n\n"
                "- This directory is the bot's non-parametric memory.\n"
                "- Add SOPs, strategy notes, broker rules, and operator playbooks here.\n"
                "- The assistant retrieves from these files before answering.\n"
                "- Runtime truth should come from tools, not static files.\n"
            )
    return knowledge_dir
