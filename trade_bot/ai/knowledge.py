from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
import math
import os
import re
from typing import Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")
WORD_DIMENSIONS = 256


@dataclass
class KnowledgeDocument:
    doc_id: str
    source: str
    title: str
    content: str
    metadata: Dict[str, str]


@dataclass
class RetrievedChunk:
    document: KnowledgeDocument
    score: float
    snippet: str


class LocalKnowledgeBase:
    """
    Lightweight local RAG store.

    Non-parametric memory is kept on disk in a knowledge directory and scored
    with a simple lexical ranking pass. This keeps the first upgrade dependency
    free and deterministic.
    """

    def __init__(self, knowledge_dir: str):
        self.knowledge_dir = knowledge_dir
        self.documents: List[KnowledgeDocument] = []
        self._doc_tokens: List[Dict[str, int]] = []
        self._idf: Dict[str, float] = {}
        self._doc_dense_vectors: List[Dict[int, float]] = []
        self.reload()

    def reload(self) -> None:
        self.documents = list(self._load_documents())
        self._doc_tokens = [self._token_counts(doc.content) for doc in self.documents]
        self._idf = self._build_idf(self._doc_tokens)
        self._doc_dense_vectors = [
            self._dense_vector(self._tokenize_with_title(doc)) for doc in self.documents
        ]

    def _load_documents(self) -> Iterable[KnowledgeDocument]:
        if not os.path.isdir(self.knowledge_dir):
            return []

        docs: List[KnowledgeDocument] = []
        for root, _dirs, files in os.walk(self.knowledge_dir):
            for filename in sorted(files):
                if not filename.lower().endswith((".md", ".txt", ".json")):
                    continue
                path = os.path.join(root, filename)
                rel_path = os.path.relpath(path, self.knowledge_dir)
                try:
                    with open(path, "r", encoding="utf-8") as handle:
                        raw = handle.read()
                except Exception:
                    continue

                content, metadata = self._normalize_file(rel_path, raw)
                title = metadata.get("title") or os.path.splitext(os.path.basename(path))[0]
                docs.append(
                    KnowledgeDocument(
                        doc_id=rel_path,
                        source=path,
                        title=title,
                        content=content,
                        metadata=metadata,
                    )
                )
        return docs

    def _normalize_file(self, rel_path: str, raw: str) -> Tuple[str, Dict[str, str]]:
        if rel_path.lower().endswith(".json"):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                return raw, {"format": "json"}

            if isinstance(payload, dict):
                content = json.dumps(payload, ensure_ascii=True, indent=2)
                metadata = {
                    "format": "json",
                    "title": str(payload.get("title", "")),
                }
                return content, metadata

            return raw, {"format": "json"}

        return raw, {"format": "text"}

    def _token_counts(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for token in TOKEN_RE.findall(text.lower()):
            counts[token] = counts.get(token, 0) + 1
        return counts

    def _tokenize_with_title(self, document: KnowledgeDocument) -> Counter[str]:
        counts = Counter(self._token_counts(document.content))
        for token in TOKEN_RE.findall(document.title.lower()):
            counts[token] += 3
        return counts

    def _build_idf(self, doc_tokens: List[Dict[str, int]]) -> Dict[str, float]:
        if not doc_tokens:
            return {}

        total_docs = len(doc_tokens)
        doc_freq: Dict[str, int] = {}
        for counts in doc_tokens:
            for token in counts:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        return {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        if not query.strip() or not self.documents:
            return []

        query_tokens = self._token_counts(query)
        if not query_tokens:
            return []

        query_counter = Counter(query_tokens)
        query_vector = self._dense_vector(query_counter)

        candidates: List[Tuple[KnowledgeDocument, float, str]] = []
        for index, document in enumerate(self.documents):
            counts = self._doc_tokens[index]
            sparse_score = self._sparse_score(query_tokens, counts, document)
            dense_score = self._cosine_similarity(query_vector, self._doc_dense_vectors[index])
            hybrid_score = (0.55 * sparse_score) + (0.45 * dense_score)

            if hybrid_score <= 0:
                continue

            snippet = self._build_snippet(document.content, query_tokens)
            reranked_score = self._rerank_score(query, query_tokens, document, snippet, hybrid_score)
            candidates.append((document, reranked_score, snippet))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievedChunk(document=document, score=score, snippet=snippet)
            for document, score, snippet in candidates[:top_k]
        ]

    def _sparse_score(
        self,
        query_tokens: Dict[str, int],
        doc_tokens: Dict[str, int],
        document: KnowledgeDocument,
    ) -> float:
        score = 0.0
        title_tokens = set(TOKEN_RE.findall(document.title.lower()))
        for token, q_tf in query_tokens.items():
            if token not in doc_tokens:
                continue
            token_score = q_tf * doc_tokens[token] * self._idf.get(token, 1.0)
            if token in title_tokens:
                token_score *= 1.25
            score += token_score
        return score

    def _dense_vector(self, counts: Counter[str] | Dict[str, int]) -> Dict[int, float]:
        vector: Dict[int, float] = {}
        for token, value in counts.items():
            index = hash(token) % WORD_DIMENSIONS
            vector[index] = vector.get(index, 0.0) + float(value)

        norm = math.sqrt(sum(component * component for component in vector.values()))
        if norm == 0:
            return {}
        return {index: component / norm for index, component in vector.items()}

    def _cosine_similarity(self, left: Dict[int, float], right: Dict[int, float]) -> float:
        if not left or not right:
            return 0.0

        if len(left) > len(right):
            left, right = right, left

        return sum(value * right.get(index, 0.0) for index, value in left.items())

    def _rerank_score(
        self,
        query: str,
        query_tokens: Dict[str, int],
        document: KnowledgeDocument,
        snippet: str,
        base_score: float,
    ) -> float:
        title_lower = document.title.lower()
        content_lower = document.content.lower()
        query_lower = query.lower()
        matched_terms = sum(1 for token in query_tokens if token in content_lower)
        coverage = matched_terms / max(len(query_tokens), 1)
        phrase_bonus = 0.15 if query_lower in content_lower else 0.0
        title_bonus = 0.10 if any(token in title_lower for token in query_tokens) else 0.0
        snippet_bonus = 0.10 if snippet and len(snippet) < 240 else 0.0
        return base_score * (1.0 + (0.35 * coverage) + phrase_bonus + title_bonus + snippet_bonus)

    def _build_snippet(self, content: str, query_tokens: Dict[str, int], width: int = 220) -> str:
        lowered = content.lower()
        positions = [lowered.find(token) for token in query_tokens if lowered.find(token) >= 0]
        if not positions:
            return content[:width].replace("\n", " ").strip()

        start = max(min(positions) - 60, 0)
        end = min(start + width, len(content))
        return content[start:end].replace("\n", " ").strip()
