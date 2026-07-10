from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from finagent.market import market_snapshot
from finagent.memory import PreferenceStore
from finagent.models import ModelGateway, ModelResponse
from finagent.retrieval import LocalRetriever, read_chunks, tokenize
from finagent.sources import Citation, EvidenceChunk, SearchResult
from finagent.websearch import search_public_web


MARKET_TERMS = re.compile(r"\b(csi\s*300|沪深300|sh000300|index|指数|market performance|market data)\b", re.IGNORECASE)
CITATION_RE = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True)
class AgentResponse:
    answer: str
    citations: list[Citation]
    preferences: list[str]
    model_trace: list[dict[str, str | bool | None]]
    evidence_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "answer": self.answer,
            "citations": [citation.to_dict() for citation in self.citations],
            "preferences": self.preferences,
            "model_trace": self.model_trace,
            "evidence_count": self.evidence_count,
        }


class FinancialAgent:
    def __init__(
        self,
        *,
        index_path: Path,
        memory_path: Path,
        market_path: Path,
        models: ModelGateway | None = None,
    ) -> None:
        self.index_path = index_path
        self.memory = PreferenceStore(memory_path)
        self.market_path = market_path
        self.models = models or ModelGateway()

    def ask(
        self,
        question: str,
        *,
        user_id: str = "default",
        company: str | None = None,
        include_web: bool = False,
        limit: int = 7,
    ) -> AgentResponse:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")
        preferences = self.memory.record(user_id, question)
        query = " ".join([question, *preferences])

        chunks = read_chunks(self.index_path)
        if company:
            company_lower = company.lower()
            chunks = [chunk for chunk in chunks if company_lower in chunk.title.lower() or company_lower in chunk.document_id.lower()]
        evidence = [item.evidence for item in LocalRetriever(chunks).search(query, limit=limit)]
        if include_web:
            evidence.extend(self._web_evidence(question))
        if MARKET_TERMS.search(question) and self.market_path.exists():
            try:
                snapshot = market_snapshot(self.market_path)
                evidence.append(EvidenceChunk(
                    chunk_id=f"market:{snapshot.symbol}:{snapshot.start_date}:{snapshot.end_date}",
                    document_id=f"market-{snapshot.symbol}",
                    title=f"{snapshot.symbol} daily close and volume ({snapshot.start_date} to {snapshot.end_date})",
                    text=(
                        f"{snapshot.symbol} moved from {snapshot.start_close:.2f} on {snapshot.start_date} "
                        f"to {snapshot.end_close:.2f} on {snapshot.end_date}, a {snapshot.change_percent:.2f}% change. "
                        f"Average daily volume in this period was {snapshot.average_volume:.0f}."
                    ),
                    source_url=snapshot.source_url,
                    published_at=snapshot.end_date,
                    source_type="market_data",
                    locator=f"{self.market_path.name}; rows {snapshot.start_date}..{snapshot.end_date}",
                ))
            except (OSError, ValueError):
                # Market data is additive; a malformed local CSV does not invalidate filing answers.
                pass

        sources = self._cite(evidence)
        plan = self.models.complete(
            "deepseek",
            "You are a financial research planner. Identify the factual dimensions needed; do not make claims or use outside facts.",
            f"Question: {question}\nPersisted user preferences: {', '.join(preferences) or 'none'}",
        )
        draft = self.models.complete(
            "doubao",
            (
                "You are the filing analyst in an evidence-first financial agent. Answer only from the supplied evidence. "
                "Every factual claim needs one of the supplied [S#] citations. If evidence is insufficient, say so plainly. "
                "Do not provide investment advice or invent figures."
            ),
            self._analysis_prompt(question, preferences, sources),
        )
        verification = self.models.complete(
            "deepseek",
            (
                "You are the final citation verifier. Rewrite the draft only where needed to remove unsupported claims. "
                "Use only the exact [S#] labels in the evidence. Retain an uncertainty statement when evidence is weak."
            ),
            self._verification_prompt(question, sources, draft.text),
        )

        fallback = self._offline_answer(question, preferences, sources)
        final = fallback
        if draft.used_remote_model and verification.used_remote_model and self._has_valid_citations(verification.text, sources):
            final = self._sanitize_citations(verification.text, sources)
        elif draft.used_remote_model and self._has_valid_citations(draft.text, sources):
            final = self._sanitize_citations(draft.text, sources)

        return AgentResponse(
            answer=final,
            citations=[source.citation for source in sources],
            preferences=preferences,
            model_trace=[self._trace("planning", plan), self._trace("analysis", draft), self._trace("verification", verification)],
            evidence_count=len(sources),
        )

    @staticmethod
    def _trace(stage: str, response: ModelResponse) -> dict[str, str | bool | None]:
        return {
            "stage": stage,
            "provider": response.provider,
            "model": response.model,
            "used_remote_model": response.used_remote_model,
            "status": "ok" if response.used_remote_model else response.error,
        }

    @staticmethod
    def _cite(evidence: list[EvidenceChunk]) -> list[SearchResult]:
        return [SearchResult(
            evidence=chunk,
            citation=Citation(
                label=f"[S{index}]",
                title=chunk.title,
                source_url=chunk.source_url,
                published_at=chunk.published_at,
                source_type=chunk.source_type,
                document_id=chunk.document_id,
                locator=f"{chunk.locator}; chunk {chunk.chunk_id}" if chunk.locator else f"chunk {chunk.chunk_id}",
            ),
        ) for index, chunk in enumerate(evidence, start=1)]

    @staticmethod
    def _web_evidence(question: str) -> list[EvidenceChunk]:
        evidence: list[EvidenceChunk] = []
        for index, result in enumerate(search_public_web(question), start=1):
            evidence.append(EvidenceChunk(
                chunk_id=f"web:{index}",
                document_id=f"web:{index}",
                title=result.title,
                text=result.snippet or result.title,
                source_url=result.url,
                published_at=None,
                source_type="web_search",
                locator="search result snippet",
            ))
        return evidence

    @staticmethod
    def _analysis_prompt(question: str, preferences: list[str], sources: list[SearchResult]) -> str:
        return (
            f"Question: {question}\n"
            f"User preferences: {', '.join(preferences) or 'none'}\n\n"
            "Evidence:\n" + "\n\n".join(
                f"{source.citation.label} {source.evidence.title}\n{source.evidence.text}" for source in sources
            )
        )

    @staticmethod
    def _verification_prompt(question: str, sources: list[SearchResult], draft: str) -> str:
        labels = ", ".join(source.citation.label for source in sources) or "none"
        evidence = "\n".join(f"{source.citation.label} {source.evidence.text}" for source in sources)
        return f"Question: {question}\nAllowed citation labels: {labels}\n\nDraft:\n{draft}\n\nEvidence:\n{evidence}"

    @staticmethod
    def _has_valid_citations(answer: str, sources: list[SearchResult]) -> bool:
        valid = {source.citation.label for source in sources}
        labels = {f"[S{number}]" for number in CITATION_RE.findall(answer)}
        return bool(labels & valid) and labels.issubset(valid)

    @staticmethod
    def _sanitize_citations(answer: str, sources: list[SearchResult]) -> str:
        valid = {source.citation.label for source in sources}
        return CITATION_RE.sub(lambda match: match.group(0) if match.group(0) in valid else "[citation unavailable]", answer).strip()

    @staticmethod
    def _offline_answer(question: str, preferences: list[str], sources: list[SearchResult]) -> str:
        if not sources:
            return (
                "## Evidence-backed answer\n\n"
                "No local evidence matched this question. Run the SEC download and index commands, or repeat with `--web` for explicitly labelled public-web snippets. "
                "The agent will not infer an answer without evidence."
            )
        terms = set(tokenize(question)) | {term for preference in preferences for term in tokenize(preference)}
        bullets = []
        for source in sources[:5]:
            quote = FinancialAgent._best_excerpt(source.evidence.text, terms)
            bullets.append(f"- {quote} {source.citation.label}")
        preference_line = f"\n\nApplied remembered preferences: {', '.join(preferences)}." if preferences else ""
        return (
            "## Evidence-backed answer\n\n"
            "Offline extractive mode is active because one or both required model credentials are unavailable. "
            "The following are retrieved source excerpts, not a synthesized investment recommendation.\n\n"
            + "\n".join(bullets)
            + preference_line
        )

    @staticmethod
    def _best_excerpt(text: str, terms: set[str]) -> str:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        ranked = sorted(
            sentences,
            key=lambda sentence: (-sum(term in tokenize(sentence) for term in terms), len(sentence)),
        )
        excerpt = (ranked[0] if ranked else text).strip()
        return excerpt[:520] + ("..." if len(excerpt) > 520 else "")


def render_markdown(response: AgentResponse, *, include_trace: bool = False) -> str:
    sources = "\n".join(
        f"- {citation.label} [{citation.title}]({citation.source_url})"
        f" | {citation.source_type} | {citation.published_at or 'date unavailable'} | {citation.locator or 'locator unavailable'}"
        for citation in response.citations
    ) or "- No sources retrieved."
    output = f"{response.answer}\n\n## Sources\n\n{sources}"
    if include_trace:
        trace = "\n".join(
            f"- {item['stage']}: {item['provider']} / {item['model']} / remote={item['used_remote_model']} / {item['status']}"
            for item in response.model_trace
        )
        output += f"\n\n## Agent trace\n\n{trace}"
    return output
