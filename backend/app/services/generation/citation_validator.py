import re
from typing import List, Dict, Any, Tuple, Optional
from app.core.logging import logger

class ValidatedCitation:
    def __init__(
        self,
        document_id: str,
        page_number: int,
        chunk_id: str,
        citation_marker: str,
        snippet: str,
        validated: bool = True,
        match_score: float = 1.0,
        warning: Optional[str] = None
    ):
        self.document_id = document_id
        self.page_number = page_number
        self.chunk_id = chunk_id
        self.citation_marker = citation_marker
        self.snippet = snippet
        self.validated = validated
        self.match_score = round(match_score, 4)
        self.warning = warning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "page_number": self.page_number,
            "chunk_id": self.chunk_id,
            "citation_marker": self.citation_marker,
            "snippet": self.snippet,
            "validated": self.validated,
            "match_score": self.match_score,
            "warning": self.warning
        }

class CitationValidator:
    def __init__(self):
        # Patterns matching citations: [Page 4], [Source: page 14], [Source 1], [Page 4, Chunk abc]
        self.citation_regex = re.compile(
            r"\[(?:Source:\s*(?:page\s*)?|Page\s*|Source\s*)?(\d+)(?:[^\]]*)?\]",
            re.IGNORECASE
        )

    def extract_and_validate(
        self,
        answer_text: str,
        retrieved_sources: List[Dict[str, Any]]
    ) -> Tuple[str, List[ValidatedCitation]]:
        """Extract and validate citations against retrieved context."""
        if not answer_text or not retrieved_sources:
            return answer_text, []

        # Index retrieved sources by page_number and source_idx
        page_to_sources: Dict[int, List[Dict[str, Any]]] = {}
        idx_to_sources: Dict[int, Dict[str, Any]] = {}

        for src in retrieved_sources:
            page_num = src["page_number"]
            src_idx = src.get("source_idx")
            if page_num not in page_to_sources:
                page_to_sources[page_num] = []
            page_to_sources[page_num].append(src)
            if src_idx is not None:
                idx_to_sources[src_idx] = src

        valid_citations: List[ValidatedCitation] = []
        hallucinated_citations: List[str] = []

        # Split answer into sentences to correlate each citation with its claim
        sentences = re.split(r"(?<=[.!?])\s+", answer_text)
        processed_sentences = []

        for sent in sentences:
            # Find citation markers in this sentence
            matches = list(self.citation_regex.finditer(sent))
            current_sent = sent

            for match in matches:
                marker = match.group(0)
                num_str = match.group(1)
                num = int(num_str)

                # Check if number matches a page or a source index
                target_chunk = None
                resolved_page = None

                if num in page_to_sources:
                    resolved_page = num
                    # Find best chunk on this page with highest lexical overlap with sentence
                    target_chunk = self._find_best_matching_chunk(sent, page_to_sources[num])
                elif num in idx_to_sources:
                    target_chunk = idx_to_sources[num]
                    resolved_page = target_chunk["page_number"]

                if target_chunk and resolved_page is not None:
                    # Valid citation grounded in retrieved sources
                    standard_marker = f"[Page {resolved_page}]"
                    current_sent = current_sent.replace(marker, standard_marker, 1)

                    # Extract snippet around matching terms
                    snippet = self._extract_supporting_snippet(sent, target_chunk["text"])
                    match_score = self._compute_sentence_chunk_overlap(sent, target_chunk["text"])

                    # Avoid duplicate identical citations for same chunk in same sentence
                    existing = [c for c in valid_citations if c.chunk_id == target_chunk["chunk_id"] and c.page_number == resolved_page]
                    if not existing:
                        valid_citations.append(ValidatedCitation(
                            document_id=target_chunk["document_id"],
                            page_number=resolved_page,
                            chunk_id=target_chunk["chunk_id"],
                            citation_marker=standard_marker,
                            snippet=snippet,
                            validated=True,
                            match_score=match_score
                        ))
                else:
                    # Hallucinated citation: cited page or source does not exist in context!
                    logger.warning(f"Hallucinated citation detected: '{marker}' (Page/Source {num} not in retrieved context)")
                    hallucinated_citations.append(marker)
                    # Remove hallucinated citation marker from answer text
                    current_sent = current_sent.replace(marker, "").strip()

            processed_sentences.append(current_sent)

        cleaned_answer = " ".join(processed_sentences).strip()

        # If answer had no citations but retrieved chunks were provided and answer is not a refusal,
        # generate grounding citations from the top retrieved chunk if confidence is high
        is_refusal = "don't know" in cleaned_answer.lower() or "not contain sufficient" in cleaned_answer.lower()
        if not valid_citations and not is_refusal and retrieved_sources:
            top_src = retrieved_sources[0]
            top_overlap = self._compute_sentence_chunk_overlap(cleaned_answer, top_src["text"])
            if top_overlap > 0.15:
                standard_marker = f"[Page {top_src['page_number']}]"
                cleaned_answer += f" {standard_marker}"
                valid_citations.append(ValidatedCitation(
                    document_id=top_src["document_id"],
                    page_number=top_src["page_number"],
                    chunk_id=top_src["chunk_id"],
                    citation_marker=standard_marker,
                    snippet=self._extract_supporting_snippet(cleaned_answer, top_src["text"]),
                    validated=True,
                    match_score=top_overlap,
                    warning="Citation mapped from top retrieved source with high lexical overlap"
                ))

        return cleaned_answer, valid_citations

    def _compute_sentence_chunk_overlap(self, sentence: str, chunk_text: str) -> float:
        """Calculate word overlap between sentence and chunk."""
        s_words = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
        c_words = set(re.findall(r"\b\w{3,}\b", chunk_text.lower()))
        if not s_words:
            return 0.0
        return len(s_words.intersection(c_words)) / len(s_words)

    def _find_best_matching_chunk(self, sentence: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Find the chunk on a given page with highest lexical overlap with sentence."""
        best_chunk = chunks[0]
        best_score = -1.0
        for c in chunks:
            score = self._compute_sentence_chunk_overlap(sentence, c["text"])
            if score > best_score:
                best_score = score
                best_chunk = c
        return best_chunk

    def _extract_supporting_snippet(self, sentence: str, chunk_text: str, max_chars: int = 250) -> str:
        """Find the most relevant sentence or snippet within the chunk."""
        chunk_sentences = re.split(r"(?<=[.!?])\s+", chunk_text)
        best_s = chunk_sentences[0] if chunk_sentences else chunk_text[:max_chars]
        best_score = -1.0

        for cs in chunk_sentences:
            score = self._compute_sentence_chunk_overlap(sentence, cs)
            if score > best_score:
                best_score = score
                best_s = cs

        clean_snippet = best_s.strip()
        if len(clean_snippet) > max_chars:
            clean_snippet = clean_snippet[:max_chars] + "..."
        return clean_snippet
