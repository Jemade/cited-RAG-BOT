from typing import List, Tuple
from app.services.retrieval.dense import RetrievalResult

SYSTEM_PROMPT = """You are CiteRAG, a precision document question-answering assistant.
Your goal is to provide accurate, factual answers grounded exclusively in the provided context sources.

RULES:
1. Answer ONLY using the facts directly stated in the Context sources below. Do not extrapolate or use outside knowledge.
2. If the context does not contain sufficient facts to answer the question with certainty, respond exactly with:
   "I don't know based on the documents available. The provided document sources do not contain sufficient evidence to answer this question."
3. Every factual assertion or sentence MUST be accompanied by an inline citation referencing the exact supporting page number in the format: [Page X] (for example: "The Transformer uses 8 parallel attention heads [Page 4].").
4. Never invent citations, and do not cite page numbers that do not appear in the Context."""

class ContextBuilder:
    @staticmethod
    def build_context(retrieved_chunks: List[RetrievalResult]) -> Tuple[str, List[dict]]:
        """Format retrieved chunks into a context string."""
        context_blocks = []
        source_meta = []

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            sec_info = f" | Section: {chunk.section}" if chunk.section else ""
            block = (
                f"[Source {idx} | Page: {chunk.page_number} | Chunk: {chunk.chunk_id}{sec_info}]\n"
                f"{chunk.text.strip()}\n"
            )
            context_blocks.append(block)
            source_meta.append({
                "source_idx": idx,
                "page_number": chunk.page_number,
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "section": chunk.section,
                "text": chunk.text
            })

        full_context = "\n---\n".join(context_blocks)
        return full_context, source_meta

    @staticmethod
    def build_prompt(query: str, context_str: str) -> str:
        """Construct the complete user prompt with context."""
        return (
            f"Context Sources:\n"
            f"---------------------\n"
            f"{context_str}\n"
            f"---------------------\n\n"
            f"Question: {query}\n\n"
            f"Instructions: Answer the question using solely the context above. Include inline [Page X] citations for every claim. If evidence is insufficient, state that you don't know based on the documents available."
        )
