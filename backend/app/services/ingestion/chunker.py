import re
from typing import List, Dict, Any, Optional
from app.services.ingestion.pdf_parser import ExtractedPage
from app.core.config import settings
from app.core.logging import logger

class ProcessedChunk:
    def __init__(
        self,
        chunk_id: str,
        document_id: str,
        page_number: int,
        chunk_index: int,
        text: str,
        section: Optional[str],
        token_count: int,
        metadata_json: Dict[str, Any]
    ):
        self.chunk_id = chunk_id
        self.document_id = document_id
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.text = text
        self.section = section
        self.token_count = token_count
        self.metadata_json = metadata_json

class ParagraphAwareChunker:
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_chunk_len: int = 40
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_len = min_chunk_len

    def is_likely_heading(self, text: str) -> bool:
        """Detect if a line looks like a section heading."""
        text = text.strip()
        if len(text) > 90 or len(text) < 3:
            return False
        # Numbered headings like '1. Introduction', '1 Introduction', '2.1 Model Architecture', 'IV. EXPERIMENTS'
        if re.match(r"^(\d+(\.\d+)*\.?|[A-ZIVXLC]+\.?)\s+[A-Z]", text):
            return True
        # All-caps headings
        if text.isupper() and len(text.split()) <= 8:
            return True
        return False

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences preserving sentence terminators."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk_document(
        self,
        extracted_pages: List[ExtractedPage],
        document_id: str,
        source_filename: str
    ) -> List[ProcessedChunk]:
        """Split extracted pages into paragraph-bounded chunks."""
        chunks: List[ProcessedChunk] = []
        global_chunk_idx = 0
        current_section: Optional[str] = None

        for page in extracted_pages:
            page_num = page.page_number
            page_text = page.text.strip()

            # Handle empty or trivial pages
            if not page_text or len(page_text) < self.min_chunk_len:
                logger.debug(f"Skipping empty or near-empty page {page_num} in doc {document_id}")
                continue

            # Process blocks
            paragraphs: List[str] = []
            for block in page.blocks:
                b_text = block["text"].strip()
                if not b_text:
                    continue

                lines = b_text.split("\n")
                if len(lines) == 1 and self.is_likely_heading(lines[0]):
                    current_section = lines[0]
                    paragraphs.append(b_text)
                else:
                    # Check first line for heading
                    if self.is_likely_heading(lines[0]):
                        current_section = lines[0]
                    paragraphs.append(b_text)

            if not paragraphs:
                paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

            # Build chunks from paragraphs within this page
            current_chunk_paragraphs: List[str] = []
            current_len = 0

            for para in paragraphs:
                para_len = len(para)

                # If single paragraph exceeds chunk size, split by sentences
                if para_len > self.chunk_size:
                    sentences = self.split_into_sentences(para)
                    for sent in sentences:
                        if current_len + len(sent) > self.chunk_size and current_chunk_paragraphs:
                            chunk_str = " ".join(current_chunk_paragraphs).strip()
                            if len(chunk_str) >= self.min_chunk_len:
                                chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                                est_tokens = max(1, int(len(chunk_str.split()) * 1.3))
                                chunks.append(ProcessedChunk(
                                    chunk_id=chunk_id,
                                    document_id=document_id,
                                    page_number=page_num,
                                    chunk_index=global_chunk_idx,
                                    text=chunk_str,
                                    section=current_section,
                                    token_count=est_tokens,
                                    metadata_json={
                                        "source_filename": source_filename,
                                        "page_number": page_num,
                                        "section": current_section,
                                        "char_length": len(chunk_str)
                                    }
                                ))
                                global_chunk_idx += 1

                            # Overlap from tail of previous sentences
                            overlap_chars = chunk_str[-self.chunk_overlap:] if len(chunk_str) > self.chunk_overlap else ""
                            current_chunk_paragraphs = [overlap_chars] if overlap_chars else []
                            current_len = len(overlap_chars)

                        current_chunk_paragraphs.append(sent)
                        current_len += len(sent) + 1

                elif current_len + para_len > self.chunk_size and current_chunk_paragraphs:
                    # Emit current chunk
                    chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                    if len(chunk_str) >= self.min_chunk_len:
                        chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                        est_tokens = max(1, int(len(chunk_str.split()) * 1.3))
                        chunks.append(ProcessedChunk(
                            chunk_id=chunk_id,
                            document_id=document_id,
                            page_number=page_num,
                            chunk_index=global_chunk_idx,
                            text=chunk_str,
                            section=current_section,
                            token_count=est_tokens,
                            metadata_json={
                                "source_filename": source_filename,
                                "page_number": page_num,
                                "section": current_section,
                                "char_length": len(chunk_str)
                            }
                        ))
                        global_chunk_idx += 1

                    # Create overlap from end of chunk
                    overlap_text = chunk_str[-self.chunk_overlap:] if len(chunk_str) > self.chunk_overlap else ""
                    current_chunk_paragraphs = [overlap_text, para] if overlap_text else [para]
                    current_len = len(overlap_text) + para_len
                else:
                    current_chunk_paragraphs.append(para)
                    current_len += para_len + 2

            # Flush remaining paragraphs on this page
            if current_chunk_paragraphs:
                chunk_str = "\n\n".join(current_chunk_paragraphs).strip()
                if len(chunk_str) >= self.min_chunk_len:
                    chunk_id = f"{document_id}_p{page_num}_c{global_chunk_idx}"
                    est_tokens = max(1, int(len(chunk_str.split()) * 1.3))
                    chunks.append(ProcessedChunk(
                        chunk_id=chunk_id,
                        document_id=document_id,
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                        text=chunk_str,
                        section=current_section,
                        token_count=est_tokens,
                        metadata_json={
                            "source_filename": source_filename,
                            "page_number": page_num,
                            "section": current_section,
                            "char_length": len(chunk_str)
                        }
                    ))
                    global_chunk_idx += 1

        logger.info(f"Chunked document {document_id} into {len(chunks)} chunks across {len(extracted_pages)} pages.")
        return chunks
