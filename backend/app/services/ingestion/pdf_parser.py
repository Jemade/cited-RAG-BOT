import os
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import hashlib
from app.core.config import settings
from app.core.logging import logger

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

class ExtractedPage:
    def __init__(self, page_number: int, text: str, blocks: List[Dict[str, Any]], preview_path: str = ""):
        self.page_number = page_number  # 1-indexed
        self.text = text
        self.blocks = blocks
        self.preview_path = preview_path

class PDFParser:
    def __init__(self, preview_dir: Path = settings.PAGE_PREVIEW_DIR):
        self.preview_dir = preview_dir
        os.makedirs(self.preview_dir, exist_ok=True)

    def compute_sha256(self, file_path: str) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def detect_headers_footers(self, doc) -> Tuple[set, set]:
        """Detect repeated header and footer lines across pages."""
        first_lines = []
        last_lines = []
        total_pages = len(doc)

        if total_pages < 3:
            return set(), set()

        for page in doc:
            lines = [line.strip() for line in page.get_text().split("\n") if line.strip()]
            if lines:
                first_lines.append(lines[0])
                last_lines.append(lines[-1])

        repeated_headers = set()
        repeated_footers = set()

        threshold = max(2, int(total_pages * 0.5))

        from collections import Counter
        header_counts = Counter(first_lines)
        footer_counts = Counter(last_lines)

        for line, count in header_counts.items():
            # If line is short, appears frequently, or is a pure number/running title
            if count >= threshold and len(line) < 100:
                repeated_headers.add(line)

        for line, count in footer_counts.items():
            if count >= threshold and (len(line) < 50 or re.match(r"^\d+$", line) or "page" in line.lower()):
                repeated_footers.add(line)

        return repeated_headers, repeated_footers

    def parse_pdf(self, file_path: str, document_id: str) -> List[ExtractedPage]:
        """Extract text, blocks, and page preview images from a PDF file."""
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is not installed. Please install PyMuPDF.")

        doc = fitz.open(file_path)
        total_pages = len(doc)
        logger.info(f"Parsing PDF '{file_path}' (Doc ID: {document_id}) with {total_pages} pages.")

        repeated_headers, repeated_footers = self.detect_headers_footers(doc)
        extracted_pages: List[ExtractedPage] = []

        doc_preview_dir = self.preview_dir / document_id
        os.makedirs(doc_preview_dir, exist_ok=True)

        for page_idx in range(total_pages):
            page_number = page_idx + 1  # 1-indexed
            page = doc[page_idx]

            # 1. Render page preview image (PNG) for the source viewer
            preview_filename = f"page_{page_number}.png"
            preview_path = str(doc_preview_dir / preview_filename)
            try:
                # Render at 150 DPI for crisp readability
                pix = page.get_pixmap(dpi=150)
                pix.save(preview_path)
            except Exception as e:
                logger.warning(f"Could not render page image preview for page {page_number}: {e}")
                preview_path = ""

            # 2. Extract structured blocks and lines
            # Page.get_text("blocks") returns list of tuples: (x0, y0, x1, y1, "text", block_no, block_type)
            raw_blocks = page.get_text("blocks")
            clean_blocks = []
            page_text_lines = []

            for b in raw_blocks:
                if len(b) >= 5 and b[4]:  # block text
                    block_text = b[4].strip()
                    if not block_text:
                        continue

                    # Filter repeated header/footer
                    lines = [line.strip() for line in block_text.split("\n") if line.strip()]
                    filtered_lines = [
                        l for l in lines 
                        if l not in repeated_headers and l not in repeated_footers
                    ]
                    if not filtered_lines:
                        continue

                    cleaned_block_text = "\n".join(filtered_lines)
                    clean_blocks.append({
                        "bbox": (b[0], b[1], b[2], b[3]),
                        "text": cleaned_block_text,
                        "block_no": b[5] if len(b) > 5 else 0
                    })
                    page_text_lines.append(cleaned_block_text)

            page_full_text = "\n\n".join(page_text_lines)
            extracted_pages.append(ExtractedPage(
                page_number=page_number,
                text=page_full_text,
                blocks=clean_blocks,
                preview_path=preview_path
            ))

        doc.close()
        return extracted_pages
