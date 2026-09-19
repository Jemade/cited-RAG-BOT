import os
import re
from typing import List, Dict, Any, Optional
import httpx
from app.core.config import settings
from app.core.logging import logger
from app.services.generation.context_builder import SYSTEM_PROMPT

class LLMClient:
    def __init__(
        self,
        provider: str = settings.LLM_PROVIDER,
        model: str = settings.LLM_MODEL
    ):
        self.provider = provider
        self.model = model

        # Auto-detect provider if 'auto'
        if self.provider == "auto":
            if settings.OPENAI_API_KEY:
                self.provider = "openai"
            elif settings.GEMINI_API_KEY:
                self.provider = "gemini"
            elif settings.GROQ_API_KEY:
                self.provider = "groq"
            else:
                self.provider = "mock"

        logger.info(f"Initialized LLMClient with provider='{self.provider}', model='{self.model}'")

    async def generate_answer(
        self,
        query: str,
        context_str: str,
        source_meta: List[Dict[str, Any]]
    ) -> str:
        """Generate answer from prompt context using configured provider or mock fallback."""
        if not source_meta or not context_str.strip():
            return "I don't know based on the documents available. The provided document sources do not contain sufficient evidence to answer this question."

        if self.provider == "openai" and settings.OPENAI_API_KEY:
            return await self._call_openai(query, context_str)
        elif self.provider == "gemini" and settings.GEMINI_API_KEY:
            return await self._call_gemini(query, context_str)
        elif self.provider == "groq" and settings.GROQ_API_KEY:
            return await self._call_groq(query, context_str)
        else:
            return self._generate_mock(query, source_meta)

    async def _call_openai(self, query: str, context_str: str) -> str:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    async def _call_gemini(self, query: str, context_str: str) -> str:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={settings.GEMINI_API_KEY}"
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": f"Context:\n{context_str}\n\nQuestion: {query}"}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 512}
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                if parts:
                    return parts[0]["text"].strip()
            return "I don't know based on the documents available."

    async def _call_groq(self, query: str, context_str: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 512
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()

    def _generate_mock(self, query: str, source_meta: List[Dict[str, Any]]) -> str:
        """Deterministic mock generator for tests and offline evaluation."""
        query_lower = query.lower()
        q_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", query_lower))

        # Check for unanswerable queries or zero overlap
        best_overlap = 0.0
        best_source = None
        best_sentence = None

        for src in source_meta:
            text = src["text"]
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in sentences:
                s_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", s.lower()))
                if not q_tokens:
                    continue
                overlap = len(q_tokens.intersection(s_tokens)) / len(q_tokens)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_source = src
                    best_sentence = s.strip()

        # If keyword overlap is too low or query is out of domain, refuse
        if best_overlap < 0.35 or best_source is None or best_sentence is None:
            return "I don't know based on the documents available. The provided document sources do not contain sufficient evidence to answer this question."

        page_num = best_source["page_number"]
        return f"{best_sentence} [Page {page_num}]"
