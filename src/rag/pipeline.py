"""
RAG Pipeline — Embedding, FAISS indexing, and Groq-powered generation.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Any

from groq import Groq
from sentence_transformers import SentenceTransformer
import faiss


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    EMBED_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 400          # words per chunk
    CHUNK_OVERLAP = 50        # overlap between chunks
    TOP_K = 5                 # chunks to retrieve per query
    SYSTEM_PROMPT = (
        "You are an expert AI project analyst. Answer questions using ONLY "
        "the provided document context. If the context doesn't contain the "
        "answer, say so clearly. Always be concise and structured."
    )

    def __init__(self, groq_key: str, model: str = "llama3-8b-8192"):
        self.client = Groq(api_key=groq_key)
        self.model = model
        self.embedder = SentenceTransformer(self.EMBED_MODEL)
        self.index: faiss.IndexFlatL2 | None = None
        self.chunks: List[Dict[str, Any]] = []

    # ── Indexing ────────────────────────────────────────────────────────────

    def build_index(self, documents: List[Dict[str, str]]) -> None:
        """Chunk documents, embed, and build FAISS index."""
        self.chunks = []
        for doc in documents:
            chunks = self._chunk_text(doc["content"], doc["filename"])
            self.chunks.extend(chunks)

        if not self.chunks:
            raise ValueError("No content found in documents.")

        texts = [c["text"] for c in self.chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def _chunk_text(self, text: str, filename: str) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        step = self.CHUNK_SIZE - self.CHUNK_OVERLAP
        for i in range(0, len(words), step):
            chunk_words = words[i: i + self.CHUNK_SIZE]
            chunks.append({
                "text": " ".join(chunk_words),
                "file": filename,
                "chunk_id": len(chunks)
            })
        return chunks

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Return top-K relevant chunks for a query."""
        q_vec = self.embedder.encode([query], show_progress_bar=False)
        q_vec = np.array(q_vec, dtype="float32")
        _, indices = self.index.search(q_vec, self.TOP_K)
        return [self.chunks[i] for i in indices[0] if i < len(self.chunks)]

    # ── Generation ───────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        chat_history: List[Dict[str, str]] | None = None
    ) -> Dict[str, Any]:
        """Retrieve context and generate a grounded answer."""
        retrieved = self.retrieve(question)
        context = "\n\n".join(
            [f"[{c['file']}]\n{c['text']}" for c in retrieved]
        )

        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}]

        # Include last 4 exchanges for conversational memory
        if chat_history:
            for msg in chat_history[-8:]:
                if msg["role"] in ("user", "assistant"):
                    messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({
            "role": "user",
            "content": (
                f"Context from documents:\n{context}\n\n"
                f"Question: {question}"
            )
        })

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=1024,
            temperature=0.2
        )

        answer = response.choices[0].message.content

        sources = [
            {
                "file": c["file"],
                "snippet": c["text"][:120]
            }
            for c in retrieved[:3]
        ]

        return {"answer": answer, "sources": sources, "context": context}

    def summarize_all(self, instruction: str = "") -> str:
        """Summarize the full document corpus for report generation."""
        all_text = " ".join([c["text"] for c in self.chunks[:40]])  # first ~16k words
        prompt = (
            f"{instruction}\n\nDocument content:\n{all_text[:6000]}"
            if instruction else
            f"Summarize the key themes, decisions, risks, and progress from these project documents:\n\n{all_text[:6000]}"
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior project analyst writing concise, structured summaries."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1200,
            temperature=0.3
        )
        return response.choices[0].message.content
