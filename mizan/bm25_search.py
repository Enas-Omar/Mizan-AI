"""
bm25_search.py — Sparse Keyword Search engine using BM25 with Arabic normalization.

Provides ArabicBM25Search:
  - Normalizes Arabic letters (alef variants, taa marbuta, alif maqsura, tatweel).
  - Builds BM25Okapi inverted index over all legal chunks.
  - Returns ranked results with BM25 scores for keyword queries.
"""

import os
import re
import json
from rank_bm25 import BM25Okapi

from config import CHUNKS_DIR

# Arabic normalization maps
ALEF_PATTERN = re.compile(r"[إأآٱ]")
DIACRITICS_PATTERN = re.compile(r"[\u064B-\u0653\u0670]")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s\u0600-\u06FF]")


def tokenize_arabic(text: str) -> list[str]:
    """
    Normalizes and tokenizes Arabic legal text for BM25 keyword matching.
    """
    # 1. Strip diacritics
    text = DIACRITICS_PATTERN.sub("", text)
    # 2. Normalize alef variants to bare alef
    text = ALEF_PATTERN.sub("ا", text)
    # 3. Normalize taa marbuta to haa
    text = text.replace("ة", "ه")
    # 4. Normalize alif maqsura to yaa
    text = text.replace("ى", "ي")
    # 5. Remove tatweel
    text = text.replace("ـ", "")
    # 6. Replace punctuation with space
    text = PUNCTUATION_PATTERN.sub(" ", text)

    # 7. Tokenize and filter out single characters
    tokens = text.lower().split()
    return [t for t in tokens if len(t) > 1]


class ArabicBM25Search:
    def __init__(self, chunks_path: str = None):
        if chunks_path is None:
            chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(f"Chunks file not found at: {chunks_path}")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Tokenize corpus for BM25
        print(f"Building BM25 index over {len(self.chunks)} legal chunks...")
        self.corpus_tokens = [
            tokenize_arabic(c["text_with_context"])
            for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        print("✓ BM25 index built successfully.")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Search corpus using BM25 algorithm.
        Returns top-k results sorted by BM25 score.
        """
        query_tokens = tokenize_arabic(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score <= 0:
                continue
            c = self.chunks[idx]
            results.append({
                "chunk_id": c["chunk_id"],
                "article": c.get("article"),
                "page": c.get("page"),
                "section": c.get("section"),
                "chapter": c.get("chapter"),
                "document": c.get("document"),
                "doc_id": c.get("doc_id"),
                "doc_type": c.get("doc_type"),
                "year": c.get("year"),
                "bm25_score": round(score, 4),
                "text": c["text_with_context"],
            })

        return results
