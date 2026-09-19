"""
retriever.py — Hybrid Semantic + Keyword Search module for Mizan AI.

Combines:
  1. Dense Semantic Search: ChromaDB with ONNX multilingual-e5-small.
  2. Sparse Keyword Search: Arabic BM25Okapi with character normalization.
  3. Fusion Engine: Reciprocal Rank Fusion (RRF) for optimal ranking.

Features:
  - Metadata filtering (doc_id, doc_type, year, specialization)
  - Libyan dialect expansion
  - RRF with configurable weights
"""

import os
import re
import warnings
from collections import defaultdict

import numpy as np
import chromadb
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import onnxruntime as ort

from config import CHROMA_DIR, CHROMA_COLLECTION
from mizan.bm25_search import ArabicBM25Search

DB_DIR = CHROMA_DIR
COLLECTION_NAME = CHROMA_COLLECTION
ONNX_REPO = "Xenova/multilingual-e5-small"
MAX_LENGTH = 512

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ─── Libyan Dialect Vocabulary Mapping ─────────────────────────────────────────
LIBYAN_DIALECT_MAP = [
    # Interrogatives
    (r"\bشن\b", "ما هي"),
    (r"\bشنو\b", "ما هو"),
    (r"\bشنهي\b", "ما هي"),
    (r"\bقداش\b", "كم مدة مقدار"),
    (r"\bوين\b", "أين"),
    (r"\bعلاش\b", "لماذا"),
    (r"\bكي فاش\b", "كيف"),

    # Desires / requests
    (r"\bنبي\s+نعرف\b", "ما هي أحكام"),
    (r"\bنبي\b", "أرغب في"),
    (r"\bباش\b", "لكي من أجل"),
    (r"\bناخذ\b", "الحصول على"),
    (r"\bنقدر\s+ناخذ\b", "شروط واستحقاق"),

    # Labor
    (r"\bفصلوني\b", "إنهاء الخدمة الفصل التعسفي"),
    (r"\bطردوني\b", "فصل تعسفي إنهاء العقد"),
    (r"\bخدمة\b", "عمل وظيفة"),
    (r"\bخدمتي\b", "عملي وخدمتي الوظيفية"),
    (r"\bعطوني\b", "منحي واستحقاق"),
    (r"\bيقدروا\b", "صلاحيات وحق"),
    (r"\bيوقفوني\b", "إيقاف عن العمل"),

    # Pension / social security
    (r"\bتقاعدت\b", "التقاعد انتهاء الخدمة"),
    (r"\bمعاشي\b", "المعاش التقاعدي"),
    (r"\bضمان\b", "الضمان الاجتماعي"),
    (r"\bتأمين\b", "التأمين"),
]


def expand_libyan_query(user_query: str) -> str:
    """Expands Libyan dialect questions with formal Arabic legal equivalents."""
    added_terms = []
    for pattern, replacement in LIBYAN_DIALECT_MAP:
        if re.search(pattern, user_query):
            added_terms.append(replacement)

    if added_terms:
        terms_suffix = " ".join(sorted(set(added_terms)))
        return f"{user_query} {terms_suffix}"
    return user_query


class LibyanLawRetriever:
    def __init__(
        self,
        db_path: str = DB_DIR,
        collection_name: str = COLLECTION_NAME,
        bm25_enabled: bool = True,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # 1. Initialize ChromaDB (Dense)
        print(f"Connecting to ChromaDB at: {self.db_path}...")
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_collection(self.collection_name)

        # 2. Initialize ONNX Embedder
        print(f"Loading ONNX embedding model: {ONNX_REPO}...")
        model_path = hf_hub_download(repo_id=ONNX_REPO, filename="onnx/model_quantized.onnx")
        tok_path = hf_hub_download(repo_id=ONNX_REPO, filename="tokenizer.json")

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_LENGTH)

        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_names = [i.name for i in self.session.get_inputs()]

        # 3. Initialize BM25 (Sparse)
        self.bm25_searcher = ArabicBM25Search() if bm25_enabled else None

        print(f"Retriever ready! ({self.collection.count()} chunks in index, BM25={bm25_enabled})")

    # ─── Query encoding ────────────────────────────────────────────────────────
    def _encode_query(self, query: str) -> list[float]:
        prefixed = f"query: {query}"
        encoded = self.tokenizer.encode(prefixed)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if "token_type_ids" in self.input_names:
            onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

        outputs = self.session.run(None, onnx_inputs)
        last_hidden = outputs[0]

        mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
        sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embedding = sum_embeddings / sum_mask

        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        norm_emb = (embedding / np.clip(norm, a_min=1e-9, a_max=None)).astype(np.float32)
        return norm_emb[0].tolist()

    # ─── Metadata filter builder ───────────────────────────────────────────────
    def _build_where_clause(self, filter_article: str = None, filters: dict = None):
        """
        Build a ChromaDB `where` clause from:
          - filter_article: exact article number (string)
          - filters: dict of metadata filters, e.g.:
              {"doc_id": "LAW_12_2010"}
              {"doc_type": "law"}
              {"year": 2010}
              {"doc_id": ["LAW_12_2010", "CIVIL_CODE"]}   # multiple values → $in
        """
        conditions = []

        if filter_article:
            conditions.append({"article": str(filter_article)})

        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple, set)):
                    conditions.append({key: {"$in": list(value)}})
                else:
                    conditions.append({key: value})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    # ─── Dense search ──────────────────────────────────────────────────────────
    def search_dense(
        self,
        query: str,
        top_k: int = 10,
        filter_article: str = None,
        filters: dict = None,
        expand_dialect: bool = True,
    ) -> list[dict]:
        """Pure dense vector search via ChromaDB."""
        search_query = expand_libyan_query(query) if expand_dialect else query
        query_embedding = self._encode_query(search_query)

        where_clause = self._build_where_clause(filter_article, filters)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        output = []
        if not results["ids"] or not results["ids"][0]:
            return output

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, docs, metas, distances):
            similarity = round(1.0 - dist, 4)
            output.append({
                "chunk_id": chunk_id,
                "article": meta.get("article"),
                "page": meta.get("page"),
                "section": meta.get("section"),
                "chapter": meta.get("chapter"),
                "document": meta.get("document"),
                "doc_id": meta.get("doc_id"),
                "doc_type": meta.get("doc_type"),
                "year": meta.get("year"),
                "score": similarity,
                "dense_score": similarity,
                "text": doc,
            })

        return output

    # ─── Sparse search ─────────────────────────────────────────────────────────
    def search_sparse(
        self,
        query: str,
        top_k: int = 10,
        filter_article: str = None,
        filters: dict = None,
    ) -> list[dict]:
        """Pure sparse keyword search via BM25, with metadata filtering."""
        if not self.bm25_searcher:
            return []
        res = self.bm25_searcher.search(query, top_k=top_k)

        # Apply metadata filters manually (BM25 has no native filtering)
        if filter_article:
            target = str(filter_article)
            res = [r for r in res if str(r.get("article")) == target]

        if filters:
            for key, value in filters.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple, set)):
                    allowed = set(str(v) for v in value)
                    res = [r for r in res if str(r.get(key)) in allowed]
                else:
                    res = [r for r in res if str(r.get(key)) == str(value)]

        for r in res:
            r["score"] = r["bm25_score"]
        return res

    # ─── Hybrid search ─────────────────────────────────────────────────────────
    def search_hybrid(
        self,
        query: str,
        top_k: int = 3,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        rrf_k: int = 60,
        filter_article: str = None,
        filters: dict = None,
        expand_dialect: bool = True,
    ) -> list[dict]:
        """Hybrid: Dense + BM25 combined via Reciprocal Rank Fusion (RRF)."""
        pool_size = max(top_k * 10, 30)

        dense_results = self.search_dense(
            query=query,
            top_k=pool_size,
            filter_article=filter_article,
            filters=filters,
            expand_dialect=expand_dialect,
        )
        sparse_results = self.search_sparse(
            query=query,
            top_k=pool_size,
            filter_article=filter_article,
            filters=filters,
        )

        rrf_scores = defaultdict(float)
        chunk_data = {}

        for rank, item in enumerate(dense_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] += dense_weight / (rrf_k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = item.copy()
            chunk_data[cid]["dense_rank"] = rank
            chunk_data[cid]["dense_score"] = item.get("dense_score", 0.0)

        for rank, item in enumerate(sparse_results, start=1):
            cid = item["chunk_id"]
            rrf_scores[cid] += bm25_weight / (rrf_k + rank)
            if cid not in chunk_data:
                chunk_data[cid] = item.copy()
            chunk_data[cid]["bm25_rank"] = rank
            chunk_data[cid]["bm25_score"] = item.get("bm25_score", 0.0)

        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        final_results = []
        for cid in sorted_ids:
            item = chunk_data[cid]
            item["rrf_score"] = round(rrf_scores[cid], 5)
            item["score"] = item["rrf_score"]
            item.setdefault("dense_rank", None)
            item.setdefault("bm25_rank", None)
            item.setdefault("dense_score", 0.0)
            item.setdefault("bm25_score", 0.0)
            final_results.append(item)

        return final_results

    # ─── Unified endpoint ──────────────────────────────────────────────────────
    def search(self, query: str, top_k: int = 3, mode: str = "hybrid", **kwargs) -> list[dict]:
        """
        Unified search endpoint.
        mode: 'hybrid' | 'dense' | 'bm25'
        Accepts: filter_article, filters, expand_dialect, dense_weight, bm25_weight
        """
        if mode == "hybrid":
            return self.search_hybrid(query=query, top_k=top_k, **kwargs)
        elif mode == "dense":
            return self.search_dense(query=query, top_k=top_k, **kwargs)
        elif mode == "bm25":
            return self.search_sparse(query=query, top_k=top_k, **kwargs)
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}. Choose 'hybrid', 'dense', or 'bm25'.")
