"""
test_retrieval.py — Test and compare Hybrid Retrieval (Dense + BM25 + RRF) vs Dense vs BM25.

Evaluates:
  1. Dense Search (Semantic understanding)
  2. BM25 Search (Exact legal keyword matching)
  3. Hybrid Fusion (Combined RRF ranking)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mizan.retriever import LibyanLawRetriever

TEST_QUERIES = [
    {
        "query": "المادة 34",
        "description": "Exact Article Lookup (Where BM25 excels)",
    },
    {
        "query": "شن شروط التقديم على وظيفة والتعيين؟",
        "description": "Libyan Dialect: Job application conditions (Where Dense excels)",
    },
    {
        "query": "ساعات العمل الإضافي ومقابله المالي",
        "description": "Specific Legal Concept & Terms: Overtime hours & compensation",
    },
    {
        "query": "فصلوني تعسفياً من الخدمة شن حقي في التعويض؟",
        "description": "Libyan Dialect: Unfair termination & compensation",
    },
    {
        "query": "إجازة خاصة بمرتب كامل للحج",
        "description": "Keyword + Conceptual: Paid Hajj leave rules",
    },
]


def format_rank_str(r: dict) -> str:
    dense_r = f"Dense#{r['dense_rank']}" if r.get('dense_rank') else "Dense:—"
    bm25_r = f"BM25#{r['bm25_rank']}" if r.get('bm25_rank') else "BM25:—"
    return f"[{dense_r} | {bm25_r}]"


def main():
    print(f"\n{'='*75}")
    print("🔬  EVALUATING HYBRID RETRIEVAL (DENSE + BM25 + RRF FUSION)")
    print(f"{'='*75}\n")

    retriever = LibyanLawRetriever(bm25_enabled=True)

    for i, item in enumerate(TEST_QUERIES, 1):
        q = item["query"]
        desc = item["description"]

        print(f"\n{'#'*75}")
        print(f"# QUERY {i}: {q}")
        print(f"# Type   : {desc}")
        print(f"{'#'*75}")

        # Run Hybrid Search
        hybrid_results = retriever.search(q, top_k=3, mode="hybrid")

        print("\n  🏆 TOP-3 HYBRID RESULTS (Combined with RRF):")
        for rank, res in enumerate(hybrid_results, 1):
            ranks_str = format_rank_str(res)
            print(f"  {rank}. المادة ({res['article']}) — RRF: {res['rrf_score']} {ranks_str}")
            print(f"     الباب: {res['section'] or 'عام'} | الفصل: {res['chapter'] or 'عام'}")
            snippet = res['text'][:200].replace('\n', ' ')
            print(f"     مقتطف: \"{snippet}...\"\n")

    print(f"\n{'='*75}")
    print("✅  Hybrid retrieval evaluation complete.")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
