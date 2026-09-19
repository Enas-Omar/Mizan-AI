# -*- coding: utf-8 -*-
"""
test_metrics.py — التقييم الكمي لمشروع الميزان (Golden Set Evaluation).

المقاييس المقاسة (مطابقة لمتطلبات التقييم):
  1. Retrieval:  Recall@K (K=1,3,5) و MRR@5 على صفوف الإجابة في golden_set.json
  2. Citation completeness : اكتمال حقول السند (المادة/الوثيقة/الصفحة)
  3. Citation correctness  : مطابقة أعلى سند مع التوسيم الذهبي
  4. Grounded-answer rate  : نسبة الأسئلة القانونية التي أنتجت إجابة مسنَدة
  5. Refusal accuracy      : دقة الرفض (خارج النطاق / لا جواب) على صفوف الرفض

سياسة منع التسريب (Leakage):
  - المتن مجمد: لا يُعدَّل الفهرس بأي سؤال من هذه المجموعة قبل/أثناء التقييم.
  - المجموعة تحمل مراجع توسيم (doc_id + article) فقط، لا نصوصاً من المتن.
  - أي أسئلة ضبط/تطوير توضع في golden_set_dev.json منفصلة.

الاستخدام:
  python evaluation/test_metrics.py            # تقييم كامل
  python evaluation/test_metrics.py --skip-g   # تخطي قسم الإجابات (سريع: رفض فقط)

الناتج: تقرير في الطرفية + evaluation/results_last_run.json
"""

import os
import sys
import json
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mizan.assistant import MizanAssistant

GOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_last_run.json")

RETRIEVAL_K = 5          # عمق التقييم
RANK_WEIGHTS = [1.0, 0.5, 1/3, 0.25, 0.2]   # أوزان الثقة الموزونة بالمرتبة


def load_gold():
    with open(GOLD_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["answer_rows"], data["refusal_rows"]


def expected_key(expected):
    """(doc_id, article) كمفتاح موحد للمطابقة."""
    return {(e["doc_id"], str(e["article"])) for e in expected}


def source_key(src):
    return (src.get("doc_id") or "", str(src.get("article") or ""))


# ─────────────────────────── 1) Retrieval metrics ──────────────────────────
def eval_retrieval(retriever, answer_rows, verbose=True):
    recalls = {1: 0, 3: 0, 5: 0}
    rr_scores = []
    per_row = []

    for row in answer_rows:
        gold_keys = expected_key(row["expected"])
        hits = retriever.search(row["query"], top_k=RETRIEVAL_K, mode="hybrid")
        ranks = [i + 1 for i, h in enumerate(hits) if source_key(h) in gold_keys]

        for k in (1, 3, 5):
            if any(r <= k for r in ranks):
                recalls[k] += 1
        rr_scores.append(1.0 / ranks[0] if ranks else 0.0)

        per_row.append({
            "id": row["id"],
            "query": row["query"],
            "expected": row["expected"],
            "retrieved_top5": [{"doc_id": h.get("doc_id"), "article": h.get("article"),
                                "dense_score": round(h.get("dense_score") or 0, 4),
                                "bm25_score": round(h.get("bm25_score") or 0, 2)}
                               for h in hits],
            "first_hit_rank": ranks[0] if ranks else None,
        })

        if verbose:
            ok = "OK " if ranks else "MISS"
            print(f"  [{ok}] {row['id']}: hit_rank={ranks[0] if ranks else '-'}  «{row['query'][:40]}»")

    n = len(answer_rows)
    return {
        "recall_at_1": round(recalls[1] / n, 3),
        "recall_at_3": round(recalls[3] / n, 3),
        "recall_at_5": round(recalls[5] / n, 3),
        "mrr_at_5": round(sum(rr_scores) / n, 3),
        "n_rows": n,
        "per_row": per_row,
    }


# ─────────────────────── 2) Citation metrics + grounded rate ───────────────
def eval_citations(assistant, answer_rows, verbose=True):
    complete = correct = grounded = 0
    details = []

    for row in answer_rows:
        res = assistant.answer_question(query=row["query"])
        cits = res.get("citations") or []

        is_grounded = (not res.get("no_match", True)) and bool(cits)
        grounded += 1 if is_grounded else 0

        is_complete = is_grounded and all(
            c.get("article") and c.get("document") and c.get("page") is not None
            for c in cits
        )
        complete += 1 if is_complete else 0

        gold_keys = expected_key(row["expected"])
        top = cits[0] if cits else None
        is_correct = bool(top) and (top.get("doc_id"), str(top.get("article"))) in gold_keys
        correct += 1 if is_correct else 0

        details.append({
            "id": row["id"],
            "grounded": is_grounded,
            "citation_complete": is_complete,
            "top_citation_correct": is_correct,
            "n_citations": len(cits),
            "confidence": round(res.get("confidence") or 0, 3),
        })

        if verbose:
            print(f"  {row['id']}: grounded={is_grounded} complete={is_complete} correct={is_correct} cit={len(cits)}")

    n = len(answer_rows)
    return {
        "grounded_answer_rate": round(grounded / n, 3),
        "citation_completeness": round(complete / n, 3),
        "citation_correctness_top1": round(correct / n, 3),
        "n_rows": n,
        "per_row": details,
    }


# ───────────────────────────── 3) Refusal accuracy ─────────────────────────
def eval_refusals(assistant, refusal_rows, verbose=True):
    correct = 0
    details = []
    for row in refusal_rows:
        res = assistant.answer_question(query=row["query"])
        if res.get("no_match"):
            # تمييز نوعي الرفض عبر نص الرسالة (نفس منطق test_gates)
            got = "refuse_out_of_scope" if "خارج نطاق" in res.get("answer", "") else "refuse_no_answer"
        else:
            got = "answer"
        is_ok = (got == row["expected_behavior"])
        correct += 1 if is_ok else 0
        details.append({"id": row["id"], "expected": row["expected_behavior"], "got": got, "pass": is_ok})
        if verbose:
            print(f"  [{'OK ' if is_ok else 'FAIL'}] {row['id']}: expected={row['expected_behavior']} got={got}")

    n = len(refusal_rows)
    return {"refusal_accuracy": round(correct / n, 3), "n_rows": n, "per_row": details}


# ─────────────────────────────────── main ──────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Golden-set quantitative evaluation")
    parser.add_argument("--skip-g", action="store_true", help="skip generation-dependent section")
    args = parser.parse_args()

    print("=" * 70)
    print("Mizan AI — Golden Set Evaluation (Recall@K / MRR / Citations / Refusal)")
    print("=" * 70)

    answer_rows, refusal_rows = load_gold()

    # retriever only (بلا مولد) لقسم الاسترجاع
    from mizan.retriever import LibyanLawRetriever
    retriever = LibyanLawRetriever(bm25_enabled=True)

    print("\n[1/3] Retrieval: Recall@1/3/5 + MRR@5 (hybrid, top-5 pool)")
    retrieval = eval_retrieval(retriever, answer_rows)
    print(f"  -> Recall@1={retrieval['recall_at_1']}  Recall@3={retrieval['recall_at_3']}  "
          f"Recall@5={retrieval['recall_at_5']}  MRR@5={retrieval['mrr_at_5']}")

    assistant = MizanAssistant(provider="fallback")

    print("\n[2/3] Citations + grounded answers (fallback generator)")
    citations = eval_citations(assistant, answer_rows)
    print(f"  -> grounded={citations['grounded_answer_rate']}  "
          f"complete={citations['citation_completeness']}  correct@1={citations['citation_correctness_top1']}")

    print("\n[3/3] Refusal accuracy")
    refusals = eval_refusals(assistant, refusal_rows)
    print(f"  -> refusal_accuracy={refusals['refusal_accuracy']} ({refusals['n_rows']} rows)")

    results = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_frozen": True,
        "gold_set_version": "1.0.0",
        "retrieval": retrieval,
        "citations": citations,
        "refusals": refusals,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {os.path.relpath(RESULTS_PATH)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
