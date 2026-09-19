"""
Step 5 & 6 — Legal chunking and final retrieval dataset creation.

Takes the structured JSON records from Step 4 and produces the final
chunk records ready for embedding and loading into a vector database.

Each chunk follows this schema:
{
  "chunk_id":  "LAW12_ART25",
  "document":  "قانون علاقات العمل رقم 12 لسنة 2010",
  "doc_id":    "LAW_12_2010",
  "doc_type":  "law",
  "year":      2010,
  "section":   "الباب الثالث - الإجازات",
  "chapter":   "الفصل الأول",
  "article":   "25",
  "page":      18,
  "text":      "...",          ← article body
  "context":   "...",          ← full hierarchical breadcrumb prepended
  "source":    "law_12_2010.pdf"
}

"context" is the breadcrumb text prepended to the article body.
It gives the embedding model (and the LLM) the surrounding hierarchy
so it does not need to guess which law an article belongs to.

Input:  data/structured/<doc_id>_structured.json
Output: data/chunks/<doc_id>_chunks.json
        data/chunks/all_chunks.json       ← merged dataset from all documents
"""

import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOCUMENTS, STRUCTURED_DIR, CHUNKS_DIR, RAW_DIR

os.makedirs(CHUNKS_DIR, exist_ok=True)

# Maximum characters per chunk.
# Long articles (> MAX_CHARS) are split by paragraph (فقرة).
MAX_CHARS = 1500

# ─── Helpers ───────────────────────────────────────────────────────────────────

def make_chunk_id(doc_id: str, article: str, part: int = 0) -> str:
    """Generate a stable, readable chunk ID."""
    short = doc_id.replace("_", "").replace("LAW", "LAW").replace("DECISION", "DEC").replace("DOC", "DOC")
    if "إصدار" in article:
        num = re.sub(r"\D", "", article)
        art_clean = f"ISDAR{num}"
    else:
        art_clean = re.sub(r"\W", "", article)
    if part > 0:
        return f"{short}_ART{art_clean}_P{part}"
    return f"{short}_ART{art_clean}"



def build_context(record: dict) -> str:
    """
    Build the hierarchical breadcrumb string for a chunk.
    This is prepended to the article text before embedding, giving
    the model the full legal context without bloating the stored chunk.

    Example:
      قانون علاقات العمل رقم 12 لسنة 2010
      الباب: الباب الثالث - الإجازات
      الفصل: الفصل الأول - الإجازة السنوية
      المادة 25:
    """
    parts = [record["title"]]
    if record.get("section"):
        parts.append(f"الباب: {record['section']}")
    if record.get("chapter"):
        parts.append(f"الفصل: {record['chapter']}")
    parts.append(f"المادة {record['article']}:")
    return "\n".join(parts)


def split_long_article(text: str) -> list[str]:
    """
    If an article body exceeds MAX_CHARS, split it at paragraph boundaries.
    Paragraphs in Arabic legal text are often delimited by:
      - Newlines followed by أ) ب) ج) (lettered sub-items)
      - Newlines followed by 1- 2- 3- (numbered sub-items)
      - Double newlines
    """
    if len(text) <= MAX_CHARS:
        return [text]

    # Try splitting on lettered paragraph markers: أ - ب - ج
    para_split = re.split(r"\n(?=[أبتثجحخدذرزسشصضطظعغفقكلمنهوي]\s*[).\-])", text)
    if len(para_split) > 1:
        return merge_short_parts(para_split)

    # Fallback: split on numbered markers 1- 2- 3-
    para_split = re.split(r"\n(?=\d+\s*[).\-])", text)
    if len(para_split) > 1:
        return merge_short_parts(para_split)

    # Last resort: split on double newlines
    para_split = re.split(r"\n{2,}", text)
    return merge_short_parts(para_split)


def merge_short_parts(parts: list[str]) -> list[str]:
    """
    Merge neighbouring short parts to avoid micro-chunks.
    A part must be at least 100 chars, otherwise it merges with the next.
    """
    merged, buffer = [], ""
    for part in parts:
        buffer = (buffer + "\n" + part).strip() if buffer else part.strip()
        if len(buffer) >= 200:
            merged.append(buffer)
            buffer = ""
    if buffer:
        merged.append(buffer)
    return merged


# ─── Chunker ───────────────────────────────────────────────────────────────────

def chunk_document(doc_cfg: dict) -> list[dict]:
    structured_path = os.path.join(STRUCTURED_DIR, f"{doc_cfg['doc_id']}_structured.json")

    if not os.path.exists(structured_path):
        print(f"  ✗ Structured file not found (run step4 first): {structured_path}")
        return []

    with open(structured_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    seen_ids = set()
    chunks = []
    for record in records:
        article_text = record["text"]
        context      = build_context(record)
        parts        = split_long_article(article_text)

        for i, part in enumerate(parts):
            part_index = i if len(parts) > 1 else 0
            chunk_id   = make_chunk_id(record["document_id"], record["article"], part_index)
            base_id = chunk_id
            suffix = 1
            while chunk_id in seen_ids:
                suffix += 1
                chunk_id = f"{base_id}_V{suffix}"
            seen_ids.add(chunk_id)


            chunk = {
                "chunk_id":  chunk_id,
                "document":  record["title"],
                "doc_id":    record["document_id"],
                "doc_type":  record["doc_type"],
                "year":      record["year"],
                "section":   record.get("section"),
                "chapter":   record.get("chapter"),
                "article":   record["article"],
                "page":      record["page"],
                # The actual text used for embedding = breadcrumb + article body.
                # Storing separately so the UI can display the clean body only.
                "text":      part,
                "context":   context,
                "text_with_context": f"{context}\n\n{part}",
                "source":    doc_cfg["pdf_file"],
                "char_count": len(part),
            }
            chunks.append(chunk)

    return chunks


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n✂️   STEP 5+6 — LEGAL CHUNKING & RETRIEVAL DATASET")

    all_chunks = []

    for doc in DOCUMENTS:
        print(f"\n{'='*60}")
        print(f"  Document : {doc['title']}")

        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

        # Save per-document chunks
        out_path = os.path.join(CHUNKS_DIR, f"{doc['doc_id']}_chunks.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        long_chunks = sum(1 for c in chunks if c["char_count"] > MAX_CHARS // 2)
        print(f"  ✓ Chunks created   : {len(chunks)}")
        print(f"  ✓ Long chunks (split) : {long_chunks}")
        print(f"  ✓ Avg chars/chunk  : {sum(c['char_count'] for c in chunks) // max(len(chunks), 1)}")
        print(f"  ✓ Saved : {out_path}")

        if chunks:
            sample = chunks[0]
            print(f"\n  Sample chunk:")
            print(f"    chunk_id : {sample['chunk_id']}")
            print(f"    article  : {sample['article']}")
            print(f"    context  :\n      {sample['context']}")
            print(f"    text     : {sample['text'][:100]}...")

    # Save the merged dataset (all documents combined)
    merged_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  📦  Total chunks across all documents : {len(all_chunks)}")
    print(f"  ✓   Merged dataset saved : {merged_path}")
    print("\n✅  Dataset is ready for embedding (Step 7 — Vector DB)")


if __name__ == "__main__":
    main()
