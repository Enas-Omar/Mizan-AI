"""
backfill_doc_id_metadata.py — One-off maintenance script.

The original index (proto2) stored no `doc_id` in ChromaDB metadata, which the
source-filter feature needs. This script sets `doc_id` (and `year` fix) on every
chunk from data/chunks/all_chunks.json.

Usage:
    python scripts/backfill_doc_id_metadata.py
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb

from config import CHUNKS_DIR, CHROMA_DIR, CHROMA_COLLECTION


def main():
    all_chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(all_chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    id_to_doc_id = {c["chunk_id"]: str(c.get("doc_id") or "") for c in all_chunks}

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(CHROMA_COLLECTION)
    total = collection.count()
    print(f"Collection '{CHROMA_COLLECTION}' has {total} chunks")

    # Fetch all ids + metadatas
    got = collection.get(include=["metadatas"])
    ids = got["ids"]
    metas = got["metadatas"]

    update_ids = []
    update_metas = []
    for cid, meta in zip(ids, metas):
        doc_id = id_to_doc_id.get(cid, "")
        if doc_id and meta.get("doc_id") != doc_id:
            meta = dict(meta)
            meta["doc_id"] = doc_id
            update_ids.append(cid)
            update_metas.append(meta)

    print(f"{len(update_ids)} chunks need doc_id backfill")
    BATCH = 500
    for i in range(0, len(update_ids), BATCH):
        collection.update(ids=update_ids[i:i + BATCH], metadatas=update_metas[i:i + BATCH])
        print(f"  updated {min(i + BATCH, len(update_ids))}/{len(update_ids)}")

    print("Done.")


if __name__ == "__main__":
    main()
