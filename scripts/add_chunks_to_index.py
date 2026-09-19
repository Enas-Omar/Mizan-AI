"""
add_chunks_to_index.py — Append pre-chunked document(s) to the existing index.

Takes a chunks JSON file (same schema as data/chunks/*.json), de-duplicates by
chunk_id, appends to data/chunks/all_chunks.json (BM25 corpus) and embeds the
new chunks into the persistent ChromaDB collection (dense index).

Usage:
    python scripts/add_chunks_to_index.py data/chunks/DECISION_888_2023_chunks.json
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chromadb
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import onnxruntime as ort

from config import CHUNKS_DIR, CHROMA_DIR, CHROMA_COLLECTION

ONNX_REPO = "Xenova/multilingual-e5-small"
MAX_LENGTH = 512
BATCH_SIZE = 32


def load_embedder():
    print(f"  ⏳ Loading ONNX model: {ONNX_REPO} ...")
    model_path = hf_hub_download(repo_id=ONNX_REPO, filename="onnx/model_quantized.onnx")
    tok_path = hf_hub_download(repo_id=ONNX_REPO, filename="tokenizer.json")

    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_LENGTH)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    print("  ✓ ONNX model and tokenizer loaded.")
    return tokenizer, session, input_names


def encode(texts: list[str], tokenizer, session, input_names, prefix: str = "passage") -> np.ndarray:
    prefixed = [f"{prefix}: {t}" for t in texts]
    encoded = [tokenizer.encode(t) for t in prefixed]

    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in input_names:
        onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

    outputs = session.run(None, onnx_inputs)
    last_hidden = outputs[0]

    mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    embeddings = sum_embeddings / sum_mask

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / np.clip(norms, a_min=1e-9, a_max=None)).astype(np.float32)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    chunks_file = sys.argv[1]
    if not os.path.isabs(chunks_file):
        chunks_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), chunks_file)

    with open(chunks_file, "r", encoding="utf-8") as f:
        new_chunks = json.load(f)
    print(f"  ✓ Loaded {len(new_chunks)} chunks from {os.path.basename(chunks_file)}")

    all_chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(all_chunks_path, "r", encoding="utf-8") as f:
        all_chunks = json.load(f)

    existing_ids = {c["chunk_id"] for c in all_chunks}
    to_add = [c for c in new_chunks if c["chunk_id"] not in existing_ids]
    skipped = len(new_chunks) - len(to_add)
    print(f"  ✓ {len(to_add)} new chunks to add ({skipped} already indexed — skipped)")

    if not to_add:
        print("Nothing to do.")
        return

    # 1. Append to BM25 corpus
    all_chunks.extend(to_add)
    with open(all_chunks_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"  ✓ all_chunks.json now contains {len(all_chunks)} chunks")

    # 2. Embed + insert into ChromaDB
    tokenizer, session, input_names = load_embedder()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(CHROMA_COLLECTION)
    print(f"  ✓ Collection '{CHROMA_COLLECTION}' currently has {collection.count()} chunks")

    ids = [c["chunk_id"] for c in to_add]
    documents = [c["text_with_context"] for c in to_add]
    texts_to_embed = documents
    metadatas = [{
        "chunk_id": str(c["chunk_id"]),
        "article":  str(c.get("article") or ""),
        "page":     int(c.get("page") or 0),
        "section":  str(c.get("section") or ""),
        "chapter":  str(c.get("chapter") or ""),
        "document": str(c.get("document") or ""),
        "doc_id":   str(c.get("doc_id") or ""),
        "source":   str(c.get("source") or ""),
        "year":     int(c.get("year") or 0) if str(c.get("year") or "0").isdigit() else 0,
        "doc_type": str(c.get("doc_type") or "law"),
    } for c in to_add]

    print(f"  ⏳ Computing embeddings for {len(texts_to_embed)} passages...")
    for i in range(0, len(texts_to_embed), BATCH_SIZE):
        batch = texts_to_embed[i:i + BATCH_SIZE]
        emb = encode(batch, tokenizer, session, input_names, prefix="passage")
        end = i + BATCH_SIZE
        collection.add(
            ids=ids[i:end],
            embeddings=emb.tolist(),
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"    indexed {min(end, len(ids))}/{len(ids)}")

    print(f"\n🎉 Done! Collection now has {collection.count()} chunks.")
    print("   (BM25 corpus updated too — restart the app to pick up changes.)")


if __name__ == "__main__":
    main()
