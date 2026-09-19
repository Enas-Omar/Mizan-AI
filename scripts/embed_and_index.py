"""
embed_and_index.py — Embed legal chunks and store them in a persistent ChromaDB collection.

Uses intfloat/multilingual-e5-small via ONNX Runtime (Xenova ONNX export).
No scipy/sklearn dependency. Runs fully locally for free.

Input:  data/chunks/all_chunks.json
Output: data/chroma_db/ (persistent Chroma collection 'libyan_labor_law')
"""

import os
import sys
import json
import warnings
import numpy as np
from tqdm import tqdm

import onnxruntime as ort
from tokenizers import Tokenizer
from huggingface_hub import hf_hub_download
import chromadb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CHUNKS_DIR, DATA_DIR

DB_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "libyan_labor_law"
ONNX_REPO = "Xenova/multilingual-e5-small"
BATCH_SIZE = 32
MAX_LENGTH = 512

warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


# ─── ONNX-based Multilingual E5 Embedder ───────────────────────────────────────

def load_model():
    """Download (if needed) and load the ONNX E5 model and tokenizer."""
    print(f"  ⏳ Loading ONNX model: {ONNX_REPO} ...")
    model_path = hf_hub_download(
        repo_id=ONNX_REPO,
        filename="onnx/model_quantized.onnx",
    )
    tok_path = hf_hub_download(
        repo_id=ONNX_REPO,
        filename="tokenizer.json",
    )

    tokenizer = Tokenizer.from_file(tok_path)
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_LENGTH)

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]

    print("  ✓ ONNX model and tokenizer loaded.")
    return tokenizer, session, input_names


def encode(texts: list[str], tokenizer, session, input_names, prefix: str = "passage") -> np.ndarray:
    """
    Encode a list of texts with the E5 model.
    prefix: 'passage' for documents, 'query' for search queries.
    """
    prefixed = [f"{prefix}: {t}" for t in texts]
    encoded = [tokenizer.encode(t) for t in prefixed]

    input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "token_type_ids" in input_names:
        onnx_inputs["token_type_ids"] = np.zeros_like(input_ids)

    outputs = session.run(None, onnx_inputs)
    last_hidden = outputs[0]

    # Mean pooling
    mask_expanded = np.expand_dims(attention_mask, -1).astype(np.float32)
    sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
    sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
    embeddings = sum_embeddings / sum_mask

    # L2 Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return (embeddings / np.clip(norms, a_min=1e-9, a_max=None)).astype(np.float32)


def main():
    print(f"\n{'='*60}")
    print("🚀  PART 2 — EMBEDDINGS & CHROMADB INDEXING")
    print(f"    Model: multilingual-e5-small (ONNX/CPU)")
    print(f"{'='*60}")

    # 1. Load chunks
    chunks_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  ✓ Loaded {len(chunks)} legal chunks")

    # 2. Load model
    tokenizer, session, input_names = load_model()

    # 3. Initialize ChromaDB
    os.makedirs(DB_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  ✓ Reset existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"  ✓ Created ChromaDB collection: '{COLLECTION_NAME}'")

    # 4. Prepare text and metadata
    ids, documents, metadatas, texts_to_embed = [], [], [], []

    for c in chunks:
        ids.append(c["chunk_id"])
        documents.append(c["text_with_context"])
        texts_to_embed.append(c["text_with_context"])
        metadatas.append({
            "chunk_id": str(c["chunk_id"]),
            "article":  str(c.get("article") or ""),
            "page":     int(c.get("page") or 0),
            "section":  str(c.get("section") or ""),
            "chapter":  str(c.get("chapter") or ""),
            "document": str(c.get("document") or ""),
            "doc_id":   str(c.get("doc_id") or ""),
            "source":   str(c.get("source") or ""),
            "year":     int(c.get("year") or 0),
            "doc_type": str(c.get("doc_type") or "law"),
        })

    # 5. Compute embeddings in batches
    print(f"  ⏳ Computing embeddings for {len(texts_to_embed)} passages...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts_to_embed), BATCH_SIZE), desc="Embedding", unit="batch"):
        batch = texts_to_embed[i:i + BATCH_SIZE]
        emb = encode(batch, tokenizer, session, input_names, prefix="passage")
        all_embeddings.append(emb)
    all_embeddings = np.vstack(all_embeddings)
    print(f"  ✓ Embedding matrix shape: {all_embeddings.shape}")

    # 6. Insert into ChromaDB in batches
    print("  ⏳ Inserting into ChromaDB...")
    for i in tqdm(range(0, len(ids), BATCH_SIZE), desc="Indexing", unit="batch"):
        end = i + BATCH_SIZE
        collection.add(
            ids=ids[i:end],
            embeddings=all_embeddings[i:end].tolist(),
            documents=documents[i:end],
            metadatas=metadatas[i:end],
        )

    count = collection.count()
    print(f"\n{'='*60}")
    print(f"🎉  Successfully indexed {count} chunks into ChromaDB!")
    print(f"    Database : {DB_DIR}")
    print(f"    Collection: {COLLECTION_NAME}")
    print(f"    Dimensions: {all_embeddings.shape[1]}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
