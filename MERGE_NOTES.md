# Mizan AI — merge metadata (internal record)
Merged from:
- "first prototype": src/ package architecture, streaming, metrics, multi-criteria
  relevance gate, source filter, Groq model cache, evaluation harness,
  DECISION_888_2023 (executive regulation of Labor Law 12/2010).
- "second attempt": 11-law indexed knowledge base (2,923 chunks), conversational
  memory + follow-up reformulation, greeting handler, anti-repetition engine,
  strict anti-hallucination prompts, hybrid Dense+BM25+RRF retrieval.

Merge decisions:
- Knowledge base: proto2 (11 laws) + proto1's DECISION_888_2023 => 12 docs, 2,929 chunks.
- Assistant: proto1's layered relevance gate (confidence bug fixed: now uses
  rank-weighted dense cosine similarity, not RRF score) + proto2's greeting
  detection, contextualize_query and conversational memory.
- Generator: proto1's streaming (Groq/Ollama) + model cache + dynamic doc-type
  prompt + proto2's history-aware generation, strong anti-repetition prompt,
  and Jaccard dedup post-processing (applied after the stream completes).
- UI (app.py): streaming + live metrics (retrieval/gen time, provider, confidence)
  + sidebar source filter over the 12 laws + citations + persona presets.
- Pipeline scripts copied from proto2 with sys.path fixed to project root;
  winocr moved out of core requirements (lazy import, Windows-only OCR step).
- data/chunks/DECISION_888_2023_chunks.json added to all_chunks.json and embedded
  into the existing ChromaDB collection via scripts/add_chunks_to_index.py.
