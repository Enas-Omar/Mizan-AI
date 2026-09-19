"""
assistant.py — Central RAG Orchestrator for Mizan AI.

Integrates:
  - Hybrid Retriever (ChromaDB Vector + Arabic BM25 + RRF Fusion)
  - Legal Answer Generator (LLM with strict grounding and citation)

Features:
  - Greeting / chitchat detection (bypasses RAG)
  - Follow-up query reformulation using conversation history
  - Multi-criteria relevance gate (dense score + BM25 + weighted confidence)
  - Streaming support
  - Metadata filtering (doc_id, doc_type, year, specialization)
  - Extensible to any legal source
"""

import os
import re

from mizan.retriever import LibyanLawRetriever
from mizan.generator import LegalAnswerGenerator
from mizan.bm25_search import tokenize_arabic

# ─── Relevance thresholds ─────────────────────────────────────────────────────
# Legal questions score >= 0.83 dense cosine; unrelated questions <= 0.79.
DENSE_HIGH_THRESHOLD     = 0.83
DENSE_MEDIUM_THRESHOLD   = 0.75
DENSE_LEGAL_TERM_THRESHOLD = 0.78
BM25_STRONG_THRESHOLD    = 5.0
CONFIDENCE_THRESHOLD     = 0.75

# ─── Greeting / Chitchat detection ────────────────────────────────────────────
# Gate 1 – the message must contain one of these specific greeting PHRASES.
# Gate 2 – the message must NOT contain any legal keyword.
# Both gates must pass for a message to be classified as a greeting.

_GREETING_PHRASES = [
    "السلام عليكم", "عليكم السلام",
    "أهلا وسهلا", "أهلاً وسهلاً", "أهلا", "أهلاً", "اهلا",
    "هلا", "مرحبا", "مرحباً", "مرحبتين",
    "صباح الخير", "صباح النور", "مساء الخير", "مساء النور",
    "كيف حالك", "كيف الحال", "كيفك",
    "شكرا", "شكراً", "شكرا جزيلا", "شكراً جزيلاً",
    "وداعا", "وداعاً", "مع السلامة", "تصبح على خير", "تمسى على خير",
    "hi", "hello", "thanks", "thank you", "bye", "good morning",
]

# If ANY of these words appear → it is a legal question, not a greeting
_LEGAL_KEYWORDS = [
    "قانون", "مادة", "ماد", "عقد", "شركة", "ضريبة", "ضرائب",
    "عمل", "راتب", "أجر", "اجر", "إجازة", "اجازة", "إجازات", "اجازات",
    "موظف", "عامل", "صاحب العمل", "فصل", "إنهاء", "انهاء",
    "تعويض", "غرامة", "جزاء", "عقوبة",
    "محكمة", "حق", "حقوق", "التزام", "واجب", "مصرف", "بنك",
    "ربح", "خسارة", "نظام", "لائحة", "قرار", "تأسيس", "شريك",
    "مكافأة", "تأمين", "تامين", "ميراث", "وصية", "عقار", "بيع", "إيجار",
    "رهن", "كفالة", "وكالة", "ضمان", "تجاري", "مدني", "مالي",
    "ربوي", "غسل", "أموال", "اموال", "أوراق مالية", "سوق", "نسبة", "رأس المال",
    "ملكية", "حكم", "نزاع", "اتفاق", "بند", "فقرة", "باب",
    "تسجيل", "ترخيص", "إفلاس", "افلاس", "تصفية", "دعوى", "شهادة", "توثيق",
    "تقاعد", "معاش", "طفل", "حقوق الطفل",
]


def _is_greeting(query: str) -> bool:
    """
    Returns True ONLY if:
      1. The query contains a known greeting phrase, AND
      2. The query contains NO legal keyword.
    """
    q = query.strip().lower()

    # Gate 2 first (cheaper) — any legal keyword disqualifies immediately
    if any(kw in q for kw in _LEGAL_KEYWORDS):
        return False

    # Gate 1 — must match a specific greeting phrase
    return any(phrase in q for phrase in _GREETING_PHRASES)


def _format_greeting_response(query: str) -> str:
    """
    Returns an appropriate reply matching the user's greeting,
    then invites them to ask their legal question.
    """
    q = query.strip().lower()
    if "سلام" in q and "عليكم" in q or "السلام عليكم" in q:
        reply = "وعليكم السلام ورحمة الله وبركاته! 😊"
    elif "صباح" in q:
        reply = "صباح النور والسرور! ☀️"
    elif "مساء" in q:
        reply = "مساء النور والسرور! 🌙"
    elif any(w in q for w in ["شكر", "thanks", "thank"]):
        reply = "العفو، على الرحب والسعة دائماً! 😊"
    elif any(w in q for w in ["وداع", "سلامة", "bye"]) or "مع السلامة" in q:
        reply = "في أمان الله وحفظه! مع السلامة. 👋"
    else:
        # Default for أهلا, مرحبا, هلا, hi, hello, etc.
        reply = "أهلاً وسهلاً بك! 😊"

    return (
        f"{reply}\n\n"
        "أنا **الميزان — المستشار القانوني الليبي الذكي**، جاهز لمساعدتك في أي استفسار حول القوانين والتشريعات الليبية.\n\n"
        "هل لديك سؤال أو استفسار قانوني تودّ طرحه؟ 🏛️"
    )


# ─── No-answer response ────────────────────────────────────────────────────────
NO_ANSWER_RESPONSE = (
    "أهلاً بك! يسعدني دائماً مساعدتك.\n\n"
    "**لا أعرف الإجابة الدقيقة عن هذا السؤال، إذ لا توجد مواد قانونية كافية أو مرتبطة به في قاعدة المعرفة الحالية.**\n\n"
    "💡 **أسباب محتملة:**\n"
    "- الموضوع المطروح قد لا يكون منظماً في أي من التشريعات المتاحة (موضوعات عامة، أسعار، سياسة، إلخ).\n"
    "- قد يتبع تشريعاً ليبياً لم يُضف بعد إلى قاعدة المعرفة.\n"
    "- يمكنك إعادة صياغة السؤال بمصطلحات قانونية أكثر تحديداً للحصول على نتيجة أفضل.\n\n"
    "📚 **التشريعات المتاحة حالياً:**\n"
    "- قانون علاقات العمل رقم (12) لسنة 2010 ولائحته التنفيذية (قرار 888 لسنة 2023)\n"
    "- القانون المدني الليبي\n"
    "- دليل حقوق الطفل في ليبيا\n"
    "- قانون المصارف والصيرفة الإسلامية رقم (46) لسنة 2012\n"
    "- قانون ضرائب الدخل رقم (7) لسنة 2010 ولائحته التنفيذية (قرار 592 لسنة 2010)\n"
    "- قانون منع المعاملات الربوية رقم (1) لسنة 2013\n"
    "- قانون مكافحة غسل الأموال رقم (2) لسنة 2005\n"
    "- قانون النظام المالي للدولة\n"
    "- النظام الأساسي لسوق الأوراق المالية الليبي (2006)\n"
    "- قانون النشاط التجاري رقم (23) لسنة 2010 (النص الكامل)\n\n"
    "⚖️ تنويه: المعلومات المقدمة استرشادية مبنية على التشريعات الليبية المتاحة في قاعدة المعرفة."
)

# ─── Out-of-scope response ──────────────────────────────────────────────────────
OUT_OF_SCOPE_RESPONSE = (
    "أهلاً بك! يسعدني دائماً مساعدتك.\n\n"
    "**هذا السؤال خارج نطاق عملي كمستشار قانوني.** أنا **الميزان — المستشار القانوني الليبي الذكي**،"
    " متخصص في التشريعات والقوانين الليبية فقط، ولا أستطيع الإفادة في الموضوعات العامة "
    "(مدارس وجامعات، مطاعم وأسواق، رياضة، طقس، سياسة وأخبار، برامج وتطبيقات... إلخ).\n\n"
    "💡 جرّب أن تسأل سؤالاً قانونياً ليبياً، مثلاً:\n"
    "- «ما هي أحكام الإجازة السنوية؟»\n"
    "- «شن حقوقي إذا فصلوني تعسفياً من العمل؟»\n"
    "- «ما هي شروط تأسيس شركة تجارية؟»\n\n"
    "⚖️ تنويه: المعلومات المقدمة استرشادية مبنية على التشريعات الليبية المتاحة في قاعدة المعرفة."
)

# ─── Legal terms (used by the relevance gate) ──────────────────────────────────
LEGAL_TERMS = {
    # Labor
    "عمل", "أجر", "اجر", "عقد", "فصل", "إجازة", "اجازه", "خدمة", "وظيفة",
    "موظف", "عامل", "صاحب", "إنهاء", "انهاء", "استقالة", "تجربة", "ساعات",
    "تعويض", "مرتب", "راتب", "جزاء", "تأديب", "تاديب",
    # Pension / Social Security
    "تقاعد", "معاش", "ضمان", "اجتماعي", "شيخوخة", "عجز", "ورثة", "تأمين", "تامين",
    # Tax
    "ضريبة", "ضرائب", "جباية", "تهرب", "إقرار", "وعاء", "رسم",
    # Civil / Commercial / General
    "مادة", "قانون", "لائحة", "قرار", "حقوق", "واجبات",
    "محكمة", "قضية", "دعوى", "حكم", "استئناف", "نقض",
    "شركة", "تأسيس", "تجاري", "مدني", "مصرف", "بنك", "ربوي", "غسل",
    "أوراق", "سوق", "طفل", "ميراث", "وصية", "عقار", "بيع", "إيجار", "رهن",
    # Scope-gate additions (institutional / procedural / general legal)
    "قطاع", "وزارة", "هيئة", "محكمة", "قاض", "محام", "عدالة", "عدل",
    "تشريع", "قانوني", "نافذ", "سجل", "شهادة", "إقامة", "شهود",
}


def _has_explicit_article(query: str) -> bool:
    return bool(re.search(r"ماد[ةه]\s*\d+", query))


def _has_legal_term(query: str) -> bool:
    try:
        tokens = set(tokenize_arabic(query))
    except Exception:
        tokens = set(query.split())
    norm_terms = set(tokenize_arabic(" ".join(LEGAL_TERMS)))
    for t in tokens:
        core = _ar_core(t)
        for term in norm_terms:
            if core == term:
                return True
            # Fuzzy stem match only for substantial roots (avoid short tokens
            # like 'ما'/'هي' matching long terms by mere prefix).
            if len(core) >= 4 and len(term) >= 4 and (
                core.startswith(term) or term.startswith(core)
            ):
                return True
    return False


# ─── Domain-scope detection & lexical topic verification ──────────────────────
# Function words / question words — carry no legal "topic" signal.
_STOPWORDS = {
    "ما", "ماذا", "من", "في", "على", "عن", "الى", "او", "و", "ثم", "ايضا",
    "كذلك", "هو", "هي", "ان", "لا", "لن", "قد", "لو", "اذا", "بل", "غير",
    "نفس", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي", "الذين", "اما", "ام",
    "لكن", "حتى", "فقط", "بعض", "كل", "لدي", "عند", "عندي", "مع", "يكون",
    "تكون", "يوجد", "بعد", "قبل", "بين", "هل", "كيف", "لماذا", "متى", "كم",
    "شن", "شنو", "قداش", "وين", "علاش", "هلا", "بش", "اي", "نبي", "اريد",
    "اعرف", "اخبرني", "رجاء", "برجاء", "شكرا", "شكراً", "هل", "وهل",
    # Quantity / vague words
    "اكثر", "اقل", "ابدا", "جدا",
    # Meta tokens injected by reformulation prompts ("في سياق: ...")
    "سياق", "سالت", "سابقا", "سابق", "سالف",
}

_TOKEN_TRIM = "؟?.,:;«»()[]\"'-،؛"


def _ar_core(t: str) -> str:
    """
    Iteratively strip Arabic prepositional compounds (وال/بال/كال/فال/لل) and
    the definite article (ال) from a word, plus surrounding punctuation.
    Single-letter conjunctions (و/ب) are NOT stripped: they would break
    words genuinely starting with them (وظيفة, وعاء, بنك, بيع...).
    Attached junk forms (وماذا, وسبق) simply never match doc content, which
    only lowers the denominator safely.
    """
    t = t.strip(_TOKEN_TRIM)
    while len(t) > 3:
        for p in ("وال", "بال", "كال", "فال", "لل", "ال"):
            if t.startswith(p) and len(t) > len(p) + 2:
                t = t[len(p):]
                break
        else:
            return t
    return t


def _is_legal_query(query: str) -> bool:
    """
    Scope gate: is the question a legal-domain question at all?
    True if it contains an explicit article reference, or any lexicon word.
    Non-legal questions (schools, restaurants, sports...) are refused
    before any retrieval happens.
    """
    if _has_explicit_article(query):
        return True
    return bool(_has_legal_term(query))


def _content_tokens(text: str) -> set:
    """Normalized contentful words (drop article + stopwords)."""
    try:
        tokens = tokenize_arabic(text)
    except Exception:
        tokens = text.split()
    out = set()
    for t in tokens:
        core = _ar_core(t)
        if core in _STOPWORDS or t in _STOPWORDS:
            continue
        if len(core) < 3:
            continue
        out.add(core)
    return out


def _tokens_related(tok: str, doc_tokens: set) -> bool:
    """Loose Arabic stem match: handles article prefix + simple inflection."""
    for d in doc_tokens:
        if d == tok:
            return True
        if len(tok) >= 3 and d.startswith(tok):
            return True
        if len(d) >= 3 and tok.startswith(d):
            return True
    return False


# Corpus document-frequency cache: (df_counter, total_chunks) per corpus.
_CORPUS_DF_CACHE: dict[int, tuple] = {}


def _compute_corpus_df(bm25_searcher) -> tuple | None:
    corpus = getattr(bm25_searcher, "corpus_tokens", None)
    if not corpus:
        return None
    key = id(bm25_searcher)
    cached = _CORPUS_DF_CACHE.get(key)
    if cached is not None:
        return cached
    from collections import Counter
    df = Counter()
    for doc in corpus:
        df.update(set(doc))
    result = (df, len(corpus))
    _CORPUS_DF_CACHE[key] = result
    return result


def _passes_topic_match(
    retrieval_query: str,
    sources: list[dict],
    corpus_df: tuple | None = None,
) -> bool:
    """
    Second verification gate (lexical): after the score gate accepts a chunk,
    confirm the chunk actually SHARES contentful words with the question.
    This blocks authoritative-looking answers built from legally-plausible but
    topically-unrelated articles (e.g. a follow-up about aspects the knowledge
    base does not cover). Score-based gates alone miss these.

    Words that appear in a large share of the corpus (e.g. عمل, قانون) cannot
    validate overlap on their own -- otherwise every chunk would 'overlap' with
    every question. Bypassed for explicit article lookups (e.g. 'المادة 34').
    """
    if not sources:
        return False
    if _has_explicit_article(retrieval_query):
        return True

    q_tokens = _content_tokens(retrieval_query)
    if not q_tokens:
        return True  # nothing to verify lexically

    common_threshold = 0.15  # appears in >15% of chunks -> corpus-ubiquitous
    n_chunks_total = 0
    df = None
    if corpus_df:
        df, n_chunks_total = corpus_df

    def _is_common(w: str) -> bool:
        return df is not None and df.get(w, 0) / n_chunks_total > common_threshold

    # Drop corpus-ubiquitous words from the question's topic signal.
    q_topic = {w for w in q_tokens if not _is_common(w)}
    if not q_topic:
        return True  # question is fully generic; trust the score gate

    for s in sources:
        doc_tokens = _content_tokens(s.get("text") or "")
        if not doc_tokens:
            continue
        doc_topic = {w for w in doc_tokens if not _is_common(w)}
        matched = [w for w in q_topic if _tokens_related(w, doc_topic)]
        ratio = len(matched) / len(q_topic)
        # Accept ONLY dominant overlap: more than half of the question's topic
        # roots appear in the chunk (prevents legally-plausible junk matches
        # like عمل/عامل/امر from validating unrelated articles).
        if len(matched) >= 2 and ratio >= 0.55:
            return True
        if len(matched) >= 1 and len(q_topic) <= 2:
            return True
    return False


def _compute_confidence(sources: list[dict]) -> float:
    """
    Weighted average of DENSE cosine similarities (rank-weighted).
    Uses dense_score (0..1 semantic similarity), NOT the RRF score,
    so the value is interpretable and comparable to the thresholds.
    """
    if not sources:
        return 0.0
    weights = [1.0 / (i + 1) for i in range(len(sources))]
    total_weight = sum(weights)
    weighted = sum(w * (s.get("dense_score") or 0) for w, s in zip(weights, sources))
    return round(weighted / total_weight, 3)


def _is_relevant(query: str, sources: list[dict], mode: str) -> bool:
    """
    Multi-criteria relevance gate. Returns True if at least one retrieved chunk satisfies:
    1. Explicit article query (e.g., 'المادة 15') is always accepted.
    2. Dense cosine similarity >= DENSE_HIGH_THRESHOLD (0.83).
    3. Rank-weighted dense confidence >= CONFIDENCE_THRESHOLD (0.75).
    4. Dense >= 0.75 AND BM25 >= 5.0 (agreement between both engines).
    5. Query contains a legal term AND dense >= 0.78.
    """
    if not sources:
        return False

    if _has_explicit_article(query):
        return True

    best_dense = max((s.get("dense_score") or 0) for s in sources)
    best_bm25 = max((s.get("bm25_score") or 0) for s in sources)

    if mode == "bm25":
        return best_bm25 >= BM25_STRONG_THRESHOLD

    if mode == "dense":
        return best_dense >= DENSE_HIGH_THRESHOLD

    confidence = _compute_confidence(sources)

    if best_dense >= DENSE_HIGH_THRESHOLD:
        return True
    if confidence >= CONFIDENCE_THRESHOLD:
        return True
    if best_dense >= DENSE_MEDIUM_THRESHOLD and best_bm25 >= BM25_STRONG_THRESHOLD:
        return True
    if _has_legal_term(query) and best_dense >= DENSE_LEGAL_TERM_THRESHOLD:
        return True

    return False


class MizanAssistant:
    def __init__(
        self,
        provider: str = "auto",
        api_key: str = None,
        model_name: str = None,
        system_prompt: str = None,
        retriever_mode: str = "hybrid",
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.retriever = LibyanLawRetriever(bm25_enabled=True)
        self.generator = LegalAnswerGenerator(
            provider=provider,
            api_key=api_key,
            model_name=model_name,
            system_prompt=system_prompt,
        )
        self.default_mode = retriever_mode
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self._corpus_df = None

    def _get_corpus_df(self) -> tuple | None:
        """(doc-freq counter, total chunks) — computed once per session."""
        if self._corpus_df is None:
            searcher = getattr(self.retriever, "bm25_searcher", None)
            if searcher is not None:
                self._corpus_df = _compute_corpus_df(searcher)
        return self._corpus_df

    # ─── Citations ─────────────────────────────────────────────────────────────
    def _build_citations(self, sources: list[dict]) -> list[dict]:
        citations = []
        for s in sources:
            citations.append({
                "article": s.get("article"),
                "page": s.get("page"),
                "section": s.get("section"),
                "chapter": s.get("chapter"),
                "document": s.get("document"),
                "doc_id": s.get("doc_id"),
                "doc_type": s.get("doc_type"),
                "year": s.get("year"),
                "score": s.get("score"),
                "rrf_score": s.get("rrf_score"),
                "dense_score": s.get("dense_score"),
                "bm25_score": s.get("bm25_score"),
                "dense_rank": s.get("dense_rank"),
                "bm25_rank": s.get("bm25_rank"),
                "text": (s.get("text") or "")[:600],
            })
        return citations

    # ─── Retrieval only (for streaming UIs) ────────────────────────────────────
    def retrieve_only(
        self,
        query: str,
        top_k: int = 3,
        mode: str = None,
        filters: dict = None,
        chat_history: list[dict] = None,
    ) -> tuple[list[dict], bool, str]:
        """
        Reformulate (if follow-up) then retrieve.
        Returns (sources, is_relevant_and_verified, retrieval_query).
        """
        search_mode = mode or self.default_mode
        retrieval_query = self.generator.contextualize_query(query, chat_history)
        sources = self.retriever.search(
            query=retrieval_query,
            top_k=top_k,
            mode=search_mode,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
            filters=filters,
        )
        relevant = _is_relevant(retrieval_query, sources, search_mode)
        verified = relevant and _passes_topic_match(
            retrieval_query, sources, self._get_corpus_df(),
        )
        return sources, verified, retrieval_query

    # ─── Streaming end-to-end ──────────────────────────────────────────────────
    def ask_stream(self, query: str, top_k: int = 3, mode: str = None,
                   filters: dict = None, chat_history: list[dict] = None) -> dict:
        """
        Prepare the RAG state for a streaming answer.
        Returns a dict:
          {"type": "greeting",  "answer": str}
          {"type": "no_match",  "answer": str, "confidence": 0.0}
          {"type": "answer",    "sources": [...], "citations": [...],
           "confidence": float, "retrieval_query": str,
           "stream": generator-of-tokens}
        After consuming the stream, call finish_stream() to get the cleaned text.
        """
        search_mode = mode or self.default_mode

        if _is_greeting(query):
            return {"type": "greeting", "answer": _format_greeting_response(query)}

        # Scope gate: refuse non-legal questions before any retrieval.
        if not _is_legal_query(query):
            return {
                "type": "no_match",
                "answer": OUT_OF_SCOPE_RESPONSE,
                "confidence": 0.0,
            }

        sources, is_relevant, retrieval_query = self.retrieve_only(
            query=query, top_k=top_k, mode=search_mode,
            filters=filters, chat_history=chat_history,
        )

        if not is_relevant:
            return {
                "type": "no_match",
                "answer": NO_ANSWER_RESPONSE,
                "confidence": 0.0,
            }

        return {
            "type": "answer",
            "sources": sources,
            "citations": self._build_citations(sources),
            "confidence": _compute_confidence(sources),
            "retrieval_query": retrieval_query,
            "stream": self.generator.generate_stream(query, sources, chat_history=chat_history),
        }

    def finish_stream(self, raw_text: str) -> str:
        """Apply anti-repetition cleaning once the stream has been consumed."""
        return self.generator.clean_answer(raw_text)

    # ─── Non-streaming end-to-end ──────────────────────────────────────────────
    def answer_question(
        self,
        query: str,
        top_k: int = 3,
        mode: str = None,
        filters: dict = None,
        chat_history: list[dict] = None,
    ) -> dict:
        """
        End-to-end RAG pipeline (non-streaming):
          1. Greeting / chitchat — bypass RAG entirely.
          2. Reformulate follow-up queries using conversation history (if any).
          3. Retrieve top-k relevant law articles using Hybrid search.
          4. Check relevance scores — return "no information" if below threshold.
          5. Generate cited, grounded legal answer with full conversation context.
        """
        search_mode = mode or self.default_mode

        if _is_greeting(query):
            return {
                "query": query,
                "answer": _format_greeting_response(query),
                "citations": [],
                "provider": self.generator.provider,
                "retriever_mode": search_mode,
                "confidence": 1.0,
                "no_match": False,
            }

        # Scope gate: refuse non-legal questions before any retrieval.
        if not _is_legal_query(query):
            return {
                "query": query,
                "answer": OUT_OF_SCOPE_RESPONSE,
                "citations": [],
                "provider": self.generator.provider,
                "retriever_mode": search_mode,
                "confidence": 0.0,
                "no_match": True,
            }

        retrieval_query = self.generator.contextualize_query(query, chat_history)

        sources = self.retriever.search(
            query=retrieval_query,
            top_k=top_k,
            mode=search_mode,
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
            filters=filters,
        )

        relevant = _is_relevant(retrieval_query, sources, search_mode)
        verified = relevant and _passes_topic_match(
            retrieval_query, sources, self._get_corpus_df(),
        )

        if not verified:
            return {
                "query": query,
                "answer": NO_ANSWER_RESPONSE,
                "citations": [],
                "provider": self.generator.provider,
                "retriever_mode": search_mode,
                "confidence": 0.0,
                "no_match": True,
            }

        answer = self.generator.generate(query, sources, chat_history=chat_history)

        return {
            "query": query,
            "answer": answer,
            "citations": self._build_citations(sources),
            "provider": self.generator.provider,
            "retriever_mode": search_mode,
            "confidence": _compute_confidence(sources),
            "no_match": False,
        }


# Backwards-compatible alias
LibyanLawAssistant = MizanAssistant
