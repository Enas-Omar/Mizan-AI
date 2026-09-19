"""
Mizan AI — Central configuration.
Edit this file to adjust paths, document metadata, or pipeline behaviour.

HOW TO ADD A NEW DOCUMENT
═══════════════════════════════════════════════════════════════════════════════
1. Place the PDF in: data/raw/
2. Add a new entry to DOCUMENTS below (copy the TEMPLATE comment).
3. Run the ingestion pipeline:
       python scripts/run_pipeline.py
   then rebuild the index:
       python scripts/embed_and_index.py
═══════════════════════════════════════════════════════════════════════════════
"""

import os

# ─── Base paths ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

RAW_DIR        = os.path.join(DATA_DIR, "raw")
EXTRACTED_DIR  = os.path.join(DATA_DIR, "extracted")
CLEANED_DIR    = os.path.join(DATA_DIR, "cleaned")
STRUCTURED_DIR = os.path.join(DATA_DIR, "structured")
CHUNKS_DIR     = os.path.join(DATA_DIR, "chunks")
CHROMA_DIR     = os.path.join(DATA_DIR, "chroma_db")

# ─── Document registry (12 documents) ──────────────────────────────────────────
DOCUMENTS = [
    {
        "pdf_file":       "law_12_2010.pdf",
        "doc_id":         "LAW_12_2010",
        "title":          "قانون علاقات العمل رقم 12 لسنة 2010",
        "year":           2010,
        "doc_type":       "law",
        "specialization": "labor",
        "has_text_layer": True,
    },
    {
        "pdf_file":       "decision_888_2023.pdf",
        "doc_id":         "DECISION_888_2023",
        "title":          "قرار مجلس الوزراء رقم 888 لسنة 2023 — اللائحة التنفيذية لقانون علاقات العمل",
        "year":           2023,
        "doc_type":       "decision",
        "specialization": "labor",
        "has_text_layer": False,
        "ocr_rotate":     0,
    },
    {
        "pdf_file":       "القانون المدني.pdf",
        "doc_id":         "CIVIL_CODE",
        "title":          "القانون المدني الليبي",
        "year":           1953,
        "doc_type":       "law",
        "specialization": "civil",
        "has_text_layer": True,
    },
    {
        "pdf_file":       "legal-guide-to-childs-rights-in-libya-arabic.pdf",
        "doc_id":         "CHILD_RIGHTS_GUIDE",
        "title":          "دليل حقوق الطفل في ليبيا",
        "year":           2023,
        "doc_type":       "guide",
        "specialization": "child_rights",
        "has_text_layer": True,
    },
    {
        "pdf_file":       "libyan_law.pdf",
        "doc_id":         "BANKING_LAW_46_2012",
        "title":          "قانون رقم 46 لسنة 2012 بتعديل أحكام قانون المصارف والصيرفة الإسلامية",
        "year":           2012,
        "doc_type":       "law",
        "specialization": "banking",
        "has_text_layer": True,
    },
    {
        "pdf_file":       "income_tax_law_7_2010.pdf",
        "doc_id":         "INCOME_TAX_LAW_7_2010",
        "title":          "قانون رقم 7 لسنة 2010 بشأن ضرائب الدخل",
        "year":           2010,
        "doc_type":       "law",
        "specialization": "tax",
        "has_text_layer": True,
    },
    {
        "pdf_file":       "usury_law_1_2013.pdf",
        "doc_id":         "USURY_LAW_1_2013",
        "title":          "قانون رقم 1 لسنة 2013 بشأن منع المعاملات الربوية",
        "year":           2013,
        "doc_type":       "law",
        "specialization": "banking",
        "has_text_layer": False,
        "ocr_rotate":     0,
    },
    {
        "pdf_file":       "anti_money_laundering_2_2005.pdf",
        "doc_id":         "ANTI_MONEY_LAUNDERING_2_2005",
        "title":          "قانون رقم 2 لسنة 2005 بشأن مكافحة غسل الأموال",
        "year":           2005,
        "doc_type":       "law",
        "specialization": "financial_crime",
        "has_text_layer": False,
        "ocr_rotate":     0,
    },
    {
        "pdf_file":       "state_financial_system_law.pdf",
        "doc_id":         "STATE_FINANCIAL_SYSTEM_LAW",
        "title":          "قانون النظام المالي للدولة",
        "year":           1967,
        "doc_type":       "law",
        "specialization": "public_finance",
        "has_text_layer": False,
        "ocr_rotate":     0,
    },
    {
        "pdf_file":       "income_tax_regulation_592_2010.pdf",
        "doc_id":         "INCOME_TAX_REGULATION_592_2010",
        "title":          "قرار رقم 592 لسنة 2010 بشأن اللائحة التنفيذية لقانون ضرائب الدخل",
        "year":           2010,
        "doc_type":       "regulation",
        "specialization": "tax",
        "has_text_layer": False,
        "ocr_rotate":     270,
    },
    {
        "pdf_file":       "stock_market_bylaws_2006.pdf",
        "doc_id":         "STOCK_MARKET_BYLAWS_2006",
        "title":          "النظام الأساسي لسوق الأوراق المالية الليبي",
        "year":           2006,
        "doc_type":       "regulation",
        "specialization": "securities",
        "has_text_layer": False,
        "ocr_rotate":     0,
    },
    {
        "pdf_file":       "commercial_activity_law_23_2010.pdf",
        "doc_id":         "COMMERCIAL_ACTIVITY_LAW_23_2010",
        "title":          "قانون رقم 23 لسنة 2010 بشأن النشاط التجاري",
        "year":           2010,
        "doc_type":       "law",
        "specialization": "commercial",
        "has_text_layer": False,
        "ocr_rotate":     0,
    },
]

# TEMPLATE — copy and fill to add a new document:
# {
#     "pdf_file":       "law_15_1980.pdf",
#     "doc_id":         "LAW_15_1980",
#     "title":          "قانون التقاعد رقم 15 لسنة 1980",
#     "year":           1980,
#     "doc_type":       "law",
#     "specialization": "pension",
#     "has_text_layer": True,
# },

# ─── OCR settings (used when has_text_layer is False) ──────────────────────────
# scripts/step2_extract.py performs OCR with Windows Media OCR (winocr package,
# Windows 10/11 only) — no external binary required. Language: Arabic (ar-SA),
# rendering DPI: 150 (set inside the script).

# ─── Arabic text normalisation ─────────────────────────────────────────────────
ARABIC_DIACRITICS = (
    "\u064B",  # tanwin fath
    "\u064C",  # tanwin damm
    "\u064D",  # tanwin kasr
    "\u064E",  # fatha
    "\u064F",  # damma
    "\u0650",  # kasra
    "\u0651",  # shadda
    "\u0652",  # sukun
    "\u0653",  # maddah above
    "\u0654",  # hamza above
    "\u0655",  # hamza below
    "\u0670",  # superscript alef
)

# ─── RAG retrieval defaults ────────────────────────────────────────────────────
DEFAULT_TOP_K          = 3
DEFAULT_RETRIEVAL_MODE = "hybrid"   # 'hybrid' | 'dense' | 'bm25'
DEFAULT_DENSE_WEIGHT   = 1.0
DEFAULT_BM25_WEIGHT    = 1.0
RRF_K                  = 60

# ─── ChromaDB collection name ──────────────────────────────────────────────────
CHROMA_COLLECTION = "libyan_labor_law"   # kept for compatibility with the shipped index
