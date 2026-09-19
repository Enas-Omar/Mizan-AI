"""
Step 4 — Structure the legal documents into clean JSON records.

Parses the cleaned text to extract the hierarchical structure:
  باب (Section) → فصل (Chapter) → مادة (Article) → نص المادة
  Or for guides:
  قسم (Section) → موضوع (Topic) → نص المحتوى

Each article/section becomes a structured JSON record:
{
  "document_id": "LAW_12_2010",
  "title": "قانون علاقات العمل رقم 12 لسنة 2010",
  "year": 2010,
  "doc_type": "law",
  "section": "الباب الثالث",
  "chapter": "الفصل الثاني",
  "article": "25",
  "article_label": "مادة (25)",
  "page": 18,
  "text": "..."
}

Input:  data/cleaned/<doc_id>_clean.txt
Output: data/structured/<doc_id>_structured.json
"""

import os
import re
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOCUMENTS, CLEANED_DIR, STRUCTURED_DIR

os.makedirs(STRUCTURED_DIR, exist_ok=True)

PAGE_MARKER = re.compile(r"^\[PAGE\s+(\d+)\]$")

ORDINAL_MAP = {
    "الأولى": "1", "الأول": "1", "األولى": "1", "االولى": "1",
    "الثانية": "2", "الثاني": "2",
    "الثالثة": "3", "الثالث": "3",
    "الرابعة": "4", "الرابع": "4",
    "الخامسة": "5", "الخامس": "5",
    "السادسة": "6", "السادس": "6",
    "السابعة": "7", "السابع": "7",
    "الثامنة": "8", "الثامن": "8",
    "التاسعة": "9", "التاسع": "9",
    "العاشرة": "10", "العاشر": "10",
    "الحادية عشرة": "11", "الحادي عشر": "11",
    "الثانية عشرة": "12", "الثاني عشر": "12",
    "الثالثة عشرة": "13", "الثالث عشر": "13",
    "الرابعة عشرة": "14", "الرابع عشر": "14",
    "الخامسة عشرة": "15", "الخامس عشر": "15",
    "السادسة عشرة": "16", "السادس عشر": "16",
    "السابعة عشرة": "17", "السابع عشر": "17",
    "الثامنة عشرة": "18", "الثامن عشر": "18",
    "التاسعة عشرة": "19", "التاسع عشر": "19",
    "العشرون": "20",
    "الحادية والعشرون": "21",
    "الثانية والعشرون": "22",
    "الثالثة والعشرون": "23",
    "الرابعة والعشرون": "24",
    "الخامسة والعشرون": "25",
    "السادسة والعشرون": "26",
    "السابعة والعشرون": "27",
    "الثامنة والعشرون": "28",
    "التاسعة والعشرون": "29",
    "الثلاثون": "30",
    "الحادية والثلاثون": "31",
    "الثانية والثلاثون": "32",
    "الثالثة والثلاثون": "33",
    "الرابعة والثلاثون": "34",
    "الخامسة والثلاثون": "35",
    "السادسة والثلاثون": "36",
    "السابعة والثلاثون": "37",
    "الثامنة والثلاثون": "38",
    "التاسعة والثلاثون": "39",
    "الأربعون": "40",
    "الخامسة والأربعون": "45",
    "السادسة والأربعون": "46",
    "الخامسة والخمسون": "55",
    "السادسة والخمسون": "56",
}


def normalize_article_number(raw: str) -> str:
    raw = raw.strip()
    if raw in ORDINAL_MAP:
        return ORDINAL_MAP[raw]
    for ord_text, num_val in sorted(ORDINAL_MAP.items(), key=lambda x: -len(x[0])):
        if ord_text in raw:
            return num_val
    # Convert Eastern Arabic digits ٠-٩ to ASCII 0-9
    out = ""
    for ch in raw:
        if "\u0660" <= ch <= "\u0669":
            out += chr(ord(ch) - 0x0660 + ord("0"))
        elif ch.isdigit():
            out += ch
    return out or raw


def preprocess_text(text: str) -> str:
    """Normalize multi-line broken headings before line-by-line parsing."""
    # 0. Civil Code format: ") مادة1 ( title" -> "مادة (1)"
    #    The closing paren comes BEFORE مادة in this PDF encoding.
    text = re.sub(
        r"(?:^|\n)\s*\)\s*(مادة|المادة)\s*([٠-٩\d]+)\s*\(",
        r"\nمادة (\2)",
        text
    )
    # Also handle ") مادةN" with no trailing paren
    text = re.sub(
        r"(?:^|\n)\s*\)\s*(مادة|المادة)\s*([٠-٩\d]+)",
        r"\nمادة (\2)",
        text
    )

    # 1. Normalize line-broken Abwab / Kutub: "الباب\nالثالث" -> "الباب الثالث"
    text = re.sub(r"(?:^|\n)\s*(الباب|باب|الكتاب|كتاب|الجزء|جزء)\s*\n\s*([^\n]+)", r"\n\1 \2", text)

    # 2. Normalize line-broken Fusoul: "الفصل\nالأول" -> "الفصل الأول"
    text = re.sub(r"(?:^|\n)\s*(الفصل|فصل|الفرع|فرع)\s*\n\s*([^\n]+)", r"\n\1 \2", text)

    # 3. Normalize line-broken Articles:
    # "مادة\n)1\n(" -> "مادة (1)"
    text = re.sub(
        r"(?:^|\n)\s*(مادة|المادة)\s*\n\s*[\(\)]?\s*([٠-٩\d]+|الأولى|الأول|الثانية|الثالثة|الرابعة|الخامسة)\s*[\(\)]?",
        r"\n\1 (\2)",
        text
    )

    # 4. Standardize inline article patterns:
    # ") مادة 12 (" -> "مادة (12)"
    text = re.sub(
        r"(?:^|\n)\s*[\(\[]?\s*(مادة|المادة)\s*([٠-٩\d]+)\s*[\)\]]?",
        r"\n\1 (\2)",
        text
    )

    # 5. Standardize ordinals: "المادة الأولى:" -> "المادة (الأولى)"
    ordinals_pattern = "|".join(re.escape(k) for k in ORDINAL_MAP.keys())
    text = re.sub(
        rf"(?:^|\n)\s*(مادة|المادة)\s*[\(\[]?\s*({ordinals_pattern})[ :]*[\]\)]?",
        r"\n\1 (\2)",
        text
    )

    return text


def parse_guide_document(clean_text: str, doc_cfg: dict) -> list[dict]:
    """Structures guide-type documents (e.g. Child Rights Guide) by sections/topics."""
    lines = clean_text.split("\n")
    records = []
    current_page = 1
    current_section = "مقدمة الدليل"
    current_topic = None
    current_sec_id = "1"
    current_lines = []

    re_heading = re.compile(r"^(?:(\d+(?:\.\d+)*)\s*[-.]?\s*)?([^\n]{5,100})$")

    def flush_section():
        nonlocal current_sec_id, current_section, current_topic, current_lines
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if not body or len(body) < 30:
            return
        records.append({
            "document_id": doc_cfg["doc_id"],
            "title": doc_cfg["title"],
            "year": doc_cfg["year"],
            "doc_type": doc_cfg["doc_type"],
            "section": current_section,
            "chapter": current_topic or current_section,
            "article": current_sec_id,
            "article_label": f"قسم ({current_sec_id}): {current_topic or current_section}",
            "page": current_page,
            "text": body,
        })

    for line in lines:
        stripped = line.strip()
        m_page = PAGE_MARKER.match(stripped)
        if m_page:
            current_page = int(m_page.group(1))
            continue

        # Check for numbered section heading e.g. "2.1.8 الأطفال ذوي الاحتياجات الخاصة"
        m_head = re.match(r"^(\d+(?:\.\d+)*)\s*[-.]?\s*([^\n]{4,80})$", stripped)
        if m_head and len(stripped) < 90:
            sec_num = m_head.group(1)
            sec_title = m_head.group(2).strip()
            flush_section()
            current_sec_id = sec_num
            current_topic = sec_title
            current_section = f"الباب {sec_num.split('.')[0]}" if '.' in sec_num else f"القسم {sec_num}"
            current_lines = [stripped]
            continue

        if stripped:
            current_lines.append(stripped)

    flush_section()
    return records


def parse_document(clean_text: str, doc_cfg: dict) -> list[dict]:
    if doc_cfg.get("doc_type") == "guide":
        return parse_guide_document(clean_text, doc_cfg)

    clean_text = preprocess_text(clean_text)
    lines = clean_text.split("\n")

    records = []
    current_page = 1
    current_baab = None
    current_fasl = None
    current_art = None
    current_label = None
    current_lines = []

    # Patterns to detect headings on single lines
    re_art = re.compile(r"^(?:مادة|المادة)\s*\(\s*([^\)]+)\s*\)")
    re_baab = re.compile(r"^(?:باب|الباب|كتاب|الكتاب|جزء|الجزء)\s+(.+)$")
    re_fasl = re.compile(r"^(?:فصل|الفصل|فرع|الفرع)\s+(.+)$")

    def flush_article():
        nonlocal current_art, current_label, current_lines
        if current_art is None:
            return
        body = "\n".join(current_lines).strip()
        if not body:
            return
        records.append({
            "document_id": doc_cfg["doc_id"],
            "title": doc_cfg["title"],
            "year": doc_cfg["year"],
            "doc_type": doc_cfg["doc_type"],
            "section": current_baab or "أحكام عامة ومواد تمهيدية",
            "chapter": current_fasl,
            "article": current_art,
            "article_label": current_label,
            "page": current_page,
            "text": body,
        })

    for line in lines:
        stripped = line.strip()

        # Track page numbers
        m_page = PAGE_MARKER.match(stripped)
        if m_page:
            current_page = int(m_page.group(1))
            continue

        # Detect الباب / الكتاب / الجزء
        m_baab = re_baab.match(stripped)
        if m_baab and len(stripped) < 120 and "مادة" not in stripped:
            flush_article()
            current_art, current_label, current_lines = None, None, []
            current_baab = stripped
            current_fasl = None
            continue

        # Detect الفصل / فصل
        m_fasl = re_fasl.match(stripped)
        if m_fasl and len(stripped) < 120 and "مادة" not in stripped:
            flush_article()
            current_art, current_label, current_lines = None, None, []
            current_fasl = stripped
            continue

        # Detect مادة
        m_art = re_art.match(stripped)
        if m_art:
            flush_article()
            raw_num = m_art.group(1).strip()
            norm_num = normalize_article_number(raw_num)
            if current_baab is None and doc_cfg["doc_id"] == "LAW_12_2010":
                current_art = f"إصدار_{norm_num}"
                current_label = f"مادة إصدار ({norm_num})"
            else:
                current_art = norm_num
                current_label = f"مادة ({norm_num})"
            current_lines = [stripped]
            continue

        # Accumulate article content
        if current_art is not None:
            current_lines.append(line)

    flush_article()
    return records


def structure_document(doc_cfg: dict) -> None:
    clean_path = os.path.join(CLEANED_DIR,    f"{doc_cfg['doc_id']}_clean.txt")
    out_path   = os.path.join(STRUCTURED_DIR, f"{doc_cfg['doc_id']}_structured.json")

    if not os.path.exists(clean_path):
        print(f"  [MISSING] Clean file not found (run step3 first): {clean_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Document : {doc_cfg['title']}")

    with open(clean_path, "r", encoding="utf-8") as f:
        clean_text = f.read()

    records = parse_document(clean_text, doc_cfg)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    sections = set(r["section"] for r in records if r["section"])
    chapters = set(r["chapter"] for r in records if r["chapter"])
    print(f"  ✓ Records/Articles parsed: {len(records)}")
    print(f"  ✓ Sections (أبواب/أقسام) : {len(sections)}")
    print(f"  ✓ Chapters (فصول/مواضيع) : {len(chapters)}")
    print(f"  ✓ Saved                  : {out_path}")

    if records:
        print(f"  First item: {records[0]['article_label']} on page {records[0]['page']}")
        print(f"  Last item : {records[-1]['article_label']} on page {records[-1]['page']}")


def main():
    print("\n🏗️   STEP 4 — DOCUMENT STRUCTURING")
    for doc in DOCUMENTS:
        structure_document(doc)
    print("\n✅  Structuring completed. Check data/structured/")


if __name__ == "__main__":
    main()
