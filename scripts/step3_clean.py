"""
Step 3 — Clean raw extracted text.

Cleans:
  ✓ Removes font artifact kashida (\u0467)
  ✓ Normalizes letter forms (e.g. Urdu Heh \u06BE -> Arabic Heh \u0647)
  ✓ Removes repeated DCAF header/footer and disclaimer text
  ✓ Removes page numbering noise
  ✓ Collapses multiple blank lines and redundant whitespace
  ✗ Preserves all legal anchors: مادة، الباب، باب، الفصل، الشروط، الاستثناءات

Input:  data/extracted/<doc_id>_raw.txt
Output: data/cleaned/<doc_id>_clean.txt
"""

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOCUMENTS, EXTRACTED_DIR, CLEANED_DIR, ARABIC_DIACRITICS

os.makedirs(CLEANED_DIR, exist_ok=True)

# Protected legal keywords
LEGAL_ANCHORS = re.compile(
    r"(مادة|المادة|الباب|باب|الفصل|فصل|الفقرة|فقرة|القانون|اللائحة|الشروط|الاستثناء|"
    r"يُعدّ|يُعتبر|يُحدد|يُلزم|يُلغى|يُضاف|يُستثنى|يجوز|لا يجوز|يعاقب|"
    r"بموجب|استناداً|استثناء|وفقاً|رقم\s+\d|لسنة\s+\d)"
)

# Common foreign disclaimer patterns (e.g. DCAF project footer)
DISCLAIMER_PATTERNS = [
    re.compile(r"This document constitutes an un-official", re.IGNORECASE),
    re.compile(r"DCAF cannot be held responsible", re.IGNORECASE),
    re.compile(r"DCAF’s Libyan Security Sector", re.IGNORECASE),
    re.compile(r"DCAF's Libyan Security Sector", re.IGNORECASE),
    re.compile(r"www\.security-legislation\.ly", re.IGNORECASE),
    re.compile(r"arise from its use", re.IGNORECASE),
    re.compile(r"For official reference, please refer", re.IGNORECASE),
    re.compile(r"financed by the DCAF Trust Fund", re.IGNORECASE),
    re.compile(r"Page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"\d+\s+of\s+\d+\s*Page", re.IGNORECASE),
]


import unicodedata

def fix_arabic_ligatures(text: str) -> str:
    """
    Fixes inverted Lam-Alef ligatures from PDF extractions:
      - 'األ' -> 'الأ' (e.g. 'األساسية' -> 'الأساسية')
      - 'اإل' -> 'الإ' (e.g. 'اإلعالن' -> 'الإعلان')
      - 'اآل' -> 'الآ' (e.g. 'اآلخرين' -> 'الآخرين')
      - 'اال' at start -> 'الا' (e.g. 'االطالع' -> 'الاطلاع', 'االتحاد' -> 'الاتحاد')
      - Standalone 'ال' -> 'لا' (e.g. 'ال يجوز' -> 'لا يجوز')
      - Reversed internal ligatures ('عالقات' -> 'علاقات', 'خالل' -> 'خلال')
    """
    # 1. Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)

    # 2. Tatweel and artifact font symbols
    text = text.replace("\u0467", "")
    text = text.replace("ـ", "")

    # 3. Non-standard Persian / Urdu glyphs
    text = text.replace("\u06CC", "\u064A")  # Farsi Yeh -> Arabic Yeh
    text = text.replace("\u06D2", "\u064A")  # Urdu Yeh Barree -> Arabic Yeh
    text = text.replace("\u06A9", "\u0643")  # Farsi Keheh -> Arabic Kaf
    text = text.replace("\u06BE", "\u0647")  # Urdu Heh -> Arabic Heh
    text = text.replace("\u06C1", "\u0647")  # Goal Heh -> Arabic Heh
    text = text.replace("\u06C0", "\u0647")  # Heh with Yeh above -> Heh

    # 4. Lam-Alef with Definite Article corrections
    text = re.sub(r'األ', 'الأ', text)
    text = re.sub(r'اإل', 'الإ', text)
    text = re.sub(r'اآل', 'الآ', text)
    text = re.sub(r'\b([وفبكل]?)اال', r'\1الا', text)

    # 5. Words starting with الال -> اللا (e.g. الالئحة -> اللائحة, الالزمة -> اللازمة)
    text = re.sub(r'\b([وفبكل]?)الال', r'\1اللا', text)

    # 6. Standalone particles and negatives
    text = re.sub(r'\bال\b', 'لا', text)
    text = re.sub(r'\bوال\b', 'ولا', text)
    text = re.sub(r'\bفال\b', 'فلا', text)
    text = re.sub(r'\bبال\b', 'بلا', text)
    text = re.sub(r'\bإال\b', 'إلا', text)
    text = re.sub(r'\bوإال\b', 'وإلا', text)
    text = re.sub(r'\bفإال\b', 'فإلا', text)
    text = re.sub(r'\bأال\b', 'ألا', text)
    text = re.sub(r'\bوأال\b', 'وألا', text)

    # 7. Common prepositions starting with لـ reversed into أل / إل
    repositions_map = {
        r'\bألحكام\b': 'لأحكام',
        r'\bألول\b': 'لأول',
        r'\bألي\b': 'لأي',
        r'\bألكثر\b': 'لأكثر',
        r'\bألقل\b': 'لأقل',
        r'\bألغراض\b': 'لأغراض',
        r'\bألجل\b': 'لأجل',
        r'\bألداء\b': 'لأداء',
        r'\bألسباب\b': 'لأسباب',
        r'\bإلجراء\b': 'لإجراء',
        r'\bإلحالة\b': 'لإحالة',
        r'\bإلدارة\b': 'لإدارة',
        r'\bإلثبات\b': 'لإثبات',
        r'\bإلنجاز\b': 'لإنجاز',
        r'\bإلصدار\b': 'لإصدار',
        r'\bإلعداد\b': 'لإعداد',
        r'\bإلتمام\b': 'لإتمام',
    }
    for pat, repl in repositions_map.items():
        text = re.sub(pat, repl, text)

    # 8. Frequent Libyan legal vocabulary with internal reversed Lam-Alef
    vocab_fixes = {
        r'عالقات': 'علاقات',
        r'إعالن': 'إعلان',
        r'اعالن': 'اعلان',
        r'([إا])طالع': r'\1طلاع',
        r'\bخالل\b': 'خلال',
        r'\bوخالل\b': 'وخلال',
        r'\bفخالل\b': 'فخلال',
        r'\bبخالل\b': 'بخلال',
        r'إخالل': 'إخلال',
        r'اخالل': 'اخلال',
        r'\bثالث\b': 'ثلاث',
        r'\bثالثة\b': 'ثلاثة',
        r'\bوثالث\b': 'وثلاث',
        r'\bوثالثة\b': 'وثلاثة',
        r'\bالثالث\b': 'الثلاث',
        r'\bالثالثة\b': 'الثلاثة',
        r'\bثالثين\b': 'ثلاثين',
        r'\bوثالثين\b': 'وثلاثين',
        r'\bوالثالثين\b': 'والثالثين',
        r'\bالثالثين\b': 'الثلاثين',
        r'\bثالثون\b': 'ثلاثون',
        r'\bوثالثون\b': 'وثلاثون',
        r'\bوالثالثون\b': 'والثالثون',
        r'\bالثالثون\b': 'الثلاثون',
        r'حاالت': 'حالات',
        r'الحاالت': 'الحالات',
        r'صالحية': 'صلاحية',
        r'صالحيات': 'صلاحيات',
        r'عالوة': 'علاوة',
        r'عالوات': 'علاوات',
        r'بطالن': 'بطلان',
        r'استقالل': 'استقلال',
        r'تعديالته': 'تعديلاته',
        r'تعديالت': 'تعديلات',
        r'وكالء': 'وكلاء',
        r'عمالء': 'عملاء',
        r'زمالء': 'زملاء',
        r'\bخالف\b': 'خلاف',
        r'\bخالفات\b': 'خلافات',
        r'\bالخالف\b': 'الخلاف',
        r'\bالخالفات\b': 'الخلافات',
        r'إطالق': 'إطلاق',
        r'اطالق': 'اطلاق',
        r'إبالغ': 'إبلاغ',
        r'ابالغ': 'ابلاغ',
        r'إحالل': 'إحلال',
        r'احالل': 'احلال',
        r'استطالع': 'استطلاع',
    }
    for pat, repl in vocab_fixes.items():
        text = re.sub(pat, repl, text)

    return text


def clean_text(raw_text: str) -> str:
    # 1. Apply full Arabic ligature and character corrections
    text = fix_arabic_ligatures(raw_text)

    # 2. Strip diacritics / harakat
    for d in ARABIC_DIACRITICS:
        text = text.replace(d, "")


    lines = text.split("\n")
    cleaned_lines = []
    prev_line = ""

    for line in lines:
        stripped = line.strip()

        # Check for [PAGE N] marker - always keep it
        if re.match(r"^\[PAGE\s+\d+\]$", stripped):
            cleaned_lines.append(stripped)
            prev_line = stripped
            continue

        # Skip disclaimer lines
        if any(p.search(stripped) for p in DISCLAIMER_PATTERNS):
            continue

        # Skip bare page number lines
        if re.fullmatch(r"\d+", stripped) and len(stripped) <= 3:
            continue

        # Skip empty lines if previous line was also empty
        if not stripped and not prev_line:
            continue

        # Skip visual noise lines (e.g. ------ or =====)
        if re.fullmatch(r"[-=_~*#|]{3,}", stripped):
            continue

        # Collapse multiple spaces
        norm_line = re.sub(r"[ \t]+", " ", line).strip()

        cleaned_lines.append(norm_line)
        prev_line = norm_line

    result = "\n".join(cleaned_lines)
    # Rejoin words broken across line wraps (e.g. "الفتر\nة " -> "الفترة ")
    result = re.sub(r'([\u0600-\u06FF]{2,})\n([\u0600-\u06FF]{1,3})(?=[ \t.,:;()!؟])', r'\1\2 ', result)
    # Collapse 3+ newlines to 2
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()



def clean_document(doc_cfg: dict) -> None:
    raw_path   = os.path.join(EXTRACTED_DIR, f"{doc_cfg['doc_id']}_raw.txt")
    clean_path = os.path.join(CLEANED_DIR,   f"{doc_cfg['doc_id']}_clean.txt")

    if not os.path.exists(raw_path):
        print(f"  [MISSING] Raw file not found (run step2 first): {raw_path}")
        return

    print(f"\n{'='*60}")
    print(f"  Document : {doc_cfg['title']}")

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned = clean_text(raw_text)

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    raw_lines   = raw_text.count("\n")
    clean_lines = cleaned.count("\n")
    print(f"  ✓ Raw    : {len(raw_text):,} chars | {raw_lines:,} lines")
    print(f"  ✓ Clean  : {len(cleaned):,} chars | {clean_lines:,} lines")
    print(f"  ✓ Saved  : {clean_path}")


def main():
    print("\n🧹  STEP 3 — TEXT CLEANING")
    for doc in DOCUMENTS:
        clean_document(doc)
    print("\n✅  Cleaning step completed. Check data/cleaned/")


if __name__ == "__main__":
    main()
