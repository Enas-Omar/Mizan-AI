"""
Step 2 — Extract raw text from PDFs.

Uses PyMuPDF (fitz) for native digital text extraction with page markers.
Falls back gracefully to Tesseract OCR if a document is scanned.

Output: data/extracted/<doc_id>_raw.txt
"""

import os
import sys
import pymupdf as fitz
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DOCUMENTS, RAW_DIR, EXTRACTED_DIR

os.makedirs(EXTRACTED_DIR, exist_ok=True)


import unicodedata
import re

def unreverse_lawsociety_line(line: str) -> str:
    """Un-reverses right-to-left word ordering from lawsociety.ly PDFs."""
    if 'LAWSOCIETY.LY' in line or 'صفحة' in line or not line.strip():
        return ''
    parts = re.split(r'([،,.:])', line)
    new_parts = []
    for part in parts:
        if part in ['،', ',', '.', ':']:
            new_parts.append(part)
        else:
            words = part.split()
            words.reverse()
            new_parts.append(' '.join(words))
    reconstructed = ''.join(new_parts)
    swaps = {
        r'\bيف\b': 'في',
        r'\bإىل\b': 'إلى',
        r'\bالل يب\b': 'الليبي',
        r'\bاليت\b': 'التي',
        r'مرصف': 'مصرف',
        r'المرصف': 'المصرف',
        r'الرصافة': 'الصرافة',
        r'صريفة': 'صيرفة',
        r'النرش': 'النشر',
        r'الوطين': 'الوطني',
        r'النتقايل': 'الانتقالي',
        r'األجنيب': 'الأجنبي',
        r'أجنيب': 'أجنبي',
        r'ورشكات': 'وشركات',
        r'رشكات': 'شركات',
        r'التأجري': 'التأجيري',
        r'واإلرشاف': 'والإشراف',
        r'المساهمني': 'المساهمين',
        r'المودعني': 'المودعين',
        r'عرشة': 'عشرة',
        r'التأخري': 'التأخير',
    }
    for pat, repl in swaps.items():
        reconstructed = re.sub(pat, repl, reconstructed)
    return reconstructed


def extract_with_pymupdf(pdf_path: str) -> str:
    """
    Extract digital text directly from PDF with NFKC normalization
    and page boundaries [PAGE N] for downstream metadata tagging.
    """
    doc = fitz.open(pdf_path)
    pages_text = []
    for page_num, page in enumerate(tqdm(doc, desc="Extracting pages", unit="page"), start=1):
        raw = page.get_text("text")
        # Unicode normalization for Arabic presentation forms (e.g. Civil Code)
        text = unicodedata.normalize("NFKC", raw)
        
        # If document is from lawsociety.ly with inverted words
        if "LAWSOCIETY.LY" in text or "المرصف" in text:
            fixed_lines = [unreverse_lawsociety_line(l) for l in text.split("\n")]
            text = "\n".join([l for l in fixed_lines if l.strip()])
            
        pages_text.append(f"[PAGE {page_num}]\n{text.strip()}")
    doc.close()
    return "\n\n".join(pages_text)



def extract_with_winocr(pdf_path: str, rotate: int = 0) -> str:
    """
    Extract Arabic text from scanned or un-copyable PDFs using native
    Windows Media OCR (Windows.Media.Ocr ar-SA) with Right-to-Left (RTL) word sorting.
    """
    import winocr
    from PIL import Image

    doc = fitz.open(pdf_path)
    pages_text = []

    for page_num, page in enumerate(tqdm(doc, desc="  OCR Scanning", unit="page"), start=1):
        try:
            pix = page.get_pixmap(dpi=150)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            if rotate:
                img = img.rotate(rotate, expand=True)

            res = winocr.recognize_pil_sync(img, lang="ar-SA")
            lines_text = []
            for line in res.get("lines", []):
                words = line.get("words", [])
                if not words:
                    continue
                # Sort words from Right to Left (x descending) for natural Arabic reading order
                sorted_words = sorted(words, key=lambda w: w["bounding_rect"]["x"], reverse=True)
                lines_text.append(" ".join(w["text"] for w in sorted_words))

            page_content = "\n".join(lines_text)
            pages_text.append(f"[PAGE {page_num}]\n{page_content.strip()}")
        except Exception as e:
            print(f"    [WARN] Page {page_num} OCR failed: {e}")
            pages_text.append(f"[PAGE {page_num}]\n")

    doc.close()
    return "\n\n".join(pages_text)


def extract_document(doc_cfg: dict) -> None:
    pdf_path = os.path.join(RAW_DIR, doc_cfg["pdf_file"])
    out_path = os.path.join(EXTRACTED_DIR, f"{doc_cfg['doc_id']}_raw.txt")

    if not os.path.exists(pdf_path):
        print(f"  [MISSING] PDF not found: {pdf_path}")
        return

    # Skip if already extracted and non-empty
    if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
        print(f"  [EXISTS] {doc_cfg['doc_id']} already extracted ({os.path.getsize(out_path):,} bytes). Skipping.")
        return

    strategy = "Direct PyMuPDF extraction" if doc_cfg["has_text_layer"] else f"Windows Media OCR (ar-SA, rot={doc_cfg.get('ocr_rotate', 0)}°)"

    print(f"\n{'='*60}")
    print(f"  Document : {doc_cfg['title']}")
    print(f"  PDF      : {doc_cfg['pdf_file']}")
    print(f"  Strategy : {strategy}")
    print(f"{'='*60}")

    if doc_cfg["has_text_layer"]:
        raw_text = extract_with_pymupdf(pdf_path)
    else:
        raw_text = extract_with_winocr(pdf_path, rotate=doc_cfg.get("ocr_rotate", 0))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    char_count = len(raw_text)
    line_count = raw_text.count("\n")
    print(f"  ✓ Saved  : {out_path}")
    print(f"  ✓ Stats  : {char_count:,} chars | {line_count:,} lines")


def main():
    print("\n🚀  STEP 2 — TEXT EXTRACTION")
    for doc in DOCUMENTS:
        extract_document(doc)
    print("\n✅  Extraction step completed. Check data/extracted/")


if __name__ == "__main__":
    main()
