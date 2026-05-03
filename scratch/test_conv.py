import sys
from pathlib import Path
import pymupdf

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import chunker

def test_conversion():
    pdf_path = ROOT / "data" / "raw_data" / "01. 2025년 서울시 청년월세지원 모집 공고문.pdf"
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return
        
    with open(pdf_path, "rb") as f:
        doc = pymupdf.open(stream=f.read(), filetype="pdf")
        md = chunker._fitz_to_markdown(doc)
        
    print(f"Result length: {len(md)}")
    print("--- First 500 chars ---")
    print(md[:500])
    print("--- End ---")

if __name__ == "__main__":
    test_conversion()
