import sys
from pathlib import Path
import pymupdf
import pymupdf4llm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import chunker

def test_pymupdf4llm():
    pdf_path = ROOT / "data" / "raw_data" / "01. 2025년 서울시 청년월세지원 모집 공고문.pdf"
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}")
        return
        
    print("Testing pymupdf4llm...")
    try:
        md = pymupdf4llm.to_markdown(str(pdf_path))
        print(f"Result length: {len(md)}")
        print("--- First 500 chars ---")
        print(md[:500])
        print("--- End ---")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_pymupdf4llm()
