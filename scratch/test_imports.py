import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_imports():
    print("Testing imports...")
    try:
        import pymupdf
        print("import pymupdf: SUCCESS")
    except Exception as e:
        print(f"import pymupdf: FAILED ({e})")
        
    try:
        import fitz
        print("import fitz: SUCCESS")
    except Exception as e:
        print(f"import fitz: FAILED ({e})")
        
    try:
        import pymupdf4llm
        print("import pymupdf4llm: SUCCESS")
    except Exception as e:
        print(f"import pymupdf4llm: FAILED ({e})")

if __name__ == "__main__":
    test_imports()
