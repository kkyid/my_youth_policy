import sys
from pathlib import Path
import chromadb

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import vector_db

SOURCE_NAME = "일반대출 안내 _ 내집마련디딤돌대출 _ 주택구입자금대출 _ 개인상품 _ 주택도시기금.pdf"

def main():
    for col_name in [vector_db.HOUSING_COLLECTION, vector_db.FINANCE_COLLECTION]:
        vs = vector_db.get_vectorstore(col_name)
        col = vs._collection
        
        # Check if exists
        res = col.get(where={"source": SOURCE_NAME})
        ids = res.get("ids", [])
        
        if ids:
            print(f"Found {len(ids)} chunks from '{SOURCE_NAME}' in '{col_name}'. Deleting...")
            col.delete(ids=ids)
            print("Deleted.")
        else:
            print(f"No chunks found from '{SOURCE_NAME}' in '{col_name}'.")

if __name__ == "__main__":
    main()
