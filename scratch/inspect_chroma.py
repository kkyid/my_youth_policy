
import os
import sys
from pathlib import Path
import chromadb
from chromadb.config import Settings

# Add current dir to sys.path if needed
sys.path.append(str(Path.cwd()))

DATA_DIR = Path("data")
HOUSING_DB_DIR = DATA_DIR / "db" / "chroma_db_v2"
HOUSING_COLLECTION = "youth_housing_policy"

def inspect_db():
    print(f"Inspecting ChromaDB at: {HOUSING_DB_DIR}")
    
    if not HOUSING_DB_DIR.exists():
        print("Directory does not exist.")
        return

    # List files in the directory
    print("\nFiles in DB directory:")
    for f in HOUSING_DB_DIR.iterdir():
        print(f" - {f.name}")

    try:
        client = chromadb.PersistentClient(
            path=str(HOUSING_DB_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        
        collection_names = client.list_collections()
        print(f"\nAvailable collections: {collection_names}")
        
        for name in collection_names:
            coll_obj = client.get_collection(name)
            count = coll_obj.count()
            print(f"\n--- Collection: {name} (Count: {count}) ---")
            
            if count > 0:
                results = coll_obj.get(limit=1, include=["metadatas", "documents"])
                if results["ids"]:
                    print(f"Sample ID: {results['ids'][0]}")
                    print(f"Sample Metadata: {results['metadatas'][0]}")
                    # Try to handle missing documents if they are just embeddings
                    if results.get("documents") and results["documents"][0]:
                        print(f"Sample Document snippet: {results['documents'][0][:200]}...")
                    else:
                        print("No document text found (maybe only embeddings stored).")
                else:
                    print("No data returned from get().")
            else:
                print("Collection is empty.")
                
    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_db()
