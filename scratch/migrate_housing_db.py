
import chromadb
from chromadb.config import Settings
from pathlib import Path

DATA_DIR = Path("data")
DB_DIR = DATA_DIR / "db" / "chroma_db_v2"
OLD_COLLECTION = "exp09_char_800_120"
NEW_COLLECTION = "youth_housing_policy"

def migrate():
    print(f"Starting migration in {DB_DIR}...")
    
    client = chromadb.PersistentClient(
        path=str(DB_DIR),
        settings=Settings(anonymized_telemetry=False)
    )
    
    # 1. Get old collection
    try:
        old_coll = client.get_collection(OLD_COLLECTION)
        print(f"Found old collection '{OLD_COLLECTION}' with {old_coll.count()} items.")
    except Exception as e:
        print(f"Error: Could not find old collection '{OLD_COLLECTION}': {e}")
        return

    # 2. Get all data
    # include embeddings to avoid re-embedding costs
    print("Fetching data from old collection...")
    all_data = old_coll.get(include=["documents", "metadatas", "embeddings"])
    
    ids = all_data["ids"]
    docs = all_data["documents"]
    metas = all_data["metadatas"]
    embeddings = all_data["embeddings"]
    
    print(f"Retrieved {len(ids)} items.")

    # 3. Create/Get new collection
    print(f"Creating new collection '{NEW_COLLECTION}'...")
    new_coll = client.get_or_create_collection(NEW_COLLECTION)
    
    # 4. Add data to new collection
    # We add in one go since it's only ~1k items, but could be batched if larger
    print(f"Adding data to '{NEW_COLLECTION}'...")
    new_coll.add(
        ids=ids,
        documents=docs,
        metadatas=metas,
        embeddings=embeddings
    )
    
    # 5. Verify
    new_count = new_coll.count()
    print(f"Migration complete!")
    print(f"New collection '{NEW_COLLECTION}' count: {new_count}")
    
    if new_count == len(ids):
        print("Verification SUCCESS: All items migrated.")
        # Optional: client.delete_collection(OLD_COLLECTION)
        # print(f"Deleted old collection '{OLD_COLLECTION}'.")
    else:
        print(f"Verification WARNING: Count mismatch! (Expected {len(ids)}, got {new_count})")

if __name__ == "__main__":
    migrate()
