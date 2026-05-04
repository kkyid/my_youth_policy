
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path.cwd()))

from core.vector_db import collections_status

def test_status():
    status = collections_status()
    print(f"Current DB Status:")
    print(f" - Housing: {status['housing']} items")
    print(f" - Finance: {status['finance']} items")

if __name__ == "__main__":
    test_status()
