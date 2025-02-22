import os
import sys
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

import uvicorn
from app import app

if __name__ == "__main__":
    # Use the app string reference for hot reloading
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
