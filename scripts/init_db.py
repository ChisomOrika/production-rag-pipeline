#!/usr/bin/env python3
"""Initialize the database: create tables and enable pgvector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.session import init_db

if __name__ == "__main__":
    init_db()
