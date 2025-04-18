#!/usr/bin/env python3
"""
Lam-RAG Ingestion Script

This script reads one or more text files and ingests their content into a SQLite
database for later retrieval-augmented generation (RaG). The database (by default,
named "knowledge.db") will store each file’s contents along with the source filename
and a timestamp. You can then use this database in your chat system to retrieve
relevant documents as context for your conversation.

Usage:
    python3 lam-rag.py new-data.txt additional-data.txt
    # Optionally, you can specify a custom database file:
    python3 lam-rag.py --db my_knowledge.db new-data.txt
"""

import argparse
import sqlite3
import os
import sys
from datetime import datetime

def create_table(conn: sqlite3.Connection):
    create_table_query = """
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_name TEXT NOT NULL,
        content TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """
    conn.execute(create_table_query)
    conn.commit()

def ingest_file(conn: sqlite3.Connection, file_path: str):
    if not os.path.exists(file_path):
        print(f"[Warning] File not found: {file_path}")
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except Exception as e:
        print(f"[Error] Reading '{file_path}': {e}")
        return

    if not content:
        print(f"[Notice] File '{file_path}' is empty. Skipping.")
        return

    insert_query = "INSERT INTO documents (file_name, content, timestamp) VALUES (?, ?, ?)"
    timestamp = datetime.utcnow().isoformat()
    conn.execute(insert_query, (os.path.basename(file_path), content, timestamp))
    conn.commit()
    print(f"[Success] Ingested '{file_path}' ({len(content)} characters).")

def list_documents(conn: sqlite3.Connection):
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_name, timestamp FROM documents ORDER BY id ASC")
    rows = cursor.fetchall()
    if not rows:
        print("No documents ingested yet.")
        return
    print("Documents in the knowledge base:")
    for row in rows:
        doc_id, file_name, timestamp = row
        print(f"  [{doc_id}] {file_name} - ingested at {timestamp}")

def main():
    parser = argparse.ArgumentParser(
        description="Ingest one or more text files into a SQLite-based knowledge base for retrieval-augmented generation."
    )
    parser.add_argument("files", nargs="+", help="Text file(s) to ingest")
    parser.add_argument("--db", type=str, default="knowledge.db", help="SQLite database file (default: knowledge.db)")
    args = parser.parse_args()

    try:
        conn = sqlite3.connect(args.db)
    except Exception as e:
        print(f"[Error] Could not connect to database '{args.db}': {e}")
        sys.exit(1)
    
    create_table(conn)

    for file_path in args.files:
        ingest_file(conn, file_path)

    print("\nIngestion complete.\n")
    list_documents(conn)
    conn.close()

if __name__ == "__main__":
    main()
