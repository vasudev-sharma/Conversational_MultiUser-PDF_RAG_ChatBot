import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from langchain.schema import BaseRetriever
from langchain.schema.document import Document
from pydantic import Extra

DB_NAME = "rag_app.db"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class LoggingRetriever(BaseRetriever):
    retriever: BaseRetriever

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.retriever.get_relevant_documents(query)
        print(f"[LOG] Query: {query}")
        for doc in docs:
            print(f"[LOG] Retrieved: {doc.page_content[:200]}...")
        return docs


# https://chatgpt.com/c/6834d738-2620-800e-882a-48cb828ded47


DB_NAME = "rag_app.db"


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def create_application_logs():
    conn = get_db_connection()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.close()


def create_document_store():
    conn = get_db_connection()
    conn.execute(
        """CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"""
    )
    conn.close()


def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)",
        (session_id, user_query, gpt_response, model),
    )
    conn.commit()
    conn.close()


def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    )
    messages = []
    for row in cursor.fetchall():
        messages.extend(
            [
                {"role": "human", "content": row["user_query"]},
                {"role": "ai", "content": row["gpt_response"]},
            ]
        )
    conn.close()
    return messages


def insert_document_record(filename):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO document_store (filename) VALUES (?)", (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id


def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM document_store WHERE id = ?", (file_id,))
    conn.commit()
    conn.close()
    return True


def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC"
    )
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]


if __name__ == "__main__":
    # Initialize the database tables
    create_application_logs()
    create_document_store()
