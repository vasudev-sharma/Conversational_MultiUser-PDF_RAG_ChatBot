# import Essential dependencies

import os

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    UnstructuredHTMLLoader,
)
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

try:
    load_dotenv(".env")
except Exception as e:
    raise Exception("Please create a .env file with your OpenAI API key") from e


def load_document(filepath):
    if filepath.lower().endswith(".pdf"):
        loader = PyPDFLoader(filepath)
    elif filepath.lower().endswith(".pdf"):
        loader = Docx2txtLoader(filepath)
    elif filepath.lower().endswith(".html"):
        loader = UnstructuredHTMLLoader(filepath)
    else:
        raise Error("Unsupported file type: {filepath}")
    documents = loader.load()
    return documents


def ingest_pdf_to_vectorstore(pdf_path, DB_FAISS_PATH):

    docs = load_document(pdf_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_FAISS_PATH)
    return db


# create a new file named vectorstore in your current directory.
if __name__ == "__main__":

    DB_FAISS_PATH = "vectorstore/db_faiss"
    db = ingest_pdf_to_vectorstore("/.input_pdf.pdf", DB_FAISS_PATH)
