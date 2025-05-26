#import Essential dependencies

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
try:
        load_dotenv(".env")
except Exception as e:
        raise Exception("Please create a .env file with your OpenAI API key") from e


def ingest_pdf_to_vectorstore(pdf_path, DB_FAISS_PATH):
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(api_key=os.environ.get("OPENAI_API_KEY"))
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(DB_FAISS_PATH)
        return db


#create a new file named vectorstore in your current directory.
if __name__=="__main__":

        DB_FAISS_PATH = 'vectorstore/db_faiss'
        db = ingest_pdf_to_vectorstore("/.input_pdf.pdf", DB_FAISS_PATH)
