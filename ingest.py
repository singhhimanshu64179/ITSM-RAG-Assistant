# ingest.py

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = "data"
VECTOR_DB_DIR = "vectordb"

def load_documents():
    docs = []

    itil_dir = os.path.join(DATA_DIR, "itil")
    sops_dir = os.path.join(DATA_DIR, "sops")

    # Load PDF docs
    if os.path.isdir(itil_dir):
        for file in os.listdir(itil_dir):
            if file.lower().endswith(".pdf"):
                path = os.path.join(itil_dir, file)
                print(f"Loading PDF: {path}")
                loader = PyPDFLoader(path)
                docs.extend(loader.load())

    # Load SOP text docs
    if os.path.isdir(sops_dir):
        for file in os.listdir(sops_dir):
            if file.lower().endswith((".txt", ".md")):
                path = os.path.join(sops_dir, file)
                print(f"Loading text: {path}")
                loader = TextLoader(path, encoding="utf-8")
                docs.extend(loader.load())

    print(f"Total documents loaded: {len(docs)}")
    return docs

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks after split: {len(chunks)}")
    return chunks

def build_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    vectordb.persist()
    print(f"Vector DB created at: {VECTOR_DB_DIR}")

def main():
    docs = load_documents()
    if not docs:
        print("❌ No documents found. Please check your data folders.")
        return

    chunks = split_documents(docs)
    build_vectorstore(chunks)

if __name__ == "__main__":
    main()
