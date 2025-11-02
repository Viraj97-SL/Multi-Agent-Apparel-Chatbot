import os
from dotenv import load_dotenv

import torch  # Add this line here
print("Torch imported successfully:", torch.__version__)

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Get the absolute path of the directory where this script is located (app/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root directory
project_root = os.path.dirname(script_dir)

# Define the paths relative to the project root
DATA_PATH = os.path.join(project_root, "data", "products.txt")
INDEX_PATH = os.path.join(project_root, "faiss_index")


def create_vector_store():
    """Reads product data, splits it, creates embeddings, and saves to FAISS."""
    print("Loading product data...")
    loader = TextLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["---", "\n\n", "\n"],
        chunk_size=1000,
        chunk_overlap=0
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Initializing embedding model (this may take a moment to download)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("Creating and saving FAISS index...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_PATH)
    print(f"Successfully created and saved FAISS index to '{INDEX_PATH}'")


if __name__ == "__main__":
    create_vector_store()
