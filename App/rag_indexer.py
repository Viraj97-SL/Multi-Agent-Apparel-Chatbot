import os
from dotenv import load_dotenv

# --- UPDATED IMPORTS ---
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables
load_dotenv()

# --- Setup Project Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

DATA_PATH = os.path.join(project_root, "data")
INDEX_PATH = os.path.join(project_root, "faiss_index")


def create_vector_store():
    """
    Reads all .txt files from the /data directory,
    splits them, creates embeddings, and saves to FAISS.
    """
    print(f"Loading documents from: {DATA_PATH}")

    # --- UPDATED LOADER ---
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.txt",  # Look for all .txt files
        loader_cls=TextLoader,  # Use TextLoader for each file

        # VVVV --- THIS IS THE FIX --- VVVV
        loader_kwargs={"encoding": "utf-8"}  # Explicitly use UTF-8
        # ^^^^ --- END OF FIX --- ^^^^
    )
    documents = loader.load()

    if not documents:
        print("No .txt files found in the /data directory. Please add some.")
        return

    print(f"Loaded {len(documents)} document(s).")

    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["---", "\n\n", "\n"],
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    print("Initializing embedding model (HuggingFace)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("Creating and saving FAISS index...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_PATH)

    print(f"Successfully created and saved FAISS index to '{INDEX_PATH}'")


if __name__ == "__main__":
    create_vector_store()
