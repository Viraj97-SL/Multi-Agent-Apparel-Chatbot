import os
from dotenv import load_dotenv

# --- LLM for chatting ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Local model for loading the index ---
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- The rest of your imports ---
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Get the Google API key (for the CHAT model)
# Make sure your .env file has your GOOGLE_API_KEY
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file. Please add it.")

# --- START OF PATH FIX ---
# Get the absolute path of the directory where this script is located (app/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root directory
project_root = os.path.dirname(script_dir)
# Define the path to the index
INDEX_PATH = os.path.join(project_root, "faiss_index")


# --- END OF PATH FIX ---

def create_rag_chain():
    """Creates the RAG chain for answering questions."""
    # 1. Initialize the LLM (This uses the Google API)
    print("Initializing chat model (Gemini)...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

    # 2. Initialize the SAME embedding model used for indexing
    print("Initializing local embedding model (HuggingFace)...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 3. Load the saved FAISS index
    print(f"Loading FAISS index from {INDEX_PATH}...")
    # Check if the index path exists
    if not os.path.exists(INDEX_PATH):
        print(f"Error: FAISS index not found at {INDEX_PATH}")
        print("Please run the 'rag_indexer.py' script first to create the index.")
        return None

    # Add allow_dangerous_deserialization=True (required by FAISS)
    db = FAISS.load_local(
        INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    # 4. Create a retriever from the vector store
    retriever = db.as_retriever(search_kwargs={"k": 3})  # Retrieve top 3 results

    # Helper function to format documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 5. Create the prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful customer service assistant for an online apparel store.
    Answer the user's question based only on the following context.
    If the context doesn't contain the answer, say "I'm sorry, I don't have that information."

    CONTEXT: {context}

    QUESTION: {input}
    """)

    # 6. Create the RAG chain using LCEL
    rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    print("RAG chain created successfully.")
    return rag_chain


def main():
    # Make sure you are in the correct Conda environment!
    rag_chain = create_rag_chain()
    if rag_chain is None:
        return  # Exit if the chain couldn't be created

    print("\n--- Apparel Chatbot RAG ---")
    print("Ask questions about products, shipping, or returns. Type 'exit' to quit.")

    while True:
        try:
            query = input("\nYou: ")
            if query.lower() == 'exit':
                break
            if not query:
                continue

            # Invoke the chain
            response = rag_chain.invoke(query)
            print(f"Bot: {response}")
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()
