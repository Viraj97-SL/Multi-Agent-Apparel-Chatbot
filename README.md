# 🤖 Agentic AI Apparel Chatbot

This repository contains the source code for an advanced, multi-agent AI chatbot for an online apparel business. The agent is designed to handle customer service inquiries by intelligently routing requests to different "specialist" agents: one for general knowledge (RAG) and another for live database interactions (Tools).

The entire system is built in Python using **LangGraph** for agent orchestration and is served via a **FastAPI** API.

---

## 🏛️ Project Architecture

This project uses a "Supervisor" agent built with LangGraph to manage a team of specialist agents.

1.  A user sends a query (e.g., "Where is my order?") to the **FastAPI** server.
2.  The `server.py` file forwards the request to the main **LangGraph** agent.
3.  The **Supervisor Agent** (the "brain") analyzes the query.
4.  The Supervisor routes the query to the correct specialist:
    * **If it's a general question** (e.g., "What colors does the Classic Tee come in?"), it routes to the **RAG Agent**. This agent retrieves context from the `faiss_index` (built from `products.txt`) and generates an answer.
    * **If it's a specific, live-data question** (e.g., "What's the status of order ORD-123?"), it routes to the **Tool Agent**. This agent uses the `SQLDatabaseToolkit` to query the `apparel.db` (SQLite) database.
    * **If it's an action** (e.g., "I need to start a return..."), it routes to the **Tool Agent**, which uses the custom `initiate_return` tool to **write** a new entry into the `returns` table in the database.
5.  The final, natural-language response is streamed back to the user via the API.
6.  All interactions are traced and logged to **LangSmith** for debugging and monitoring.

---

## ✨ Features

* **Multi-Agent System:** Uses LangGraph to create a supervisor and multiple specialist agents.
* **RAG for Knowledge:** Answers general questions about products and policies using a FAISS vector store.
* **Live Database Tools (Read/Write):**
    * **Read:** Connects to a live SQLite database (`apparel.db`) to check order status, customer details, and more.
    * **Write:** Processes new returns by writing directly to the `returns` table.
* **API Server:** A production-ready API built with FastAPI, including a `/chat` endpoint and automatic `/docs`.
* **Persistent Memory:** Uses `SqliteSaver` to give each conversation (via `thread_id`) a persistent memory.
* **Local Embeddings:** Uses HuggingFace `all-MiniLM-L6-v2` for free, fast, and private embeddings.
* **LLM Integration:** Powered by the Google Gemini API.
* **Full Observability:** Integrated with LangSmith for end-to-end tracing of agent decisions and tool calls.

---

## 🛠️ Tech Stack

* **Core Framework:** LangChain, LangGraph
* **LLM:** Google Gemini
* **API:** FastAPI, Uvicorn
* **Database:** SQLite
* **Database Toolkit:** SQLAlchemy
* **RAG:** FAISS (Vector Store), HuggingFace Embeddings
* **Monitoring:** LangSmith
* **Environment:** Conda

---

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
```

### 2. Set Up the Conda Environment

This project is configured to run in a clean Conda environment to avoid package conflicts.

```bash
# 1. Create a new Conda environment
conda create -n newchatbot python=3.11 -y

# 2. Activate the new environment
conda activate newchatbot
```

### 3. Install Dependencies

Install all required Python packages from the `requirements.txt` file.

```bash
# Make sure you are in the (newchatbot) environment
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

You must create a `.env` file in the project's root directory to store your API keys.

1.  Create the file: `touch .env`
2.  Paste the following into the `.env` file, adding your own keys.

```.env
# .env.example

# Get your Google API key from Google AI Studio
GOOGLE_API_KEY="your-google-api-key-here"

# Get your LangSmith keys from smith.langchain.com
LANGCHAIN_TRACING_V2="true"
LANGCHAIN_ENDPOINT="[https://api.smith.langchain.com](https://api.smith.langchain.com)"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="apparel-chatbot" # Or any name you choose
```

### 5. Build Project Data

You need to run two scripts **one time** to build your database and your vector index.

```bash
# 1. Build the SQLite database (apparel.db)
python app/db_builder.py

# 2. Build the FAISS vector store (faiss_index)
python app/rag_indexer.py
```

### 6. Run the API Server

You're all set! Run the FastAPI server with Uvicorn.

```bash
# This will start the server on [http:Request the detailes]
# The --reload flag automatically restarts the server when you save code
uvicorn server:api --reload
```

---

## 🧪 How to Use

Once the server is running, you can interact with your chatbot in two ways:

### 1. Via the API Docs (Recommended)

1.  Open your browser and go to **`Request the detailes`**.
2.  Click on the `POST /chat` endpoint and then "Try it out".
3.  Use the "Request body" to send queries.



---

## 🗣️ Example Queries to Try

#### RAG Agent (General Knowledge)
* `"What is your return policy?"`
* `"What colors does the Classic Tee come in?"`
* `"How much is the Voyager Hoodie?"`

#### Tool Agent (SQL Read)
* `"What is the status of order ORD-123?"`
* `"Who is customer CUS-A45?"`
* `"Find the tracking number for order ORD-123."`

#### Tool Agent (SQL Write)
* `"I need to start a return for order ORD-456."`
* `(Follow up): "Are there any returns in the system now?"`

#### Multi-Tool Query
* `"What items are in order ORD-123 and who is the customer?"`
