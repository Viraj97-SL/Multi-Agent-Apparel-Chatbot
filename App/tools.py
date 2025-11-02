import os
import json
from dotenv import load_dotenv
from typing import List

# --- Pydantic and Tool Imports ---
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# --- LLM Import ---
# The SQL toolkit needs an LLM to translate natural language into SQL
from langchain_google_genai import ChatGoogleGenerativeAI

# --- SQL Toolkit Imports ---
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# Load environment variables
load_dotenv()

# Get the Google API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# --- Database Setup ---
# Get the absolute path to the project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DB_PATH = os.path.join(project_root, "apparel.db")

# Check if the database file exists
if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"Database file not found at {DB_PATH}. "
        "Please run 'app/db_builder.py' first."
    )

# Connect to the database
# Connect to the database
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

# --- Initialize the LLM ---
# This LLM is specifically for the SQL tool to generate queries
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# --- Define Our Toolset ---

# Tool 1: SQL Database Toolkit
# This creates a "kit" of tools for our agent, like:
# - sql_db_query: To run queries
# - sql_db_schema: To see the table structure
# - sql_db_list_tables: To list available tables
print("Initializing SQLDatabaseToolkit...")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

# Get the list of tools from the toolkit
sql_tools = sql_toolkit.get_tools()
print(f"SQL tools created: {[t.name for t in sql_tools]}")


# Tool 2: Initiate Return (Our custom tool)
class InitiateReturnArgs(BaseModel):
    """Input schema for initiate_return tool."""
    order_id: str = Field(..., description="The ID of the order to be returned. Example: 'ORD-123'")
    product_ids: List[str] = Field(..., description="A list of product IDs to be returned. Example: ['T-001', 'J-002']")


@tool("initiate_return", args_schema=InitiateReturnArgs)
def initiate_return(order_id: str, product_ids: List[str]) -> str:
    """
    Use this tool to initiate a return for one or more products from a specific order.
    You must provide the 'order_id' and a list of 'product_ids'.
    This tool CANNOT check order status or product details.
    """
    print(f"\n--- DEBUG: Tool 'initiate_return' called for order {order_id} with products {product_ids} ---")

    # In a real app, this would write to the database or call a returns API.
    # We will just simulate it.

    # We could query our new DB, but we'll keep the mock logic for simplicity
    # to show a separation of concerns (a "write" action vs. a "read" action).

    print(f"Simulating return for order {order_id}...")

    # Simulate a successful return
    return json.dumps({
        "status": "Return Initiated",
        "return_label_link": "https://shipping-co.com/labels/RTN-987XYZ",
        "order_id": order_id,
        "returned_product_ids": product_ids,
        "next_steps": "Please print the return label and drop off the package at any FedEx location."
    })


# --- Combine all tools ---

# We'll export this list for our agent to use in Phase 3
def get_all_tools():
    all_tools = sql_tools + [initiate_return]
    return all_tools


if __name__ == "__main__":
    # This just tests that the tools can be loaded
    tools = get_all_tools()
    print(f"\n--- All Tools Initialized ---")
    for t in tools:
        print(f"- Tool: {t.name}")
        print(f"  Description: {t.description}\n")
