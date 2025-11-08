import os
import json
import uuid
import datetime
import sqlite3  # We'll use this for our new tool
from dotenv import load_dotenv
from typing import List, Optional

from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI

# SQL Toolkit Imports (for orders/returns)
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# SQLAlchemy Imports (for writing returns)
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file.")

# --- Database Setup ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
DB_PATH = os.path.join(project_root, "apparel.db")
DB_URI = f"sqlite:///{DB_PATH}"

if not os.path.exists(DB_PATH):
    raise FileNotFoundError(
        f"Database file not found at {DB_PATH}. "
        "Please run 'app/db_builder.py' first."
    )

# This 'db' object is for the READ-ONLY toolkit (orders, returns)
db = SQLDatabase.from_uri(DB_URI, include_tables=['orders', 'customers', 'returns'])

# This 'engine' is for our WRITE tool (initiate_return)
write_engine = create_engine(DB_URI)

# --- Initialize the LLM ---
# (Keeping your change to gemini-2.5-flash)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)

# --- Define Our Toolset ---

# Tool 1: SQL Database Toolkit (for Orders/Returns)
print("Initializing SQLDatabaseToolkit (for Orders, Customers, Returns)...")
sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
sql_tools = sql_toolkit.get_tools()
print(f"SQL tools created: {[t.name for t in sql_tools]}")


# Tool 2: Initiate Return (WRITE tool)
class InitiateReturnArgs(BaseModel):
    order_id: str = Field(..., description="The ID of the order to be returned. Example: 'ORD-123'")
    product_ids: List[str] = Field(..., description="A list of product IDs to be returned. Example: ['T-001', 'J-002']")


@tool("initiate_return", args_schema=InitiateReturnArgs)
def initiate_return(order_id: str, product_ids: List[str]) -> str:
    """Initiate a return for the specified order and products.

    This tool inserts a new return record into the 'returns' table with a generated return ID,
    the provided order ID, product IDs, and a 'Pending' status. It returns a JSON string
    with the return details, including a mock shipping label link.
    """
    print(f"\n--- DEBUG: Tool 'initiate_return' called for order {order_id} with products {product_ids} ---")
    try:
        return_id = f"RET-{uuid.uuid4().hex[:6].upper()}"
        return_date = datetime.date.today().isoformat()
        products_json = json.dumps(product_ids)
        with write_engine.connect() as conn:
            conn.execute(
                text(
                    "INSERT INTO returns (return_id, order_id, product_ids, status, return_date) "
                    "VALUES (:return_id, :order_id, :product_ids, :status, :return_date)"
                ),
                {
                    "return_id": return_id,
                    "order_id": order_id,
                    "product_ids": products_json,
                    "status": "Pending",
                    "return_date": return_date
                }
            )
            conn.commit()
        print(f"Successfully wrote return {return_id} to the database.")
        return json.dumps({
            "status": "Return Initiated Successfully",
            "return_id": return_id,
            "order_id": order_id,
            "status": "Pending",
            "return_label_link": f"https://shipping-co.com/labels/{return_id}"
        })
    except Exception as e:
        print(f"\n--- ERROR in initiate_return: {e} ---")
        return f"Error: Failed to initiate return. Reason: {e}"

# --- NEW TOOL ---
# In app/tools.py

# --- NEW TOOL ---
# Tool 3: Query Product Database (NEW)
class ProductQueryArgs(BaseModel):
    product_name: Optional[str] = Field(None, description="The name of the product. Example: 'Rosalind Blossom Dress'")
    category: Optional[str] = Field(None, description="The category of the product. Example: 'MINI DRESS'")
    colour: Optional[str] = Field(None, description="The colour of the product. Example: 'RED & BLACK'")


@tool("query_product_database", args_schema=ProductQueryArgs)
def query_product_database(product_name: Optional[str] = None, category: Optional[str] = None,
                           colour: Optional[str] = None) -> str:
    """
    Use this tool to search the 'products' table for product details,
    including name, price, colours, categories, and image URLs.
    Do NOT use this for orders or returns.
    """
    print(
        f"\n--- DEBUG: Tool 'query_product_database' called with: name={product_name}, category={category}, colour={colour} ---")

    # Build a dynamic query
    query = "SELECT product_name, category, colour, price, price_at_sale, image_url FROM products WHERE 1=1"
    params = {}

    if product_name:
        query += " AND product_name LIKE :product_name"
        params['product_name'] = f"%{product_name}%"
    if category:
        query += " AND category LIKE :category"
        params['category'] = f"%{category}%"
    if colour:
        query += " AND colour LIKE :colour"
        params['colour'] = f"%{colour}%"

    query += " LIMIT 5"  # Limit to 5 results

    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row  # Allows us to access columns by name
            cursor = conn.cursor()
            results = cursor.execute(query, params).fetchall()

            if not results:
                return "No products found matching those criteria. Try a broader search."

            # Format the results
            formatted_results = []
            for row in results:
                # Format price
                price = row['price_at_sale'] if row['price_at_sale'] < row['price'] else row['price']

                # Format image (as you requested)
                # Format image (as you requested)
                image_tag = ""
                if row['image_url']:
                    # Corrected f-string to create a valid HTML image tag
                    image_tag = f'<img src="{row["image_url"]}" alt="Image" />'

                formatted_results.append(
                    f"Product: {row['product_name']}\n"
                    f"Category: {row['category']}\n"
                    f"Colour: {row['colour']}\n"
                    f"Price: {price}\n"
                    f"{image_tag}"  # Add the image tag
                )

            return "\n\n".join(formatted_results)

    except Exception as e:
        print(f"\n--- ERROR in query_product_database: {e} ---")
        return f"Error: Failed to query product database. Reason: {e}"



# --- Combine all tools ---
def get_data_query_tools():
    """Returns all tools for the Data Query Agent."""
    # We combine the order/return tools with our new product tool
    all_tools = sql_tools + [initiate_return, query_product_database]
    return all_tools


if __name__ == "__main__":
    tools = get_data_query_tools()
    print(f"\n--- All Data Query Tools Initialized ---")
    for t in tools:
        print(f"- Tool: {t.name}")
        print(f"  Description: {t.description}\n")
