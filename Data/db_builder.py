import os
import sqlite3
import pandas as pd  # Import pandas

# --- Setup Project Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Database path
DB_PATH = os.path.join(project_root, "apparel.db")
# Point to the .xlsx file
EXCEL_PATH = os.path.join(project_root, "clothing_details.xlsx")


def create_products_table(conn):
    """Creates the new 'products' table, dropping the old one if it exists."""
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS products")
    print("Dropped old 'products' table (if it existed).")

    cursor.execute("""
                   CREATE TABLE products
                   (
                       product_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                       product_name  TEXT NOT NULL,
                       category      TEXT,
                       colour        TEXT,
                       price         REAL,
                       price_at_sale REAL,
                       description   TEXT,
                       image_url     TEXT
                   )
                   """)
    print("Created new 'products' table.")
    conn.commit()


def load_products_from_excel(conn):
    """Loads all products from the Excel file into the 'products' table."""
    print(f"Opening Excel file: {EXCEL_PATH}")

    try:
        df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")

        # --- THIS IS THE FIX ---
        # We explicitly rename all columns we care about.
        # This handles case differences (e.g., 'image_url' vs 'Image_URL').
        df_renamed = df.rename(columns={
            'Product Name': 'product_name',
            'Category': 'category',
            'Colour': 'colour',
            'Price': 'price',
            'Price at sale': 'price_at_sale',
            'Image_URL': 'image_url',  # Handles 'Image_URL'
            'image_url': 'image_url'  # Handles 'image_url'
        })

        # Add the 'description' column if it doesn't exist
        if 'description' not in df_renamed.columns:
            df_renamed['description'] = None

        # Check if 'image_url' column exists *after* renaming
        if 'image_url' not in df_renamed.columns:
            print("WARNING: No 'image_url' or 'Image_URL' column found in Excel. Image URLs will be empty.")
            df_renamed['image_url'] = None

        # Select only the columns that match our database table
        columns_to_insert = [
            'product_name', 'category', 'colour', 'price',
            'price_at_sale', 'description', 'image_url'
        ]
        df_final = df_renamed[columns_to_insert]

        if df_final.empty:
            print("No products found in the Excel file.")
            return

        print(f"Found {len(df_final)} products. Inserting into database...")

        df_final.to_sql(
            'products',
            conn,
            if_exists='append',
            index=False
        )

        conn.commit()
        print(f"Successfully inserted {len(df_final)} products.")

    except FileNotFoundError:
        print(f"ERROR: Could not find the file at {EXCEL_PATH}")
        print("Please make sure 'clothing_details.xlsx' is in the root of your project folder.")
    except Exception as e:
        print(f"An error occurred while reading the Excel file: {e}")


def main():
    try:
        conn = sqlite3.connect(DB_PATH)
        print(f"Connected to database at: {DB_PATH}")

        cursor = conn.cursor()

        # --- 1. Recreate Customers Table ---
        cursor.execute("DROP TABLE IF EXISTS customers")
        cursor.execute("""
                       CREATE TABLE customers
                       (
                           customer_id VARCHAR(10) PRIMARY KEY,
                           name        VARCHAR(100),
                           email       VARCHAR(100)
                       )
                       """)
        cursor.execute(
            "INSERT INTO customers (customer_id, name, email) VALUES ('CUS-A45', 'Alice Smith', 'alice@example.com')")
        cursor.execute(
            "INSERT INTO customers (customer_id, name, email) VALUES ('CUS-B12', 'Bob Johnson', 'bob@example.com')")
        print("Recreated 'customers' table.")

        # --- 2. Recreate Orders Table ---
        cursor.execute("DROP TABLE IF EXISTS orders")
        cursor.execute("""
                       CREATE TABLE orders
                       (
                           order_id         VARCHAR(10) PRIMARY KEY,
                           customer_id      VARCHAR(10),
                           status           VARCHAR(50),
                           shipping_carrier VARCHAR(50),
                           tracking_number  VARCHAR(20),
                           items            TEXT,
                           FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
                       )
                       """)
        items_1 = '[{"product_id": "T-001", "name": "Classic Tee", "quantity": 2}, {"product_id": "J-002", "name": "Urban Denim Jeans", "quantity": 1}]'
        items_2 = '[{"product_id": "H-003", "name": "Voyager Hoodie", "quantity": 1}]'
        cursor.execute(
            f"INSERT INTO orders (order_id, customer_id, status, shipping_carrier, tracking_number, items) VALUES ('ORD-123', 'CUS-A45', 'Shipped', 'FedEx', 'FX77890123', '{items_1}')")
        cursor.execute(
            f"INSERT INTO orders (order_id, customer_id, status, shipping_carrier, tracking_number, items) VALUES ('ORD-460', 'CUS-B12', 'Processing', NULL, NULL, '{items_2}')")
        print("Recreated 'orders' table.")

        # --- 3. Recreate Returns Table ---
        cursor.execute("DROP TABLE IF EXISTS returns")
        cursor.execute("""
                       CREATE TABLE returns
                       (
                           return_id   VARCHAR(10) PRIMARY KEY,
                           order_id    VARCHAR(10),
                           product_ids TEXT,
                           status      VARCHAR(50),
                           return_date TEXT,
                           FOREIGN KEY (order_id) REFERENCES orders (order_id)
                       )
                       """)
        print("Recreated 'returns' table.")

        conn.commit()
        print("--- Base tables (customers, orders, returns) created successfully. ---")

        # --- 4. Create and Load Products Table ---
        create_products_table(conn)
        load_products_from_excel(conn)

        print("\nDatabase build complete!")

    except sqlite3.Error as e:
        print(f"A database error occurred: {e}")
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")


if __name__ == "__main__":
    main()
