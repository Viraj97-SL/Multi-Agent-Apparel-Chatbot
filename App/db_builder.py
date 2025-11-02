import os
import sqlalchemy
from sqlalchemy import create_engine, text

# Get the absolute path of the directory where this script is located (app/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to get the project root directory
project_root = os.path.dirname(script_dir)
# Database will be created in the project root
DB_PATH = os.path.join(project_root, "apparel.db")

print(f"Database will be created at: {DB_PATH}\n")

# Create a connection to the SQLite database
# The file will be created if it doesn't exist
engine = create_engine(f"sqlite:///{DB_PATH}")


def create_and_populate_tables():
    """Creates tables and inserts mock data."""
    with engine.connect() as conn:
        # Drop tables if they exist (for a clean slate)
        conn.execute(text("DROP TABLE IF EXISTS orders"))
        conn.execute(text("DROP TABLE IF EXISTS customers"))
        print("Dropped existing tables (if any).")

        # Create Customers table
        conn.execute(text("""
                          CREATE TABLE customers
                          (
                              customer_id VARCHAR(10) PRIMARY KEY,
                              name        VARCHAR(100),
                              email       VARCHAR(100)
                          )
                          """))
        print("Created 'customers' table.")

        # Create Orders table
        conn.execute(text("""
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
                          """))
        print("Created 'orders' table.")

        # Insert data into Customers
        conn.execute(text(
            "INSERT INTO customers (customer_id, name, email) VALUES ('CUS-A45', 'Alice Smith', 'alice@example.com')"))
        conn.execute(text(
            "INSERT INTO customers (customer_id, name, email) VALUES ('CUS-B12', 'Bob Johnson', 'bob@example.com')"))
        print("Inserted data into 'customers'.")

        # Insert data into Orders
        # We'll store 'items' as a JSON string
        items_1 = '[{"product_id": "T-001", "name": "Classic Tee", "quantity": 2}, {"product_id": "J-002", "name": "Urban Denim Jeans", "quantity": 1}]'
        items_2 = '[{"product_id": "H-003", "name": "Voyager Hoodie", "quantity": 1}]'

        conn.execute(text(f"""
        INSERT INTO orders (order_id, customer_id, status, shipping_carrier, tracking_number, items) 
        VALUES ('ORD-123', 'CUS-A45', 'Shipped', 'FedEx', 'FX77890123', '{items_1}')
        """))

        conn.execute(text(f"""
        INSERT INTO orders (order_id, customer_id, status, shipping_carrier, tracking_number, items) 
        VALUES ('ORD-456', 'CUS-B12', 'Processing', NULL, NULL, '{items_2}')
        """))
        print("Inserted data into 'orders'.")

        # Commit the changes
        conn.commit()
        print("\nDatabase creation and population complete!")


def test_database():
    """Tests that the data can be read."""
    print("\n--- Testing Database ---")
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT order_id, status, tracking_number FROM orders WHERE customer_id = 'CUS-A45'"))
        for row in result:
            print(f"Found order: {row}")
    print("--- Test Complete ---")


if __name__ == "__main__":
    create_and_populate_tables()
    test_database()
