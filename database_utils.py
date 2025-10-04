# database_utils.py

import sqlite3
import re
import config

def get_db_connection():
    """Initializes and returns a database connection to the SQLite file."""
    try:
        conn = sqlite3.connect("vasuki_inventory.db")
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        print(f"SQLite Database connection error: {e}")
        return None

def format_product_results(results: list, start_index: int = 0, batch_size: int = 3) -> str:
    """Formats a list of database rows into a readable string without markdown."""
    if not results:
        return ""

    response_parts = []
    # Add a header only if it's the first batch of a multi-batch response
    if start_index == 0:
        response_parts.append("I found the following product(s) for you:\n")

    end_index = start_index + batch_size
    paginated_results = results[start_index:end_index]

    for i, product_row in enumerate(paginated_results, start=start_index + 1):
        product = dict(product_row)

        product_name = f"{product.get('category_name', '')} {product.get('subcategory_name', '')}".strip()
        product_desc = f"\n{i}. {product_name} (SKU: {product.get('SKU Number', 'N/A')})\n"

        details = []
        if product.get('stone_name'): details.append(f"   - Stone: {product['stone_name']}")
        if product.get('color_name'): details.append(f"   - Color: {product['color_name']}")
        if product.get('finish_name'): details.append(f"   - Finish: {product['finish_name']}")
        if product.get('Weight'): details.append(f"   - Weight: {product['Weight']:.2f} grams")

        dims = []
        if product.get('length'): dims.append(f"{product['length']:.2f}")
        if product.get('width'): dims.append(f"{product['width']:.2f}")
        if dims: details.append(f"   - Dimensions: {' x '.join(dims)}")

        product_desc += "\n".join(details)

        price = product.get('unit_price')
        if price is not None:
            product_desc += f"\n   - Price: ₹{float(price):.2f}\n"
        else:
            product_desc += "\n   - Price: Not available\n"

        response_parts.append(product_desc)

    return "".join(response_parts)

def get_products_by_skus(skus: list[str], price_limit: int = None) -> list:
    """
    Queries the database for a list of SKUs and returns their full details as a list of rows.
    """
    if not skus:
        return []
    conn = get_db_connection()
    if not conn:
        print("Database connection error.")
        return []
    cursor = conn.cursor()
    try:
        placeholders = ','.join('?' for _ in skus)
        params = list(skus)
        sql = f"""
        SELECT i.`SKU Number`, c.category_name, c.subcategory_name, s.stone_name,
               cl.color_name, f.finish_name, i.`Weight`, i.length, i.width,
               p.`Final SP` as unit_price, i.`Status`
        FROM inventory_items i
        LEFT JOIN categories c ON i.category_id = c.id
        LEFT JOIN stones s ON i.stone_id = s.id
        LEFT JOIN colors cl ON i.color_id = cl.id
        LEFT JOIN finishes f ON i.finish_id = f.id
        LEFT JOIN pricing p ON i.`SKU Number` = p.`SKU Number`
        WHERE i.`SKU Number` IN ({placeholders}) AND i.`Status` = 'In Stock'
        """
        if price_limit is not None:
            sql += " AND p.`Final SP` <= ?"
            params.append(price_limit)

        cursor.execute(sql, tuple(params))
        results = cursor.fetchall()
        return results
    except Exception as e:
        print(f"Database SKU query error: {e}")
        return []
    finally:
        if conn:
            conn.close()

def search_products(query_text: str) -> str:
    """Parses a natural language query to search products by category and price."""
    # This function is not used in the main query flow anymore but kept for potential direct use
    conn = get_db_connection()
    if not conn:
        return "Database connection error."
    cursor = conn.cursor()
    try:
        product_types = {
            "bangle": ["bangle", "bangles"], "ring": ["ring", "rings"],
            "necklace": ["necklace", "necklaces", "chain", "chains", "pendant", "pendants"],
            "earring": ["earring", "earrings", "stud", "studs", "dangler", "danglers", "jhumka", "jhumkas"],
            "choker": ["choker", "chokers"],
        }
        found_product_type = next((ptype for ptype, kw in product_types.items() if any(k in query_text.lower() for k in kw)), None)
        
        price_limit = None
        price_match = re.search(r"(under|less than|below)\s*(\d+)", query_text, re.IGNORECASE)
        if price_match:
            price_limit = int(price_match.group(2))

        sql = """
        SELECT i.`SKU Number`, c.category_name, c.subcategory_name, s.stone_name,
               cl.color_name, f.finish_name, i.`Weight`, i.length, i.width,
               p.`Final SP` as unit_price, i.`Status`
        FROM inventory_items i
        LEFT JOIN categories c ON i.category_id = c.id
        LEFT JOIN stones s ON i.stone_id = s.id
        LEFT JOIN colors cl ON i.color_id = cl.id
        LEFT JOIN finishes f ON i.finish_id = f.id
        LEFT JOIN pricing p ON i.`SKU Number` = p.`SKU Number`
        WHERE i.`Status` = 'In Stock'
        """
        params = []

        if found_product_type:
            sql += " AND (LOWER(c.category_name) LIKE ? OR LOWER(c.subcategory_name) LIKE ?)"
            params.extend([f"%{found_product_type}%", f"%{found_product_type}%"])

        if price_limit is not None:
            sql += " AND p.`Final SP` <= ?"
            params.append(price_limit)
        
        if not found_product_type and not price_limit:
             return ""

        sql += " ORDER BY p.`Final SP` LIMIT 10"
        cursor.execute(sql, tuple(params))
        results = cursor.fetchall()
        # Note: This function returns raw results now, formatting is handled by the caller.
        return results
    except Exception as e:
        print(f"Database search error: {e}")
        return "Error searching for products."
    finally:
        if conn:
            conn.close()