# data_loader_main.py

import config
import vector_store_utils as vsu
from chromadb.utils import embedding_functions

# --- New Imports for Shopify ---
import shopify
from bs4 import BeautifulSoup
from config import SHOPIFY_SHOP_NAME, SHOPIFY_API_KEY, SHOPIFY_ADMIN_API_ACCESS_TOKEN
# --- End New Imports ---

def load_shopify_products():
    """
    Fetches product data from the Shopify API and returns it in a format
    ready for ChromaDB, with enriched content for better searchability.
    """
    print("Attempting to load products from Shopify...")
    documents, ids, metadatas = [], [], []
    try:
        shop_url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_ADMIN_API_ACCESS_TOKEN}@{SHOPIFY_SHOP_NAME}.myshopify.com/admin"
        shopify.ShopifyResource.set_site(shop_url)
        
        products = shopify.Product.find()

        if not products:
            print("🛑 CRITICAL: No products found on Shopify. The product collection will be empty.")
            return None

        print(f"Found {len(products)} products on Shopify. Preparing them for the vector store...")

        for i, product in enumerate(products):
            soup = BeautifulSoup(product.body_html, 'html.parser')
            clean_description = soup.get_text(separator=' ', strip=True)

            # --- NEW: Get SKU and Price safely ---
            variant = product.variants[0] if product.variants else None
            price = variant.price if variant else "N/A"
            sku = variant.sku if variant else "N/A"
            
            # Enriched content for better search retrieval
            content = (
                f"SKU: {sku}. "
                f"Product Name: {product.title}. "
                f"Product Type: {product.product_type}. "
                f"Description: {clean_description}. "
                f"Price: {price}"
            )
            
            documents.append(content)
            # --- NEW: Add SKU to metadata for potential future use ---
            metadatas.append({"source": "shopify", "product_title": product.title, "sku": sku})
            ids.append(f"shopify_product_{product.id}_{i}")
        
        print(f"✅ Successfully prepared {len(documents)} products from Shopify.")
        return {"documents": documents, "ids": ids, "metadatas": metadatas}

    except Exception as e:
        print(f"🛑 CRITICAL ERROR loading products from Shopify: {e}")
        print("Please double-check your API credentials, shop name in config.py, and app permissions on Shopify.")
        return None


def main():
    """
    Standalone script to initialize ChromaDB, load all documents from all sources,
    and store their embeddings.
    """
    print("Starting ChromaDB data loading process...")

    # 1. Initialize ChromaDB Client and Embedding Function
    try:
        chroma_client = vsu.init_chroma_client()
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_ID)
        print("ChromaDB client and embedding function initialized.")
    except Exception as e:
        print(f"Error initializing ChromaDB client or embedding function: {e}")
        return

    # 2. Get or Create Collections
    try:
        collections = vsu.get_chroma_collections(chroma_client, chroma_ef)
        shopify_collection_name = "shopify_products"
        if shopify_collection_name not in [c.name for c in chroma_client.list_collections()]:
             collections[shopify_collection_name] = chroma_client.create_collection(name=shopify_collection_name, embedding_function=chroma_ef)
             print(f"Created new collection: {shopify_collection_name}")
        else:
             collections[shopify_collection_name] = chroma_client.get_collection(name=shopify_collection_name, embedding_function=chroma_ef)

        print(f"Collections retrieved/created: {[c.name for c in chroma_client.list_collections()]}")
    except Exception as e:
        print(f"Error getting/creating ChromaDB collections: {e}")
        return

    # 3. Load Documents from standard sources (files, database)
    print("Loading documents from files and local database...")
    all_document_data = vsu.load_documents_for_vector_store()
    if not all_document_data:
        print("No documents were loaded from standard sources.")
    else:
        print("Standard documents loaded successfully.")

    # --- Load Shopify Product Data ---
    shopify_data = load_shopify_products()
    if shopify_data:
        all_document_data[shopify_collection_name] = shopify_data

    if not all_document_data:
        print("No documents loaded from any source. Exiting.")
        return

    # 4. Clear existing data and add new documents to collections
    for collection_key, data_dict in all_document_data.items():
        if collection_key not in collections:
            print(f"Warning: No collection object found for key '{collection_key}'. Skipping.")
            continue
        
        current_collection = collections[collection_key]
        print(f"\nProcessing collection: {current_collection.name} (for key '{collection_key}')")

        print(f"Clearing existing documents from {current_collection.name}...")
        if vsu.safely_clear_collection(current_collection):
            print(f"Successfully cleared {current_collection.name}.")
        else:
            print(f"Warning: Could not fully clear {current_collection.name}. Proceeding with adding new data.")

        docs = data_dict.get("documents")
        ids = data_dict.get("ids")
        metadatas = data_dict.get("metadatas")

        if not all([docs, ids, metadatas]) or not docs:
            print(f"No documents to add for {collection_key}. Skipping add operation.")
            continue
        
        print(f"Adding {len(docs)} documents to {current_collection.name}...")
        success = vsu.add_documents_in_batches(
            current_collection,
            docs,
            ids,
            metadatas,
            batch_size=50
        )
        if success:
            print(f"Successfully added documents to {current_collection.name}.")
            print(f"Collection {current_collection.name} now contains {current_collection.count()} documents.")
        else:
            print(f"Failed to add all documents to {current_collection.name}.")

    print("\nChromaDB data loading process completed.")
    print(f"Vector store data is persisted in: {config.CHROMA_PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()