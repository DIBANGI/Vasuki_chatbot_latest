# data_loader_main.py

import config
import vector_store_utils as vsu
from chromadb.utils import embedding_functions
import shopify
from bs4 import BeautifulSoup
from config import SHOPIFY_SHOP_NAME, SHOPIFY_API_KEY, SHOPIFY_ADMIN_API_ACCESS_TOKEN

def load_shopify_products_for_vector_store():
    """
    Fetches all product data from the Shopify API, handling pagination,
    and returns it in a format ready for ChromaDB with rich metadata.
    """
    print("Connecting to Shopify to load the full product catalog...")
    documents, ids, metadatas = [], [], []
    
    try:
        shop_url = f"https://{SHOPIFY_API_KEY}:{SHOPIFY_ADMIN_API_ACCESS_TOKEN}@{SHOPIFY_SHOP_NAME}.myshopify.com/admin"
        shopify.ShopifyResource.set_site(shop_url)
        
        page = shopify.Product.find(limit=250)
        products = list(page)
        
        while page.has_next_page():
            page = page.next_page()
            products.extend(page)

        if not products:
            print("🛑 CRITICAL: No products found on Shopify. The product collection will be empty.")
            return None

        print(f"Found a total of {len(products)} products on Shopify. Preparing them for the vector store...")

        for i, product in enumerate(products):
            soup = BeautifulSoup(product.body_html or "", 'html.parser')
            clean_description = soup.get_text(separator=' ', strip=True)
            
            variant = product.variants[0] if product.variants else None
            price = float(variant.price) if variant and variant.price else 0.0
            sku = variant.sku if variant and variant.sku else f"SKU_MISSING_{product.id}"
            
            # This is the text that will be embedded for semantic search.
            # It's enriched with key terms to improve retrieval accuracy.
            content = (
                f"Product Name: {product.title}. "
                f"Type: {product.product_type}. "
                f"Description: {clean_description}. "
                f"Tags: {product.tags}."
            )
            
            documents.append(content)
            
            # This rich metadata is crucial for hybrid retrieval and filtering.
            metadatas.append({
                "source": "shopify",
                "product_title": product.title,
                "sku": sku,
                "product_type": product.product_type or "",
                "price": price,
                "tags": product.tags or ""
                # You can add more filterable fields here, e.g., vendor, color from tags, etc.
            })
            ids.append(f"shopify_product_{product.id}")
        
        print(f"✅ Successfully prepared {len(documents)} products from Shopify for embedding.")
        return {"documents": documents, "ids": ids, "metadatas": metadatas}

    except Exception as e:
        print(f"🛑 CRITICAL ERROR loading products from Shopify: {e}")
        print("Please double-check your API credentials, shop name in config.py, and app permissions on Shopify.")
        return None

def main():
    """
    Initializes ChromaDB and loads all product data directly from Shopify,
    overwriting the existing collection to ensure data is fresh.
    """
    print("Starting ChromaDB data loading process...")

    try:
        chroma_client = vsu.init_chroma_client()
        chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=config.EMBEDDING_MODEL_ID)
        print("ChromaDB client and embedding function initialized.")
    except Exception as e:
        print(f"Fatal error initializing ChromaDB: {e}")
        return

    shopify_collection_name = "shopify_products"
    
    try:
        collections = {
            shopify_collection_name: chroma_client.get_or_create_collection(
                name=shopify_collection_name, 
                embedding_function=chroma_ef
            )
        }
        print(f"Successfully connected to collection: '{shopify_collection_name}'")
    except Exception as e:
        print(f"Fatal error connecting to Chroma collection: {e}")
        return

    # Load and process Shopify data
    shopify_data = load_shopify_products_for_vector_store()
    
    if shopify_data:
        current_collection = collections["shopify_products"]
        print(f"\nProcessing collection: {current_collection.name}")

        print("Clearing all existing documents to ensure freshness...")
        if vsu.safely_clear_collection(current_collection):
             print(f"Successfully cleared {current_collection.name}.")
        else:
            print(f"Warning: Collection could not be cleared. Data will be upserted.")

        
        print(f"Adding {len(shopify_data['documents'])} new documents to the vector store...")
        success = vsu.add_documents_in_batches(
            collection=current_collection,
            documents=shopify_data["documents"],
            ids=shopify_data["ids"],
            metadatas=shopify_data["metadatas"],
            batch_size=100  # Batches of 100 are efficient
        )
        if success:
            print(f"✅ Collection '{current_collection.name}' now contains {current_collection.count()} documents.")
        else:
            print(f"🛑 Failed to add all documents to the collection.")
    else:
        print("🛑 Failed to load any data from Shopify. The vector store was not updated.")

    print("\nChromaDB data loading process completed.")
    print(f"Vector store data is persisted in: {config.CHROMA_PERSIST_DIRECTORY}")

if __name__ == "__main__":
    main()