# vector_store_utils.py

import os
import pandas as pd
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import Optional, Dict

import config
import database_utils

def init_chroma_client():
    """Initializes and returns a persistent ChromaDB client."""
    os.makedirs(config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
    return client

def get_chroma_collections(client: chromadb.PersistentClient, embedding_func):
    """Gets or creates all necessary ChromaDB collections."""
    collections = {}
    collection_names_map = {
        "faqs": config.CHROMA_COLLECTION_FAQS,
        "policies": config.CHROMA_COLLECTION_POLICIES,
        "products": "shopify_products",  # Use the new Shopify collection name
    }
    for name_key, collection_name_val in collection_names_map.items():
        try:
            collection = client.get_or_create_collection(
                name=collection_name_val,
                # metadata={"hnsw:space": "cosine"} # Optional: specify distance function
            )
            print(f"Retrieved/Created collection: {collection_name_val}")
            collections[name_key] = collection
        except Exception as e:
            print(f"Error getting or creating collection {collection_name_val}: {e}")
    return collections

def load_documents_for_vector_store():
    """
    Loads and processes documents from static files (FAQs, Policies).
    The product loading is now handled separately by the Shopify-specific function.
    """
    all_data = {}

    # --- Load FAQs ---
    faq_docs, faq_ids, faq_metadata = [], [], []
    try:
        faqs_df = pd.read_csv(config.FAQS_FILE_PATH)
        print(f"Successfully loaded {config.FAQS_FILE_PATH} with {len(faqs_df)} questions")
        for idx, row in faqs_df.iterrows():
            if pd.isna(row['Question']) or pd.isna(row['Answer']): continue
            question, answer = str(row['Question']).strip(), str(row['Answer']).strip()
            if not question or not answer: continue
            
            faq_docs.append(f"Question: {question}\nAnswer: {answer}")
            faq_ids.append(f"faq_{idx}_main")
            faq_metadata.append({"type": "faq", "question": question, "source": config.FAQS_FILE_PATH})
    except Exception as e:
        print(f"Error loading FAQs from {config.FAQS_FILE_PATH}: {e}")
    
    all_data["faqs"] = {"documents": faq_docs, "ids": faq_ids, "metadatas": faq_metadata}

    # --- Load Policy Documents ---
    policy_docs, policy_ids, policy_metadata = [], [], []
    policy_files_map = {
        "privacy": config.PRIVACY_POLICY_FILE_PATH,
        "return": config.RETURN_POLICY_FILE_PATH,
        "shipping": config.SHIPPING_POLICY_FILE_PATH,
    }
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=150)
    for policy_type, file_path in policy_files_map.items():
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                policy_docs.append(chunk)
                policy_ids.append(f"{policy_type}_{i}")
                policy_metadata.append({"type": "policy", "policy_type": policy_type, "source": file_path})
        except Exception as e:
            print(f"Error loading policy file {file_path}: {e}")

    all_data["policies"] = {"documents": policy_docs, "ids": policy_ids, "metadatas": policy_metadata}
    
    return all_data

def validate_metadata(metadata_list: list) -> list:
    """Ensures all metadata values are valid types for ChromaDB."""
    valid_metadata_list = []
    for metadata_dict in metadata_list:
        validated_item = {}
        if isinstance(metadata_dict, dict):
            for key, value in metadata_dict.items():
                if isinstance(value, (str, int, float, bool)):
                    validated_item[key] = value
                elif value is None:
                    # ChromaDB prefers not to have None values. Use an empty string or a default.
                    validated_item[key] = ""
                else:
                    validated_item[key] = str(value)
        valid_metadata_list.append(validated_item)
    return valid_metadata_list

def add_documents_in_batches(collection, documents: list, ids: list, metadatas: list, batch_size=100):
    """Adds documents to a ChromaDB collection in batches, with validation."""
    if not all(isinstance(lst, list) for lst in [documents, ids, metadatas]):
        print("Error: Input data must be lists.")
        return False
    if not (len(documents) == len(ids) == len(metadatas)):
        print("Error: Document, ID, and metadata lists must be the same length.")
        return False
    if not documents:
        print("No documents to add.")
        return True

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        batch_meta_raw = metadatas[i:i + batch_size]
        batch_meta_validated = validate_metadata(batch_meta_raw)

        try:
            collection.add(documents=batch_docs, ids=batch_ids, metadatas=batch_meta_validated)
            print(f"Added batch {i//batch_size + 1}/{ -(-len(documents) // batch_size)} to '{collection.name}'.")
        except Exception as e:
            print(f"Error adding batch {i//batch_size + 1} to '{collection.name}': {e}")
            continue # Continue with the next batch
    return True

def safely_clear_collection(collection):
    """Safely clears all documents from a ChromaDB collection."""
    try:
        count = collection.count()
        if count == 0:
            return True
        # Fetch all items without payload to just get IDs
        all_items = collection.get(limit=count, include=[])
        if all_items and all_items['ids']:
            collection.delete(ids=all_items['ids'])
        
        final_count = collection.count()
        if final_count == 0:
            print(f"Successfully cleared {count} items from '{collection.name}'.")
            return True
        else:
            print(f"Warning: After deletion, {final_count} items still remain in '{collection.name}'.")
            return False
    except Exception as e:
        print(f"Error clearing collection '{collection.name}': {e}")
        return False

# --- REFACTORED: This is the key change for hybrid search ---
def get_langchain_chroma_retriever(
    collection_name: str, 
    embedding_model, 
    k_results: int = 5, 
    filter_criteria: Optional[Dict] = None
):
    """
    Creates a Langchain Chroma retriever for a given collection with optional metadata filtering.
    """
    vectorstore = Chroma(
        client=init_chroma_client(),
        collection_name=collection_name,
        embedding_function=embedding_model,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )
    
    search_kwargs = {"k": k_results}
    if filter_criteria:
        # The 'where' clause is used for metadata filtering in ChromaDB
        search_kwargs["filter"] = filter_criteria
        print(f"Retriever for '{collection_name}' created with filter: {filter_criteria}")
        
    return vectorstore.as_retriever(search_kwargs=search_kwargs)

def get_context_from_vector_store(query_text: str, collection_key: str, llm_app_components):
    """Helper to retrieve context from a specific vector store collection."""
    if not llm_app_components or "embedding_model" not in llm_app_components:
        print("Error: Embedding model not available.")
        return ""

    collection_name_map = {
        "faqs": config.CHROMA_COLLECTION_FAQS,
        "policies": config.CHROMA_COLLECTION_POLICIES,
        "products": "shopify_products",
    }
    
    collection_name = collection_name_map.get(collection_key)
    if not collection_name:
        print(f"Warning: Invalid collection key '{collection_key}'.")
        return ""

    try:
        retriever = get_langchain_chroma_retriever(
            collection_name=collection_name,
            embedding_model=llm_app_components["embedding_model"],
            k_results=5 # Retrieve a decent number of docs for context
        )
        retrieved_docs = retriever.invoke(query_text)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context
    except Exception as e:
        print(f"Error retrieving context from '{collection_key}': {e}")
        return ""