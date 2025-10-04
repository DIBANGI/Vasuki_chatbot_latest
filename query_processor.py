# query_processor.py
from typing import List, Dict, Any
import json
import re
from langchain.docstore.document import Document

import config
import llm_utils
import vector_store_utils
from session_manager import get_or_create_session, update_session, clear_session

# --- NEW: Hybrid Retrieval Logic ---
def hybrid_product_retrieval(query: str, session_id: str, llm_app_components: Dict) -> List[Dict[str, Any]]:
    """
    Performs hybrid retrieval (semantic + keyword) with metadata filtering.
    """
    session_state = get_or_create_session(session_id)
    
    # 1. Keyword Search (for SKUs)
    sku_match = re.search(r'([A-Z0-9]{5,})', query.upper())
    if sku_match:
        sku = sku_match.group(1)
        # Use ChromaDB's metadata filtering to find the exact SKU
        retriever = vector_store_utils.get_langchain_chroma_retriever(
            collection_name="shopify_products",
            embedding_model=llm_app_components["embedding_model"],
            k_results=1,
            filter_criteria={"sku": sku}
        )
        docs = retriever.invoke(query)
        if docs:
            return [doc.metadata for doc in docs]

    # 2. Semantic Search with Metadata Filtering
    filters = {}
    if session_state.category:
        filters["product_type"] = session_state.category
    if session_state.min_price is not None or session_state.max_price is not None:
        price_filter = {}
        if session_state.min_price is not None:
            price_filter["$gte"] = session_state.min_price
        if session_state.max_price is not None:
            price_filter["$lte"] = session_state.max_price
        filters["price"] = price_filter

    retriever = vector_store_utils.get_langchain_chroma_retriever(
        collection_name="shopify_products",
        embedding_model=llm_app_components["embedding_model"],
        k_results=10, # Retrieve more to filter down
        filter_criteria=filters if filters else None
    )
    
    retrieved_docs = retriever.invoke(query)
    
    # Simple reranking: prioritize docs with query keywords in the title
    query_words = set(query.lower().split())
    
    # Convert retrieved_docs to a list of metadata dictionaries
    results = [doc.metadata for doc in retrieved_docs]

    # You could add a more sophisticated reranking step here if needed
    
    return results[:5] # Return top 5 results

# --- REFACTORED: process_query ---
def process_query(
    query_text: str,
    conversation_history: List[Dict[str, str]],
    llm_app_components: dict,
    conversation_id: str
) -> str:
    """
    Processes user query using the new pipeline: Intent -> Slot Filling -> Retrieval -> Generation.
    """
    llm_chains = llm_app_components["llm_chains"]

    # --- 1. Intent Classification ---
    intent = llm_utils.classify_intent_with_llm(query_text, llm_chains)
    print(f"Classified Intent: {intent}")

    # --- 2. Process Based on Intent ---
    if intent == "greeting":
        return "Hello! I'm Vasuki, your personal jewelry assistant. How can I help you find the perfect piece today?"
    
    elif intent in ["return_policy", "shipping_policy", "privacy_policy"]:
        context = vector_store_utils.get_context_from_vector_store(query_text, "policies", llm_app_components)
        chain_input = {"question": query_text, "history": conversation_history, "context": context}
        return llm_chains["policy"].invoke(chain_input)

    elif intent == "general_faq":
        context = vector_store_utils.get_context_from_vector_store(query_text, "faqs", llm_app_components)
        chain_input = {"question": query_text, "history": conversation_history, "context": context}
        return llm_chains["faq"].invoke(chain_input)

    elif intent == "product_query":
        # --- 3. Slot Filling ---
        session_state = get_or_create_session(conversation_id)
        slot_filling_chain = llm_chains["slot_filler"]
        
        # Rewrite query for better retrieval if it's a follow-up
        if conversation_history:
            rewritten_query = llm_chains["query_rewriter"].invoke({
                "question": query_text,
                "history": conversation_history
            })
            print(f"Rewritten Query: {rewritten_query}")
        else:
            rewritten_query = query_text

        # Extract slots
        extracted_slots_str = slot_filling_chain.invoke({
            "question": rewritten_query, 
            "current_slots": session_state.model_dump_json()
        })
        print(f"Extracted Slots (raw): {extracted_slots_str}")
        try:
            # The LLM should return a clean JSON string
            updates = json.loads(extracted_slots_str)
            session_state = update_session(conversation_id, updates)
            print(f"Updated Session State: {session_state.model_dump()}")
        except json.JSONDecodeError:
            print("Warning: Could not decode JSON from slot filler.")

        # --- 4. Hybrid Retrieval ---
        retrieved_products = hybrid_product_retrieval(rewritten_query, conversation_id, llm_app_components)
        
        if not retrieved_products:
            clear_session(conversation_id) # Clear session if no results are found
            return "I'm sorry, I couldn't find any products that match your criteria. Please try a different search."

        # --- 5. Dynamic Prompt and Generation ---
        qa_chain = llm_chains["shopify_qa"]
        
        # Format the context for the LLM
        context_for_llm = "\n---\n".join([
            f"Product: {p.get('product_title', 'N/A')}\nSKU: {p.get('sku', 'N/A')}\nPrice: ₹{p.get('price', 0):.2f}\nTags: {p.get('tags', 'N/A')}"
            for p in retrieved_products
        ])
        
        chain_input = {
            "input_documents": [Document(page_content=context_for_llm)],
            "question": query_text,
            "history": conversation_history
        }
        
        response = qa_chain.invoke(chain_input)
        return response.get("output_text", "I had trouble generating a response. Please try again.")

    else: # Fallback
        return "I'm sorry, I'm not sure how to help with that. I can assist with product searches or questions about our policies."