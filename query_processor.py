# query_processor.py

from typing import List, Dict
import config
import llm_utils
import database_utils
import vector_store_utils
import re
import random
# --- NEW: Add this import ---
from langchain.docstore.document import Document

# In-memory store for product recommendation sessions
product_recommendation_sessions: Dict[str, Dict] = {}

def get_context_from_vector_store(query_text: str, collection_key: str, llm_app_components):
    """Helper to retrieve context from a specific vector store collection."""
    if not llm_app_components or "embedding_model" not in llm_app_components:
        print("Error: Embedding model not available in llm_app_components.")
        return ""

    collection_name_map = {
        "faqs": config.CHROMA_COLLECTION_FAQS,
        "policies": config.CHROMA_COLLECTION_POLICIES,
        # --- MODIFIED: Point to the new Shopify collection ---
        "products": "shopify_products",
    }
    if collection_key not in collection_name_map:
        print(f"Warning: Invalid collection key '{collection_key}' for context retrieval.")
        return ""

    try:
        retriever = vector_store_utils.get_langchain_chroma_retriever(
            collection_name=collection_name_map[collection_key],
            embedding_model=llm_app_components["embedding_model"],
            # --- MODIFIED: Get more context for better answers ---
            k_results=5
        )
        retrieved_docs = retriever.invoke(query_text)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        return context
    except Exception as e:
        print(f"Error retrieving context from '{collection_key}': {e}")
        return ""

def query_policy_context(query_text: str, policy_type: str, llm_app_components) -> str:
    """Queries the policy vector store for relevant context."""
    return get_context_from_vector_store(query_text, "policies", llm_app_components)


def query_faq_context(query_text: str, llm_app_components) -> str:
    """Queries the FAQ vector store for relevant context."""
    return get_context_from_vector_store(query_text, "faqs", llm_app_components)


def process_query(
    query_text: str,
    conversation_history: List[Dict[str, str]],
    llm_app_components: dict,
    conversation_id: str
) -> str:
    """
    Processes the user query: classifies intent, retrieves context, generates and refines response.
    """
    query_lower = query_text.lower().strip()

    # --- Initial Checks ---
    if not llm_app_components or "llm_chains" not in llm_app_components:
        return "I'm sorry, the system is not fully initialized. Please try again later."
    llm_chains = llm_app_components["llm_chains"]

    # --- 1. Handle "Show More" Intent ---
    # This logic is tied to the old database method, can be removed if you fully switch to Shopify QA
    if conversation_id in product_recommendation_sessions and any(word in query_lower for word in ["yes", "more", "show more", "sure", "ok", "give me more"]):
        session = product_recommendation_sessions[conversation_id]
        results = session["results"]
        start_index = session["index"]

        if start_index < len(results):
            formatted_products = database_utils.format_product_results(results, start_index=start_index, batch_size=3)
            session["index"] += 3
            final_response = formatted_products
            if session["index"] < len(results):
                final_response += "\nWould you like to see more?"
            else:
                final_response += "\nThose are all the recommendations I have for now."
                del product_recommendation_sessions[conversation_id] # End of session
        else:
            final_response = "There are no more products to show from your previous search."
            del product_recommendation_sessions[conversation_id] # End of session
        return final_response.strip()

    # --- 2. Clear previous product session for any new query ---
    if conversation_id in product_recommendation_sessions:
        del product_recommendation_sessions[conversation_id]

    # --- 3. Handle Greetings ---
    # (This section remains unchanged)
    if "how are you" in query_lower:
        return random.choice([
            "I'm doing great, thank you! How can I assist you today?",
            "I'm just a bot, but I'm ready to help! What can I do for you?",
            "Feeling helpful! Thanks for asking. What can I get for you?"
        ])
    greetings = {
        "good morning": ["Good morning! How can I help you today?", "Good morning! Hope you have a great day. What can I assist you with?", "A very good morning to you! What are you looking for today?"],
        "good evening": ["Good evening! How may I assist you?", "Good evening! I hope you're having a pleasant evening. How can I help?"],
        "namaste": ["Namaste! How can I help you today?", "Namaste! Welcome to Vasuki. What can I do for you?"],
        "hello": ["Hello! This is Vasuki, your jewelry assistant. How can I help?", "Hi there! I'm Vasuki. Ask me anything about our products or policies."],
        "hi": ["Hi! I'm Vasuki, ready to help with your E-commerce questions.", "Hey! Vasuki here. What can I do for you?"],
        "hey": ["Hey there! How can I assist you?", "Hey! I'm Vasuki. Let me know what you need."]
    }
    for trigger, responses in greetings.items():
        if trigger in query_lower:
            return random.choice(responses)

    print(f"Processing query: '{query_text}' with history length: {len(conversation_history)}")

    # --- 4. Intent Classification ---
    intent = llm_utils.classify_intent_with_llm(query_text, llm_chains)
    if intent not in ["product_query", "return_policy", "shipping_policy", "privacy_policy", "general_faq", "greeting"]:
        intent = llm_utils.classify_intent_rules(query_text)
    print(f"Final classified intent: {intent}")

    # --- 5. Process Based on Intent ---
    draft_response = ""
    try:
        if intent == "greeting": # Fallback greeting
            return random.choice([
                "Hello! VASUKI is a premier jewelry company. How can I assist you with our collections or services today?",
                "Greetings! Welcome to VASUKI. How may I help you?",
                "Hi! You've reached VASUKI. Let me know if you have any questions about our jewelry."
            ])

        # --- MODIFIED: Replaced the entire product_query block ---
        elif intent == "product_query":
            print("Handling product query using Shopify QA chain.")
            
            # 1. Get relevant product context from our new Shopify vector store
            context = get_context_from_vector_store(query_text, "products", llm_app_components)
            
            if not context:
                draft_response = "I'm sorry, I couldn't find any products that match your description. You can browse our full collection at https://shopvasuki.com/"
            else:
                # 2. Prepare the input for our dedicated Shopify QA chain
                # The chain expects a list of Langchain Document objects for the context
                chain_input = {
                    "input_documents": [Document(page_content=context)], 
                    "question": query_text
                }
                
                # 3. Run the chain to get a natural language answer
                # The key "output_text" is the default for load_qa_chain
                draft_response = llm_chains["shopify_qa"].invoke(chain_input).get("output_text", "Sorry, I had trouble generating a response.")
        # --- END MODIFICATION ---

        elif intent.endswith("_policy"):
            policy_type_key = intent.split('_')[0]
            chain_input = {"question": query_text, "history": conversation_history, "context": query_policy_context(query_text, policy_type_key, llm_app_components)}
            draft_response = llm_chains["policy"].invoke(chain_input)

        elif intent == "general_faq":
            if "what is vasuki" in query_text.lower():
                draft_response = "VASUKI is a distinguished jewelry company, known for its exquisite craftsmanship and unique designs."
            else:
                chain_input = {"question": query_text, "history": conversation_history, "context": query_faq_context(query_text, llm_app_components)}
                draft_response = llm_chains["faq"].invoke(chain_input)
        
        else:
            draft_response = "I'm sorry, I'm not sure how to help with that. Can I assist with a product search or a policy question?"

    except Exception as e:
        print(f"Error during query processing pipeline for query '{query_text}': {e}")
        draft_response = "I'm sorry, I encountered an unexpected issue while processing your request. Please try again."

    # Final cleanup of the response string
    final_response = re.sub(r'[ \t]{2,}', ' ', draft_response).strip()
    return final_response