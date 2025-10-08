# llm_utils.py

import os
import re
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate # MODIFIED: Added PromptTemplate
from langchain.chains.question_answering import load_qa_chain # --- NEW ---
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from typing import List, Dict

import config
import vector_store_utils

# --- LLM and Embeddings Initialization ---
def get_embedding_model():
    """Initializes and returns the HuggingFace embedding model."""
    print(f"Initializing embedding model: {config.EMBEDDING_MODEL_ID}")
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_ID)

def get_groq_chat_model():
    """Initializes and returns the ChatGroq model."""
    print(f"Initializing Groq LLM: {config.LLM_MODEL_ID}")
    if not config.GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

    return ChatGroq(
        temperature=config.LLM_TEMPERATURE,
        model_name=config.LLM_MODEL_ID,
        groq_api_key=config.GROQ_API_KEY,
        max_tokens=config.LLM_MAX_NEW_TOKENS,
    )

# --- Prompt Templates ---
DEFAULT_SYSTEM_PROMPT = """You are an AI customer service representative for VASUKI Jewelry Store.
Your role is to help customers with their queries in a friendly, professional, and concise manner.
Always maintain a polite and helpful tone. Only provide information that's available in the provided context or your general knowledge if no context is given.
If information is not available, politely say so and offer to help with something else.
Format prices and measurements consistently (e.g., Price: ₹XX.XX).
Do not make up information. If the answer is not in the context, say you don't know.
Directly answer the user's query without any introductory phrases about your own response.
Do not include emotive expressions or actions in asterisks (e.g., *smiles*).
"""

# --- NEW: Dedicated Shopify QA Prompt ---
SHOPIFY_QA_SYSTEM_PROMPT = """
You are an expert product assistant for "shopvasuki.com".
Your task is to answer user questions based ONLY on the product data provided in the CONTEXT.

**CRITICAL RULES:**

1.  **PRICE FILTERING:** If the user specifies a price limit (e.g., "under 5000"), you MUST ONLY show products where the price is less than that number. You MUST silently ignore any products from the context that are over the price limit. If no products meet the price criteria, you MUST respond with: "I'm sorry, I couldn't find any products in that price range. Would you like me to search for something else?"

2.  **DATA SOURCE:** You can ONLY use information from the CONTEXT. Do not make up any details.

3.  **OUTPUT FORMAT:** For each product you present, you MUST follow this format exactly:
    - **SKU:** [SKU of the product]
    - **Product:** [Product Name]
    - **Description:** [Description of the product]
    - **Price:** ₹[Price]

**CONTEXT:**
{context}

**USER QUESTION:** {question}

**HELPFUL ANSWER:**"""
# --- END NEW ---

PRODUCT_QUERY_SYSTEM_PROMPT = """You are a product data presenter for VASUKI Jewelry Store.
Your ONLY task is to present the product information given to you in the 'Database Query Results'. You must not alter, add to, or invent any information.

**CRITICAL RULES:**
1.  **EXCLUSIVE DATA SOURCE:** You MUST ONLY use the data within the "Database Query Results" section. Do NOT use any other context or prior knowledge.
2.  **NO RESULTS CASE:** If the "Database Query Results" section is empty or contains no products, your ONLY permitted response is: "I'm sorry, I couldn't find any products that match your description. Can I help with anything else?" Do NOT say anything else.
3.  **NO FABRICATION:** You are strictly forbidden from inventing product details, SKUs, prices, specifications, or availability. If a detail is not in the provided data, you must omit it. You do not have access to the full database.
4.  **PRESENTATION:** Present the information clearly. Do not add conversational filler before the product list. Start directly with the products.

This is your single source of truth. Adhere to it strictly.
"""

POLICY_SYSTEM_PROMPT = """You are a customer service specialist for VASUKI Jewelry Store, responsible for explaining our policies clearly and accurately.
Use ONLY information provided in the context to answer policy-related questions.
Structure your response in a clear, easy-to-understand format. Highlight key points.
If specific details aren't available in the context, acknowledge this and suggest they contact customer support directly.
Directly answer the user's query without any introductory phrases.
"""

FAQ_SYSTEM_PROMPT = """You are a friendly jewelry expert at VASUKI Jewelry Store.
Answer the customer's question based ONLY on the provided FAQ context.
Keep responses conversational and natural.
If the exact question isn't in the FAQ context, state that the information is not available in the FAQs and offer to help with other questions.
Directly answer the user's query.
"""

INTENT_CLASSIFICATION_SYSTEM_PROMPT = """TASK: Your single-minded purpose is to classify the user's intent.
Analyze the user's question and respond with ONLY ONE of the following category names.

RULES:
- Your response MUST be a single word from the list below.
- Do NOT add any pleasantries, explanations, or punctuation.
- Analyze ONLY the LATEST customer question.

CATEGORIES:
- product_query
- return_policy
- shipping_policy
- privacy_policy
- general_faq
- greeting
- other
"""

REFINEMENT_SYSTEM_PROMPT = """You are a professional customer service specialist.
Refine the DRAFT RESPONSE to be conversational and customer-friendly.
The final response should directly address the user's query using ONLY the factual information present in the DRAFT RESPONSE.
DO NOT introduce any new product details, SKUs, prices, or other factual information not already in the draft.
Remove any technical language and format for easy reading. Add a polite closing if appropriate.
"""

QUERY_REWRITING_SYSTEM_PROMPT = """You are an expert at rephrasing questions. Your task is to rewrite a follow-up question from a user into a self-contained, standalone question.
The rewritten question should be concise and optimized for a vector database search.

Example 1:
Follow-up Question: "what about under 10000?"
Standalone Question: "gold necklaces under 10000"

Example 2:
Follow-up Question: "only the silver ones"
Standalone Question: "silver bangles"

Example 3:
Follow-up Question: "give me more details"
Standalone Question: "full details for SPSLB0004"

Respond ONLY with the rewritten standalone question. Do not add any other text.
"""


# --- User Templates ---
GENERAL_USER_TEMPLATE = "Context:\n{context}\n\nCurrent Question: {question}"
PRODUCT_USER_TEMPLATE = "Database Query Results:\n{db_query_results}\n\nContext (from vector store, for reference ONLY):\n{context}\n\nCurrent Question: {question}"
POLICY_USER_TEMPLATE = "Policy Context:\n{context}\n\nCurrent Question: {question}"
FAQ_USER_TEMPLATE = "FAQ Context:\n{context}\n\nCurrent Question: {question}"
INTENT_USER_TEMPLATE = "Customer Question: {question}"
REFINEMENT_USER_TEMPLATE = "Original Customer Question: {question}\n\nDraft Response to Refine:\n{draft_response}"
QUERY_REWRITING_USER_TEMPLATE = "Follow-up Question: {question}"


# --- NEW: Function to create the dedicated Shopify QA chain ---
def get_shopify_product_qa_chain(llm: ChatGroq):
    """
    Creates and returns a LangChain chain specifically for answering questions
    about Shopify products based on provided context.
    """
    prompt = PromptTemplate(
        template=SHOPIFY_QA_SYSTEM_PROMPT,
        input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    return chain
# --- END NEW ---


# --- Langchain Chains ---
# --- MODIFIED: Renamed from create_llm_chains to initialize_llm_chains for clarity ---
def initialize_llm_chains(llm: ChatGroq, embedding_model):
    """Creates and returns a dictionary of Langchain chains."""

    def create_chain(system_prompt_text, user_template_text):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("human", user_template_text),
        ])
        chain = (
            prompt
            | llm
            | StrOutputParser()
        )
        return chain

    chains = {
        "intent": create_chain(INTENT_CLASSIFICATION_SYSTEM_PROMPT, INTENT_USER_TEMPLATE),
        "general": create_chain(DEFAULT_SYSTEM_PROMPT, GENERAL_USER_TEMPLATE),
        "product": create_chain(PRODUCT_QUERY_SYSTEM_PROMPT, PRODUCT_USER_TEMPLATE),
        "policy": create_chain(POLICY_SYSTEM_PROMPT, POLICY_USER_TEMPLATE),
        "faq": create_chain(FAQ_SYSTEM_PROMPT, FAQ_USER_TEMPLATE),
        "refinement": create_chain(REFINEMENT_SYSTEM_PROMPT, REFINEMENT_USER_TEMPLATE),
        "query_rewriter": create_chain(QUERY_REWRITING_SYSTEM_PROMPT, QUERY_REWRITING_USER_TEMPLATE),
        # --- NEW: Add the dedicated Shopify QA chain to the dictionary ---
        "shopify_qa": get_shopify_product_qa_chain(llm)
    }
    print("All LLM chains initialized.")
    return chains

def classify_intent_with_llm(query: str, llm_chains) -> str:
    """Classifies intent using the LLM. Based on current query ONLY."""
    if "intent" not in llm_chains:
        return "other"
    try:
        response = llm_chains["intent"].invoke({"question": query})
        cleaned_intent = response.strip().lower().replace(".", "")

        valid_intents = [
            "product_query", "return_policy", "shipping_policy",
            "privacy_policy", "general_faq", "greeting", "other"
        ]

        if cleaned_intent in valid_intents:
            return cleaned_intent

        for intent in valid_intents:
            if intent in cleaned_intent:
                return intent

        return "other"
    except Exception as e:
        print(f"Error during LLM intent classification: {e}")
        return "other"

def classify_intent_rules(query: str) -> str:
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["hello", "hi", "hey", "greetings"]):
        return "greeting"
    if any(kw in query_lower for kw in ["return", "refund", "exchange"]):
        return "return_policy"
    if any(kw in query_lower for kw in ["shipping", "delivery", "track"]):
        return "shipping_policy"
    if any(kw in query_lower for kw in ["privacy", "data", "personal information"]):
        return "privacy_policy"
    if any(kw in query_lower for kw in ["product", "item", "jewelry", "ring", "necklace", "price", "cost", "buy", "find"]):
        return "product_query"
    return "general_faq"