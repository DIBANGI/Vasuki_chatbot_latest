# llm_utils.py

import os
import re
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains.question_answering import load_qa_chain
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
INTENT_CLASSIFICATION_SYSTEM_PROMPT = """
Your single task is to classify the user's intent based on their latest message.
Respond with ONLY ONE of the following category names. Do not add explanations or punctuation.
Analyze ONLY the "Customer Question". Ignore the "Conversation History".

**CATEGORIES:**
- **product_query**: For any question about products, including searches, details, price, availability, or recommendations.
- **return_policy**: For questions about returns, refunds, or exchanges.
- **shipping_policy**: For questions about shipping, delivery, or tracking.
- **privacy_policy**: For questions about data privacy or personal information.
- **general_faq**: For general questions about the company, materials, or contact info.
- **greeting**: For simple greetings like "hello", "hi", "good morning".
- **compliment**: For compliments about the products or service.
- **other**: If the intent does not fit any other category.
"""

QUERY_REWRITING_SYSTEM_PROMPT = """
You are an expert at rephrasing questions. Your task is to rewrite a follow-up question from a user into a self-contained, standalone question.
Use the conversation history to understand the context of the follow-up question.
The rewritten question should be concise and optimized for a vector database search.

**Example 1:**
History:
User: "Do you have any gold necklaces?"
Assistant: "Yes, we have several..."
Follow-up Question: "what about under 10000?"
Standalone Question: "gold necklaces under 10000"

**Example 2:**
History:
User: "show me bangle options"
Assistant: "Here are some bangles..."
Follow-up Question: "only the silver ones"
Standalone Question: "silver bangles"

Respond ONLY with the rewritten standalone question. Do not add any other text.
"""

SLOT_FILLING_SYSTEM_PROMPT = """
Your task is to extract product search criteria (slots) from the user's query and update a JSON object of the current search state.
You must adhere to the following rules:
1.  Analyze the "USER QUERY" to identify any of these slots: `category`, `stone`, `color`, `finish`, `min_price`, `max_price`.
2.  Look at the "CURRENT SLOTS" to see what the user has already specified.
3.  If the user provides a new value for an existing slot, UPDATE it.
4.  If the user provides a value for a new slot, ADD it.
5.  If the user's query implies clearing a filter (e.g., "actually, any color is fine"), set that slot's value to `null`.
6.  Price Extraction: If you see "under 5000", set `max_price: 5000`. If you see "over 3000", set `min_price: 3000`. If you see "between 2000 and 7000", set `min_price: 2000` and `max_price: 7000`.
7.  Your final output MUST be ONLY the updated JSON object. Do not include any other text, reasoning, or formatting.
"""

SHOPIFY_QA_SYSTEM_PROMPT = """
You are VASUKI, a friendly and expert jewelry assistant for shopvasuki.com.
Your goal is to provide a conversational, helpful, and accurate response based ONLY on the CONTEXT of retrieved products and the CONVERSATION HISTORY.

**CRITICAL RULES:**
1.  **STICK TO THE CONTEXT**: Only mention products, SKUs, prices, or details present in the CONTEXT section. Never invent information or mention products not listed.
2.  **HANDLE NO RESULTS**: If the CONTEXT is empty or contains no product information, your ONLY response must be: "I'm sorry, I couldn't find any products that match your search. Would you like to try different criteria?"
3.  **CONVERSATIONAL TONE**: Be warm and helpful. Instead of just listing data, frame it naturally. For example: "I found a beautiful necklace that might be perfect for you! It's the..."
4.  **SUMMARIZE AND RECOMMEND**: If there are multiple products in the context, briefly summarize them and perhaps recommend one based on the user's query. Don't just list all of them mechanically.
5.  **OUTPUT FORMAT**: When you present a product, always include its **Name**, **SKU**, and **Price** (formatted as ₹XX.XX).

---
CONVERSATION HISTORY:
{history_string}
---
CONTEXT (Retrieved Products):
{context}
---
USER QUESTION: {question}
---
VASUKI'S HELPFUL ANSWER:"""

POLICY_SYSTEM_PROMPT = """You are a customer service specialist for VASUKI Jewelry Store. Use ONLY the Policy Context and Conversation History to answer the user's question clearly and accurately."""
FAQ_SYSTEM_PROMPT = """You are a friendly jewelry expert at VASUKI. Answer the customer's question based ONLY on the provided FAQ Context and the Conversation History."""

# --- User Templates ---
INTENT_USER_TEMPLATE = "Customer Question: {question}"
QUERY_REWRITING_USER_TEMPLATE = "Conversation History:\n{history_string}\n\nFollow-up Question: {question}"
SLOT_FILLING_USER_TEMPLATE = "CURRENT SLOTS:\n{current_slots}\n\nUSER QUERY:\n{question}"
CONTEXT_USER_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"

# --- Chain Creation Functions ---
def get_shopify_product_qa_chain(llm: ChatGroq):
    """Creates the dedicated RAG chain for answering product questions."""
    prompt = PromptTemplate(
        template=SHOPIFY_QA_SYSTEM_PROMPT,
        input_variables=["context", "question", "history_string"]
    )
    
    def prepare_history_string(input_dict: Dict):
        """Prepares history_string for the QA prompt."""
        raw_history = input_dict.get("history", [])
        history_str_parts = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in raw_history]
        input_dict["history_string"] = "\n".join(history_str_parts)
        return input_dict

    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)
    
    # Wrap the chain to include history string preparation
    return RunnableLambda(prepare_history_string) | qa_chain

def initialize_llm_chains(llm: ChatGroq, embedding_model):
    """Creates and returns a dictionary of all required Langchain chains."""

    def create_rag_chain(system_prompt_text, user_template_text):
        """A robust factory for creating chains that handle history."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt_text),
            ("placeholder", "{history}"),
            ("human", user_template_text),
        ])

        def prepare_chain_input(input_dict: Dict):
            """Prepares history messages and a simple history string."""
            history_messages = []
            raw_history = input_dict.get("history", [])
            for turn in raw_history:
                role = turn.get("role")
                if role == "user":
                    history_messages.append(HumanMessage(content=turn["content"]))
                elif role == "assistant":
                    history_messages.append(AIMessage(content=turn["content"]))
            input_dict["history"] = history_messages
            
            history_str_parts = [f"{turn['role'].capitalize()}: {turn['content']}" for turn in raw_history]
            input_dict["history_string"] = "\n".join(history_str_parts)
            return input_dict

        return RunnableLambda(prepare_chain_input) | prompt | llm | StrOutputParser()

    # --- Build all chains ---
    # ** THIS IS THE CORRECTED SECTION **
    slot_filler_prompt = ChatPromptTemplate.from_messages([
        ("system", SLOT_FILLING_SYSTEM_PROMPT),
        ("human", SLOT_FILLING_USER_TEMPLATE)
    ])
    slot_filler_chain = slot_filler_prompt | llm | StrOutputParser()

    chains = {
        "intent": create_rag_chain(INTENT_CLASSIFICATION_SYSTEM_PROMPT, INTENT_USER_TEMPLATE),
        "policy": create_rag_chain(POLICY_SYSTEM_PROMPT, CONTEXT_USER_TEMPLATE),
        "faq": create_rag_chain(FAQ_SYSTEM_PROMPT, CONTEXT_USER_TEMPLATE),
        "query_rewriter": create_rag_chain(QUERY_REWRITING_SYSTEM_PROMPT, QUERY_REWRITING_USER_TEMPLATE),
        "shopify_qa": get_shopify_product_qa_chain(llm),
        "slot_filler": slot_filler_chain # Correctly defined chain
    }
    print("✅ All LLM chains initialized.")
    return chains

def classify_intent_with_llm(query: str, llm_chains, history: List = []) -> str:
    """Classifies intent using the LLM. It's designed to focus on the current query."""
    try:
        # We pass an empty history to the intent classifier to keep it focused on the latest query.
        response = llm_chains["intent"].invoke({"question": query, "history": []})
        cleaned_intent = response.strip().lower().replace(".", "")
        
        valid_intents = [
            "product_query", "return_policy", "shipping_policy", 
            "privacy_policy", "general_faq", "greeting", "compliment", "other"
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
    """A fast, rule-based intent classifier as a fallback."""
    query_lower = query.lower()
    if any(kw in query_lower for kw in ["hello", "hi", "hey", "good morning", "namaste"]):
        return "greeting"
    if any(kw in query_lower for kw in ["return", "refund", "exchange"]):
        return "return_policy"
    if any(kw in query_lower for kw in ["shipping", "delivery", "track"]):
        return "shipping_policy"
    if any(kw in query_lower for kw in ["privacy", "data", "personal information"]):
        return "privacy_policy"
    if any(kw in query_lower for kw in ["thank you", "great", "awesome", "love it"]):
        return "compliment"
    if any(kw in query_lower for kw in ["product", "item", "jewelry", "ring", "necklace", "price", "cost", "buy", "find", "show me"]):
        return "product_query"
    return "general_faq"