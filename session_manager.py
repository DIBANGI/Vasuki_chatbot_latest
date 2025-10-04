# session_manager.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict

class ProductSearchState(BaseModel):
    """Stores the user's current search preferences."""
    category: Optional[str] = None
    subcategory: Optional[str] = None
    stone: Optional[str] = None
    color: Optional[str] = None
    finish: Optional[str] = None
    min_price: Optional[float] = Field(default=None, alias="price_min")
    max_price: Optional[float] = Field(default=None, alias="price_max")

# In-memory session store (can be replaced with Redis for production)
user_sessions: Dict[str, ProductSearchState] = {}

def get_or_create_session(session_id: str) -> ProductSearchState:
    """Retrieves or creates a new session state."""
    if session_id not in user_sessions:
        user_sessions[session_id] = ProductSearchState()
    return user_sessions[session_id]

def update_session(session_id: str, updates: Dict):
    """Updates the session state with new slot values."""
    session = get_or_create_session(session_id)
    new_state = session.model_dump()
    new_state.update(updates)
    user_sessions[session_id] = ProductSearchState(**new_state)
    return user_sessions[session_id]

def clear_session(session_id: str):
    """Clears the search state for a session."""
    if session_id in user_sessions:
        del user_sessions[session_id]