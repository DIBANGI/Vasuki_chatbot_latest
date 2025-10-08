# main.py

import os
import json
from typing import Dict, Optional, List, TypedDict
import time

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

import config
import llm_utils
import query_processor

app = FastAPI(title="VASUKI Jewelry Chatbot API")

# --- Global variable to store initialized components ---
llm_application_components = {}


# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Static Files ---
try:
    app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
except RuntimeError as e:
    print(f"Warning: Could not mount static files from '{config.STATIC_DIR}': {e}. Ensure directory exists.")


# --- Application Startup Event ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes LLM, chains, and other necessary components when the application starts.
    """
    print("Application starting up...")
    try:
        print("Initializing embedding model...")
        embedding_model = llm_utils.get_embedding_model()
        llm_application_components["embedding_model"] = embedding_model
        print("Embedding model initialized.")

        print("Initializing Groq LLM...")
        llm = llm_utils.get_groq_chat_model()
        llm_application_components["llm"] = llm
        print("Groq LLM initialized.")

        print("Creating LLM chains...")
        llm_chains = llm_utils.initialize_llm_chains(llm, embedding_model)
        llm_application_components["llm_chains"] = llm_chains
        print("LLM chains created.")

        print("Application components initialized successfully.")
    except Exception as e:
        print(f"FATAL ERROR during application startup: {e}")
        llm_application_components["initialization_error"] = str(e)


# --- Pydantic Models ---
class QueryRequest(BaseModel):
    query: str
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    response: str
    conversation_id: str


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    """Serves the main HTML page for the chatbot interface."""
    index_html_path = os.path.join(config.STATIC_DIR, "index.html")
    if not os.path.exists(index_html_path):
        index_html_path = os.path.join("templates", "index.html")

    try:
        with open(index_html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Chatbot</h1><p>Error: Frontend interface not found.</p>", status_code=500)


@app.post("/query", response_model=QueryResponse)
async def handle_query_api(request: QueryRequest):
    """Handles user queries sent via HTTP POST."""
    if "initialization_error" in llm_application_components:
        return JSONResponse(
            status_code=503,
            content={"response": f"System is currently unavailable: {llm_application_components['initialization_error']}",
                     "conversation_id": request.conversation_id or f"error_session_{os.urandom(4).hex()}"}
        )
    if not llm_application_components.get("llm_chains"):
         return JSONResponse(
            status_code=503,
            content={"response": "System is not ready. LLM components not initialized.",
                     "conversation_id": request.conversation_id or f"error_session_{os.urandom(4).hex()}"}
        )

    conv_id = request.conversation_id or f"session_http_{os.urandom(8).hex()}"

    bot_response_text = query_processor.process_query(
        query_text=request.query,
        llm_app_components=llm_application_components,
        conversation_id=conv_id
    )

    return QueryResponse(response=bot_response_text, conversation_id=conv_id)

# --- WebSocket Communication ---
active_connections: Dict[str, WebSocket] = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handles real-time communication with clients via WebSockets."""
    await websocket.accept()
    connection_id_ws = f"ws_conn_{os.urandom(8).hex()}"
    active_connections[connection_id_ws] = websocket

    ws_conv_id: Optional[str] = None

    if "initialization_error" in llm_application_components:
        await websocket.send_json({"response": f"System is currently unavailable: {llm_application_components['initialization_error']}", "conversation_id": "error_session"})
        await websocket.close(code=1011)
        active_connections.pop(connection_id_ws, None)
        return

    if not llm_application_components.get("llm_chains"):
        await websocket.send_json({"response": "System is not ready.", "conversation_id": "error_session"})
        await websocket.close(code=1011)
        active_connections.pop(connection_id_ws, None)
        return

    try:
        while True:
            data = await websocket.receive_json()
            query_text = data.get("query", "")
            client_conv_id = data.get("conversation_id")

            if not ws_conv_id:
                ws_conv_id = client_conv_id or f"session_ws_{os.urandom(8).hex()}"

            bot_response_text = query_processor.process_query(
                query_text=query_text,
                llm_app_components=llm_application_components,
                conversation_id=ws_conv_id
            )

            await websocket.send_json({"response": bot_response_text, "conversation_id": ws_conv_id})

    except WebSocketDisconnect:
        print(f"WebSocket client disconnected: {connection_id_ws}")
    except Exception as e:
        print(f"WebSocket error for {connection_id_ws}: {e}")
    finally:
        active_connections.pop(connection_id_ws, None)


@app.get("/health")
async def health_check():
    """Provides a health check endpoint."""
    if "initialization_error" in llm_application_components:
        return {"status": "unhealthy", "reason": llm_application_components["initialization_error"]}
    if not llm_application_components.get("llm_chains"):
        return {"status": "unhealthy", "reason": "LLM components not initialized"}
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8001")))