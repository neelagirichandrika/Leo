
import os
import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ollama_engine import OllamaEngine
from backend.gemini_engine import GeminiEngine
from backend.groq_engine import GroqEngine
from backend.hybrid_engine import HybridEngine

from backend.tavily_search import TavilySearchEngine
from backend.knowledge_engine import LEOKnowledgeEngine


# ==========================================
# LEO CONFIGURATION
# ==========================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("leo")

engine = None
ai_engine = None
search_engine = None

provider = os.getenv("LEO_PROVIDER", "ollama").lower().strip()
model_name = None
web_search_enabled = False

# Temporary conversation memory
conversation_history = {}

MAX_HISTORY_MESSAGES = 20


# ==========================================
# MODEL LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global engine, ai_engine, search_engine
    global provider, model_name, web_search_enabled

    if provider == "hybrid":

        ai_engine = HybridEngine()
        model_name = "Gemini primary + Groq fallback"

    elif provider == "groq":

        ai_engine = GroqEngine(
            model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        )
        model_name = ai_engine.model

    elif provider == "gemini":

        ai_engine = GeminiEngine(
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        )
        model_name = ai_engine.model

    elif provider == "ollama":

        ai_engine = OllamaEngine(
            model=os.getenv("OLLAMA_MODEL", "qwen3.5:4b"),
            base_url=os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434"
            )
        )
        model_name = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")

    else:
        raise RuntimeError(f"Unsupported LEO_PROVIDER: {provider}")

    # Initialize web research if configured.

    if os.getenv("TAVILY_API_KEY"):

        search_engine = TavilySearchEngine()

        engine = LEOKnowledgeEngine(
            ai_engine=ai_engine,
            search_engine=search_engine
        )

        web_search_enabled = True

    else:

        engine = ai_engine
        search_engine = None
        web_search_enabled = False

        logger.warning("Web search is disabled.")

    logger.info("LEO initialized.")
    logger.info("Provider: %s", provider)
    logger.info("Model: %s", model_name)

    yield

    engine = None
    ai_engine = None
    search_engine = None
    conversation_history.clear()
    web_search_enabled = False

    logger.info("LEO stopped.")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="LEO AI API",
    description="LEO AI assistant with conversation memory.",
    version="7.0.0",
    lifespan=lifespan
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://neelagirichandrika.github.io"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"]
)


# ==========================================
# REQUEST MODELS
# ==========================================

class ChatRequest(BaseModel):

    message: str = Field(min_length=1, max_length=500)

    conversation_id: str | None = Field(
        default=None,
        max_length=100
    )

    max_new_tokens: int = Field(
        default=150,
        ge=1,
        le=1000
    )

    temperature: float = Field(
        default=0.4,
        gt=0,
        le=2.0
    )


class ChatResponse(BaseModel):

    assistant: str
    response: str
    conversation_id: str


# ==========================================
# HOME
# ==========================================

@app.get("/")
def home():

    return {
        "application": "LEO AI",
        "status": "running",
        "intelligence_engine": provider,
        "model": model_name,
        "web_search_enabled": web_search_enabled,
        "conversation_memory": True
    }


# ==========================================
# HEALTH
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": ai_engine is not None,
        "intelligence_engine": provider,
        "model": model_name,
        "web_search_enabled": web_search_enabled,
        "active_conversations": len(conversation_history)
    }


# ==========================================
# CHAT
# ==========================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="LEO is not initialized."
        )

    conversation_id = request.conversation_id or str(uuid.uuid4())

    history = conversation_history.setdefault(
        conversation_id, []
    )

    # Construct conversation context.

    context = ""

    for item in history:
        role = "User" if item["role"] == "user" else "LEO"
        context += f"{role}: {item['content']}\n"

    prompt = (
        "You are LEO, a helpful AI assistant. "
        "Use the conversation history to understand context. "
        "Remember relevant details from previous messages in this conversation. "
        "Answer the latest user message directly.\n\n"
        f"Conversation history:\n{context}\n"
        f"User: {request.message}\n"
        "LEO:"
    )

    try:

        response = await engine.generate_response(
            prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )

        # Save messages only after successful generation.

        history.append({
            "role": "user",
            "content": request.message
        })

        history.append({
            "role": "assistant",
            "content": response
        })

        # Retain only the latest 20 messages.

        conversation_history[conversation_id] = history[
            -MAX_HISTORY_MESSAGES:
        ]

        return ChatResponse(
            assistant="LEO",
            response=response,
            conversation_id=conversation_id
        )

    except ValueError as error:

        raise HTTPException(status_code=400, detail=str(error))

    except RuntimeError as error:

        logger.error("LEO request failed: %s", error)

        raise HTTPException(
            status_code=503,
            detail="LEO is temporarily unavailable."
        )


# ==========================================
# CLEAR CONVERSATION
# ==========================================

@app.delete("/conversations/{conversation_id}")
def clear_conversation(conversation_id: str):

    if conversation_id not in conversation_history:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found."
        )

    del conversation_history[conversation_id]

    return {
        "status": "success",
        "message": "Conversation cleared."
    }
