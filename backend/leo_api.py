
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.ollama_engine import OllamaEngine
from backend.gemini_engine import GeminiEngine
from backend.groq_engine import GroqEngine


# ==========================================
# LEO - CONFIGURABLE INTELLIGENCE ENGINE
# ==========================================

engine = None
provider = os.getenv("LEO_PROVIDER", "groq").lower().strip()
model_name = None


# ==========================================
# MODEL LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, provider, model_name

    if provider == "groq":

        engine = GroqEngine(
            model=os.getenv(
                "GROQ_MODEL",
                "openai/gpt-oss-20b"
            )
        )

        model_name = engine.model

    elif provider == "gemini":

        engine = GeminiEngine(
            model=os.getenv(
                "GEMINI_MODEL",
                "gemini-3.8-flash"
            )
        )

        model_name = engine.model

    elif provider == "ollama":

        engine = OllamaEngine(
            model="qwen3.5:4b",
            base_url="http://localhost:11434"
        )

        model_name = "qwen3.5:4b"

    else:
        raise RuntimeError(
            f"Unsupported LEO_PROVIDER: {provider}"
        )

    print("LEO intelligence engine initialized.")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")

    yield

    engine = None

    print("LEO intelligence engine stopped.")


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="LEO AI API",
    description="LEO AI assistant with configurable AI providers.",
    version="4.0.0",
    lifespan=lifespan
)


# ==========================================
# CORS CONFIGURATION
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "https://neelagirichandrika.github.io"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ==========================================
# REQUEST STRUCTURE
# ==========================================

class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=500
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


# ==========================================
# RESPONSE STRUCTURE
# ==========================================

class ChatResponse(BaseModel):

    assistant: str
    response: str


# ==========================================
# HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {
        "application": "LEO AI",
        "status": "running",
        "intelligence_engine": provider,
        "model": model_name
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "intelligence_engine": provider,
        "model": model_name
    }


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):

    if engine is None:

        raise HTTPException(
            status_code=503,
            detail="LEO intelligence engine is not initialized."
        )

    try:

        response = await engine.generate_response(
            prompt=request.message,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature
        )

        return ChatResponse(
            assistant="LEO",
            response=response
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except RuntimeError as error:

        raise HTTPException(
            status_code=503,
            detail=str(error)
        )
