from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.leo_engine import LeoEngine


engine = None


# ==========================================
# MODEL LIFECYCLE
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    global engine

    engine = LeoEngine()

    yield

    engine = None


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="LEO AI API",
    description="Backend API for the LEO language model",
    version="1.0.0",
    lifespan=lifespan
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
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
        default=100,
        ge=1,
        le=150
    )

    temperature: float = Field(
        default=0.8,
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
        "status": "running"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": engine is not None
    }


# ==========================================
# CHAT ENDPOINT
# ==========================================

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="LEO model is not loaded."
        )

    try:

        response = engine.generate_response(
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
