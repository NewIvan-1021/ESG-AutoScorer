import sys
import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import google.generativeai as genai
from app.routers import scoring

# --- Environment Setup ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- AI SDK Initialization ---
try:
    if not API_KEY:
        raise ValueError("Fatal Error: GOOGLE_API_KEY not found. Please check your .env file.")
    genai.configure(api_key=API_KEY)
    logger.info(f"✅ Google AI SDK configured successfully (Version: {genai.__version__}).")
except Exception as e:
    logger.error(f"🔴 AI SDK configuration failed: {e}", exc_info=True)
    sys.exit(1)

app = FastAPI(
    title="ESG AutoScorer API",
    description="An API for automated ESG report scoring based on TCSA criteria.",
    version="3.0.0",
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routers ---
app.include_router(scoring.router)

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ESG AutoScorer API...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
