import sys
import os
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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

# --- CORS Middleware (保留，良好的開發習慣) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 掛載靜態檔案目錄 ---
# 使用絕對路徑以避免執行位置造成的問題
APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

# --- CI/CD 修復 ---
# 在掛載前，確保 static 目錄存在，以解決 CI 環境中找不到目錄的問題
STATIC_DIR.mkdir(parents=True, exist_ok=True)
# --- 修復結束 ---

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Routers ---
app.include_router(scoring.router)

# --- 提供前端主頁面 ---
@app.get("/")
async def read_index(request: Request):
    # 假設您的 index.html 檔案位於專案根目錄 (與 'app' 資料夾同層)
    # 如果路徑不同，請修改 '..' 的部分
    index_path = APP_DIR / ".." / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    return {"error": "index.html not found"}

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting ESG AutoScorer API...")
    uvicorn.run("main:app", host="12-7.0.0.1", port=8000, reload=True)

