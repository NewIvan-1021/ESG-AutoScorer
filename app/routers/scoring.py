import asyncio
import logging
from typing import List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models import ScoringResult
from app.services import call_gemini_for_scoring_sync, extract_text_from_pdf_sync

router = APIRouter()
logger = logging.getLogger(__name__)

async def process_single_file(file_content: bytes, filename: str, company_name: str, website_url: str) -> dict:
    """
    非同步地處理單一檔案，包含 PDF 提取與 AI 評分。
    """
    loop = asyncio.get_event_loop()
    try:
        logger.info(f"ℹ️  開始處理檔案: {filename}")
        pdf_text = await loop.run_in_executor(None, extract_text_from_pdf_sync, file_content, filename)
        
        if pdf_text.startswith("錯誤："):
            return { "company": company_name, "overview_comment": pdf_text, "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }

        ai_result = await loop.run_in_executor(None, call_gemini_for_scoring_sync, company_name, pdf_text, website_url)
        logger.info(f"✅ 成功處理檔案: {filename}")
        return ai_result
    except Exception as e:
        logger.error(f"🔴 在 process_single_file 中處理檔案 '{filename}' 時發生未預期的錯誤: {e}", exc_info=True)
        return { "company": company_name, "overview_comment": f"處理檔案 '{filename}' 時發生嚴重錯誤，請檢查後端日誌。", "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }

@router.get("/health", tags=["General"])
def health_check():
    """健康檢查端點，用於確認後端服務是否正常運行。"""
    return {"status": "ok", "message": "後端伺服器運行中"}

@router.post("/scoring/batch", response_model=List[ScoringResult], tags=["Scoring"])
async def scoring_batch_endpoint(
    files: List[UploadFile] = File(...),
    company_names: List[str] = Form(...),
    website_urls: List[str] = Form(...),
):
    """
    接收多份 PDF 檔案及對應的公司資料，並行處理後回傳評分結果列表。
    """
    if not (len(files) == len(company_names) == len(website_urls)):
        raise HTTPException(status_code=400, detail="檔案、公司名稱和網站 URL 的數量必須一致。")

    tasks = []
    for i, file in enumerate(files):
        if file.content_type != "application/pdf":
            logger.warning(f"⚠️ 檔案 '{file.filename}' 不是 PDF，將略過處理。")
            continue

        # 讀取檔案內容（bytes）
        file_content = await file.read()

        # 建立非同步任務
        task = process_single_file(file_content, file.filename, company_names[i], website_urls[i])
        tasks.append(task)

    if not tasks:
        raise HTTPException(status_code=400, detail="未提供任何有效的 PDF 檔案。")

    # ✅ 等待所有任務完成
    results = await asyncio.gather(*tasks)

    # ✅ 檢查結果
    if not results:
        raise HTTPException(status_code=500, detail="所有檔案處理失敗，未產生任何結果。請檢查後端日誌。")

    return results

