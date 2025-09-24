import os
import json
import asyncio
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
import google.generativeai as genai
from pypdf import PdfReader
import io

# --- 環境設定 ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- 日誌設定 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ESG 報告書自動評分系統 API",
    description="提供基於 TCSA 準則的 AI 評分功能",
    version="1.0.1",
)


# --- CORS 中介軟體設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 為了開發方便，暫時允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AI 模型設定 ---
if not API_KEY:
    logger.error("🔴 致命錯誤：找不到 GOOGLE_API_KEY。請檢查您的 .env 檔案是否已建立且內容正確。")
else:
    try:
        genai.configure(api_key=API_KEY)
        logger.info("✅ Google AI API Key 已成功設定。")
    except Exception as e:
        logger.error(f"🔴 致命錯誤：設定 Google AI API Key 時發生錯誤: {e}")

# --- Pydantic 完整的資料模型 (用於驗證與 API 文件) ---
# 這些模型確保了從 AI 接收和發送到前端的資料結構都是正確的。

class SubCriterionScore(BaseModel):
    title: str
    max_score: float
    score: float

class CriterionScore(BaseModel):
    title: str
    max_score: float
    score: float
    sub_criteria: List[SubCriterionScore] = Field(default_factory=list)

class SectionScore(BaseModel):
    title: str
    max_score: float
    score: float
    ai_comment: Optional[str] = None
    criteria: List[CriterionScore] = Field(default_factory=list)

class BreakdownItem(BaseModel):
    id: str
    raw_score: Optional[float] = 0.0
    raw_max_score: Optional[float] = 0.0
    ai_comment: Optional[str] = None
    sections: List[SectionScore] = Field(default_factory=list)

class TotalsScore(BaseModel):
    report: float
    media: float
    final: float

class ScoringResult(BaseModel):
    company: str
    overview_comment: Optional[str] = None
    strengths: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    improvements: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    breakdown: List[BreakdownItem] = Field(default_factory=list)
    totals: Optional[TotalsScore] = None


# --- 核心功能函式 ---

def extract_text_from_pdf_sync(file_content: bytes, filename: str) -> str:
    """同步地從 PDF 檔案的二進位內容中提取文字"""
    try:
        pdf_file = io.BytesIO(file_content)
        reader = PdfReader(pdf_file)
        text = " ".join([page.extract_text() or "" for page in reader.pages])
        return " ".join(text.split())
    except Exception as e:
        logger.error(f"🔴 處理 PDF 檔案 '{filename}' 時失敗: {e}")
        return f"錯誤：無法讀取 PDF 檔案 '{filename}'。檔案可能已損壞或格式不支援。"

def call_gemini_for_scoring_sync(company_name: str, pdf_text: str, website_url: str) -> dict:
    """同步地呼叫 Gemini AI 進行評分，並包含更穩健的錯誤處理"""
    if not API_KEY:
        return { "company": company_name, "overview_comment": "後端 AI 服務未初始化：缺少 API Key。", "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }

    generation_config = {"response_mime_type": "application/json"}
    model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", generation_config=generation_config)
    
    # 在 Prompt 中再次強調 JSON 格式的嚴格性
    prompt = f"""
    請你扮演一位專業且嚴謹的台灣企業永續獎(TCSA)評審。
    你的任務是根據我提供的企業永續報告書內文和官方網站，依照以下的 TCSA 詳細評選準則，逐項進行評分。

    ## 評分對象
    - **公司名稱:** {company_name}
    - **官方網站:** {website_url}
    - **報告書內文摘要:** {pdf_text[:15000]}... (僅顯示前 15000 字)

    ## 你的任務與輸出格式
    請**嚴格**依照以下 JSON 格式回傳你的評分結果。你的整個輸出**必須**是一個單一、無註解、且嚴格符合 RFC 8259 規範的 JSON 物件。
    - **重要**: 所有 key 和 string value 都必須使用雙引號 `""`。
    - **重要**: 你必須為 `sub_criteria` 陣列中的每一個項目評分，分數級距為 0.5。
    - **重要**: `criteria` 的 `score` 必須是其底下所有 `sub_criteria` 分數的總和。
    - **重要**: `sections` 的 `score` 必須是其底下所有 `criteria` 分數的總和。
    - **重要**: `strengths` 和 `improvements` 必須是物件(object)，其 `key` 為評分構面（完整性、可信度、溝通性、多元媒體應用），`value` 為該構面下的優點或建議列表(string array)。

    ```json
    {{
      "company": "{company_name}",
      "overview_comment": "一句話總結這份報告書與網站的整體表現。",
      "strengths": {{ "完整性": ["..."], "可信度": ["..."] }},
      "improvements": {{ "溝通性": ["..."], "多元媒體應用": ["..."] }},
      "breakdown": [
        {{
          "id": "report",
          "raw_score": 0.0,
          "raw_max_score": 100.0,
          "ai_comment": "針對「永續報告書」整體的評論。",
          "sections": [
            {{
              "title": "完整性", "max_score": 40.0, "score": 0.0, "ai_comment": "針對「完整性」的評論。",
              "criteria": [
                {{ "title": "重大性議題", "max_score": 8.0, "score": 0.0, "sub_criteria": [ {{"title": "是否清楚列出或呈現重大性議題分析之矩陣圖或其他圖表，且清楚標明各項議題的種類", "max_score": 2.0, "score": 0.0}}, {{"title": "是否清楚說明組織重大性議題分析之過程與方法", "max_score": 2.0, "score": 0.0}}, {{"title": "是否有呈現出重大性議題在報告書中的連結性", "max_score": 2.0, "score": 0.0}}, {{"title": "是否清楚說明重大性議題對於組織的意義", "max_score": 2.0, "score": 0.0}} ] }},
                {{ "title": "利害關係人共融", "max_score": 6.0, "score": 0.0, "sub_criteria": [ {{"title": "是否清楚列出組織的利害關係人之種類與意義", "max_score": 1.0, "score": 0.0}}, {{"title": "是否清楚說明各種利害關係人議合之方法", "max_score": 2.0, "score": 0.0}}, {{"title": "是否清楚說明各種利害關係人關注之議題", "max_score": 1.0, "score": 0.0}}, {{"title": "是否清楚說明組織針對各項議題的因應之道", "max_score": 2.0, "score": 0.0}} ] }},
                {{ "title": "策略", "max_score": 12.0, "score": 0.0, "sub_criteria": [ {{"title": "報告書中是否有說明永續對組織的重要性與意義(價值鏈的呈現)", "max_score": 2.0, "score": 0.0}}, {{"title": "報告書中是否有揭露組織營運相關之內外部環境分析", "max_score": 3.0, "score": 0.0}}, {{"title": "報告書中是否有揭露組織對於環境、社會、治理等面向的發展原則與管理機制(長期策略)", "max_score": 3.0, "score": 0.0}}, {{"title": "是否有在各個面向或是各類重大性議題說明組織未來改善目標(中期策略)", "max_score": 2.0, "score": 0.0}}, {{"title": "針對各項重大性議題是否有設定隔年度之量化或是質化目標(短期策略)", "max_score": 2.0, "score": 0.0}} ] }},
                {{ "title": "組織介紹", "max_score": 2.0, "score": 0.0, "sub_criteria": [ {{"title": "揭露資訊：主要產品與服務、財務績效、地理分布、員工資訊、整體環境與組織營運之關聯性等", "max_score": 2.0, "score": 0.0}} ] }},
                {{ "title": "重大永續規範執行及資訊揭露", "max_score": 12.0, "score": 0.0, "sub_criteria": [ {{"title": "氣候相關財務揭露(TCFD)", "max_score": 3.0, "score": 0.0}}, {{"title": "永續會計準則委員會準則(SASB)", "max_score": 3.0, "score": 0.0}}, {{"title": "自然相關財務揭露(TNFD)", "max_score": 3.0, "score": 0.0}}, {{"title": "國際財務報導準則(IFRS) S1,S2揭露", "max_score": 3.0, "score": 0.0}} ] }}
              ]
            }},
            {{
              "title": "可信度", "max_score": 35.0, "score": 0.0, "ai_comment": "針對「可信度」的評論。",
              "criteria": [
                {{ "title": "管理流程", "max_score": 10.0, "score": 0.0, "sub_criteria": [ {{"title": "報告揭露採用之指引與準則", "max_score": 1.0, "score": 0.0}}, {{"title": "是否揭露報告書主要負責單位", "max_score": 1.0, "score": 0.0}}, {{"title": "報告書的管理方式", "max_score": 4.0, "score": 0.0}}, {{"title": "針對各項重大性議題皆說明管理方針", "max_score": 4.0, "score": 0.0}} ] }},
                {{ "title": "利害關係人回應", "max_score": 5.0, "score": 0.0, "sub_criteria": [ {{"title": "針對利害關係人關注之議題，組織是否實際回應議題，並提出相對應之作為、策略與規劃等政策", "max_score": 2.0, "score": 0.0}}, {{"title": "組織是否有針對組織鑑別出之實質性議題進行回應，並提出相對應之策略與作為", "max_score": 3.0, "score": 0.0}} ] }},
                {{ "title": "治理", "max_score": 5.0, "score": 0.0, "sub_criteria": [ {{"title": "是否有說明組織組織針對永續報告的責任單位", "max_score": 1.0, "score": 0.0}}, {{"title": "報告書是否有說明董事會的薪酬與永續績效的連結性", "max_score": 2.0, "score": 0.0}}, {{"title": "報告書是否有揭露組織組織的風險與可能之機會(因應之道)", "max_score": 1.0, "score": 0.0}}, {{"title": "組織績效指標管理方針是否與組織永續原則一致", "max_score": 1.0, "score": 0.0}} ] }},
                {{ "title": "績效", "max_score": 5.0, "score": 0.0, "sub_criteria": [ {{"title": "績效之揭露是否完整(重大性議題涵蓋經濟、環境與社會，是否有質化的說明與數據)", "max_score": 2.0, "score": 0.0}}, {{"title": "重大性議題是否有量化的圖表說明", "max_score": 1.0, "score": 0.0}}, {{"title": "是否有揭露過去負面訊息", "max_score": 1.0, "score": 0.0}}, {{"title": "績效的呈現是否易懂", "max_score": 1.0, "score": 0.0}} ] }},
                {{ "title": "保證/確信", "max_score": 10.0, "score": 0.0, "sub_criteria": [ {{"title": "是否已建立永續資訊編制內部控制制度及相關流程", "max_score": 2.0, "score": 0.0}}, {{"title": "永續資訊編制內部控制制度及其內部稽核執行情形說明", "max_score": 3.0, "score": 0.0}}, {{"title": "是否有外部第三方獨立保證/確信之佐證資料", "max_score": 2.0, "score": 0.0}}, {{"title": "外部保證是否有說明保證等級、範疇與方法(中度/有限等級者最多得2分，高度/合理等級者做多可得3分)", "max_score": 3.0, "score": 0.0}} ] }}
              ]
            }},
            {{
              "title": "溝通性", "max_score": 25.0, "score": 0.0, "ai_comment": "針對「溝通性」的評論。",
              "criteria": [
                {{ "title": "展現", "max_score": 10.0, "score": 0.0, "sub_criteria": [ {{"title": "版面是否圖表與文字說明比例恰當，內容清晰且易於閱讀", "max_score": 3.0, "score": 0.0}}, {{"title": "具有英文版報告書", "max_score": 3.0, "score": 0.0}}, {{"title": "展現創新的資訊呈現方式", "max_score": 2.0, "score": 0.0}}, {{"title": "報告書之份量是否適當(頁數120-150頁為參考範圍)", "max_score": 2.0, "score": 0.0}} ] }},
                {{ "title": "利害關係人共融", "max_score": 5.0, "score": 0.0, "sub_criteria": [ {{"title": "組織永續報告書是否公開下載", "max_score": 1.0, "score": 0.0}}, {{"title": "是否有說明利害關係人議合(溝通資訊)的方法", "max_score": 2.0, "score": 0.0}}, {{"title": "利害關係人議合的結果，組織是否公開揭露其相對應的回應與作為", "max_score": 2.0, "score": 0.0}} ] }},
                {{ "title": "架構", "max_score": 10.0, "score": 0.0, "sub_criteria": [ {{"title": "是否清楚整理並呈現本年度的亮點作為報告書的總結", "max_score": 3.0, "score": 0.0}}, {{"title": "完整的索引設計(包括GRI, SASB及其他重要規範等)", "max_score": 3.0, "score": 0.0}}, {{"title": "報告書附有清楚的連結，使讀者可透過網頁的說明獲得更細節的資訊", "max_score": 2.0, "score": 0.0}}, {{"title": "架構呈現完整易於查閱", "max_score": 2.0, "score": 0.0}} ] }}
              ]
            }}
          ]
        }},
        {{
          "id": "media",
          "raw_score": 0.0,
          "raw_max_score": 19.0,
          "ai_comment": "針對「多元媒體應用」整體的評論。",
          "sections": [
            {{
              "title": "多元媒體應用及內容品質", "max_score": 19.0, "score": 0.0, "ai_comment": "針對「多元媒體應用」細項的評論。",
              "criteria": [
                {{ "title": "組織永續專區", "max_score": 3.0, "score": 0.0, "sub_criteria": [ {{"title": "設置組織永續專區", "max_score": 0.5, "score": 0.0}}, {{"title": "是否將組織永續專區連結設於首頁", "max_score": 0.5, "score": 0.0}}, {{"title": "是否提供報告書下載", "max_score": 0.5, "score": 0.0}}, {{"title": "是否有網站地圖", "max_score": 0.5, "score": 0.0}}, {{"title": "站內搜尋引擎", "max_score": 0.5, "score": 0.0}}, {{"title": "是否將組織永續專區分類且內容充實", "max_score": 0.5, "score": 0.0}} ] }},
                {{ "title": "網頁管理與即時更新", "max_score": 4.0, "score": 0.0, "sub_criteria": [ {{"title": "判斷依據：由最新消息觀察網頁是否為最新訊息、是否即時更新", "max_score": 4.0, "score": 0.0}} ] }},
                {{ "title": "電子版報告書與關鍵資訊連結", "max_score": 4.0, "score": 0.0, "sub_criteria": [ {{"title": "按照永續報告定義，須符合環境、社會與治理(ESG)以及供應鏈管理等四項議題之揭露", "max_score": 4.0, "score": 0.0}} ] }},
                {{ "title": "多元媒體展現", "max_score": 4.0, "score": 0.0, "sub_criteria": [ {{"title": "文字說明", "max_score": 1.0, "score": 0.0}}, {{"title": "圖表說明", "max_score": 1.0, "score": 0.0}}, {{"title": "使用影片", "max_score": 1.0, "score": 0.0}}, {{"title": "互動式網頁", "max_score": 1.0, "score": 0.0}} ] }},
                {{ "title": "溝通回饋管道與社群網絡互動", "max_score": 4.0, "score": 0.0, "sub_criteria": [ {{"title": "線上回饋機制之應用(網路填寫或連結至電子信箱)", "max_score": 1.0, "score": 0.0}}, {{"title": "線上互動式機制之應用", "max_score": 1.0, "score": 0.0}}, {{"title": "社交網站之應用", "max_score": 1.0, "score": 0.0}}, {{"title": "提供訂閱電子報", "max_score": 1.0, "score": 0.0}} ] }}
              ]
            }}
          ]
        }}
      ]
    }}
    """
    try:
        response = model.generate_content(prompt)
        # 移除 AI 可能回傳的 markdown 標籤
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        
        # 嘗試解析 JSON
        ai_data = json.loads(cleaned_response)

        # --- 分數加權計算 (使用 .get() 來安全地存取資料) ---
        report_breakdown = next((item for item in ai_data.get("breakdown", []) if item.get("id") == "report"), {})
        media_breakdown = next((item for item in ai_data.get("breakdown", []) if item.get("id") == "media"), {})

        report_raw_score = sum(s.get("score", 0) for s in report_breakdown.get("sections", []))
        report_raw_max = sum(s.get("max_score", 0) for s in report_breakdown.get("sections", []))
        
        media_raw_score = sum(c.get("score", 0) for s in media_breakdown.get("sections", []) for c in s.get("criteria", []))
        media_raw_max = sum(c.get("max_score", 0) for s in media_breakdown.get("sections", []) for c in s.get("criteria", []))
        
        report_scaled = (report_raw_score / report_raw_max) * 60 if report_raw_max > 0 else 0
        media_scaled = (media_raw_score / media_raw_max) * 40 if media_raw_max > 0 else 0
        
        ai_data["totals"] = {
            "report": round(report_scaled, 2),
            "media": round(media_scaled, 2),
            "final": round(report_scaled + media_scaled, 2)
        }
        return ai_data

    except json.JSONDecodeError as e:
        logger.error(f"🔴 JSON 解析失敗: {e}")
        logger.error("👇 AI 回傳的原始文字 (可能有問題) 👇")
        logger.error(cleaned_response)
        logger.error("👆 AI 回傳的原始文字 (可能有問題) 👆")
        # ✅ 修正點：將 totals 的值從 {} 改為 None
        return { "company": company_name, "overview_comment": f"AI 分析時發生嚴重錯誤: JSON 解析失敗 - {e}。請檢查後端終端機日誌以獲取 AI 的原始回應。", "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }
    except Exception as e:
        logger.error(f"🔴 呼叫 Gemini AI 或處理其回應時發生未預期的錯誤: {e}")
        # ✅ 修正點：將 totals 的值從 {} 改為 None
        return { "company": company_name, "overview_comment": f"AI 分析時發生嚴重錯誤: {e}", "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }

# --- API 端點 ---

@app.get("/health", tags=["General"])
def health_check():
    """健康檢查端點"""
    return {"status": "ok", "message": "後端伺服器靜靜的等待著"}

async def process_single_file(file_content: bytes, filename: str, company_name: str, website_url: str) -> dict:
    """非同步地處理單一檔案的 PDF 提取與 AI 評分"""
    loop = asyncio.get_event_loop()
    try:
        logger.info(f"ℹ️  開始處理檔案: {filename}")
        # 在執行緒池中運行同步的 PDF 提取函數
        pdf_text = await loop.run_in_executor(None, extract_text_from_pdf_sync, file_content, filename)
        
        if pdf_text.startswith("錯誤："):
            return { "company": company_name, "overview_comment": pdf_text, "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }

        # 在執行緒池中運行同步的 AI 呼叫函數
        ai_result = await loop.run_in_executor(None, call_gemini_for_scoring_sync, company_name, pdf_text, website_url)
        logger.info(f"✅ 成功處理檔案: {filename}")
        return ai_result
    except Exception as e:
        logger.error(f"🔴 在 process_single_file 中處理檔案 '{filename}' 時發生未預期的錯誤: {e}", exc_info=True)
        return { "company": company_name, "overview_comment": f"處理檔案 '{filename}' 時發生嚴重錯誤，請檢查後端日誌。", "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }

@app.post("/scoring/batch", response_model=List[ScoringResult], tags=["Scoring"])
async def scoring_batch_endpoint(
    files: List[UploadFile] = File(...),
    company_names: List[str] = Form(...),
    website_urls: List[str] = Form(...)
):
    """接收多份 PDF 檔案及對應的公司資料，並行處理後回傳評分結果列表。"""
    if not (len(files) == len(company_names) == len(website_urls)):
        raise HTTPException(status_code=400, detail="檔案、公司名稱和網站 URL 的數量必須一致。")

    # 建立一個非同步任務列表
    tasks = []
    for i, file in enumerate(files):
        if file.content_type != 'application/pdf':
             logger.warning(f"⚠️  檔案 '{file.filename}' 不是 PDF，將略過處理。")
             continue
        task = process_single_file(await file.read(), file.filename, company_names[i], website_urls[i])
        tasks.append(task)
    
    if not tasks:
        raise HTTPException(status_code=400, detail="未提供任何有效的 PDF 檔案。")

    # 等待所有任務完成
    results = await asyncio.gather(*tasks)
    
    if not results:
        raise HTTPException(status_code=500, detail="所有檔案處理失敗，未產生任何結果。請檢查後端日誌。")
    
    # FastAPI 會使用 Pydantic 的 response_model (ScoringResult) 來自動驗證 results 的格式
    return results

# --- 為了方便本地開發，可以直接執行此檔案 ---
if __name__ == "__main__":
    import uvicorn
    # 執行 uvicorn 伺服器，並啟用自動重載
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

