import sys
import os
import json
import asyncio
import logging
import re # 引入正規表示式模組
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

# --- 備案模型清單 (依照您的要求維持不變) ---
FALLBACK_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
]

# --- AI SDK 初始化 ---
try:
    if not API_KEY:
        raise ValueError("致命錯誤：找不到 GOOGLE_API_KEY。請檢查您的 .env 檔案。")
    genai.configure(api_key=API_KEY)
    logger.info(f"✅ Google AI SDK 已成功設定 (版本: {genai.__version__})。")
except Exception as e:
    logger.error(f"🔴 AI SDK 設定失敗: {e}", exc_info=True)
    sys.exit(1) 

app = FastAPI(
    title="ESG 報告書自動評分系統 API",
    description="提供基於 TCSA 準則的 AI 評分功能",
    version="2.5.0", # 🎉 增強 AI 提示與證據追溯
)

# --- CORS 中介軟體設定 ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 完整的資料模型 (維持不變) ---
class SubCriterionScore(BaseModel):
    title: str
    max_score: float
    score: Optional[float] = None
    rationale: Optional[str] = None

class CriterionScore(BaseModel):
    title: str
    max_score: float
    score: Optional[float] = None
    sub_criteria: List[SubCriterionScore] = Field(default_factory=list)

class SectionScore(BaseModel):
    title: str
    max_score: float
    score: Optional[float] = None
    ai_comment: Optional[str] = None
    criteria: List[CriterionScore] = Field(default_factory=list)

class BreakdownItem(BaseModel):
    id: str
    raw_score: Optional[float] = 0.0
    raw_max_score: Optional[float] = 0.0
    ai_comment: Optional[str] = None
    sections: List[SectionScore] = Field(default_factory=list)

class TotalsScore(BaseModel):
    report: Optional[float] = None
    media: Optional[float] = None
    final: Optional[float] = None

class ScoringResult(BaseModel):
    company: str
    overview_comment: Optional[str] = None
    strengths: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    improvements: Optional[Dict[str, List[str]]] = Field(default_factory=dict)
    breakdown: List[BreakdownItem] = Field(default_factory=list)
    totals: Optional[TotalsScore] = None

# --- 核心功能函式 ---

def _get_prompt(company_name: str, pdf_text: str, website_url: str) -> str:
    """產生用於 AI 評分的完整提示文字"""
    # 改善：強化 rationale 的要求，使其包含頁碼或證據
    return f"""
    請你扮演一位專業且嚴謹的台灣企業永續獎(TCSA)評審。
    你的任務是根據我提供的企業永續報告書內文和官方網站，依照以下的 TCSA 詳細評選準則，逐項進行評分。

    ## 評分對象
    - **公司名稱:** {company_name}
    - **官方網站:** {website_url}
    - **報告書內文摘要:** {pdf_text[:18000]}... (僅顯示前 18000 字)

    ## 你的任務與輸出格式
    請**嚴格**依照以下 JSON 格式回傳你的評分結果。你的整個輸出**必須**是一個單一、無註解、且嚴格符合 RFC 8259 規範的 JSON 物件。
    - **重要**: 所有 key 和 string value 都必須使用雙引號 `""`。
    - **重要**: 你必須為 `sub_criteria` 陣列中的每一個項目評分，分數級距為 0.5。
    - **重要**: 對於每一個 `sub_criteria` 項目，你都必須提供一個 `rationale` 欄位。這個欄位必須簡潔地說明你給予該分數的**主要理由**，並且**盡可能**包含具體的**文本證據**，例如頁碼、圖表編號或章節名稱 (例如: "報告書第23頁的圖表顯示..." 或 "官網『永續專區』明確揭露...")。
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
          "sections": [
            {{
              "title": "完整性", "max_score": 40.0,
              "criteria": [
                {{ "title": "重大性議題", "max_score": 8.0, "sub_criteria": [ 
                    {{"title": "是否清楚列出或呈現重大性議題分析之矩陣圖或其他圖表，且清楚標明各項議題的種類", "max_score": 2.0, "score": 0.0, "rationale": "報告書p.15呈現了清晰的重大性議題矩陣圖。"}}, 
                    {{"title": "是否清楚說明組織重大性議題分析之過程與方法", "max_score": 2.0, "score": 0.0, "rationale": "未找到具體的分析過程與方法說明。"}}, 
                    {{"title": "是否有呈現出重大性議題在報告書中的連結性", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, 
                    {{"title": "是否清楚說明重大性議題對於組織的意義", "max_score": 2.0, "score": 0.0, "rationale": "..."}} 
                ] }},
                {{ "title": "利害關係人共融", "max_score": 6.0, "sub_criteria": [ {{"title": "是否清楚列出組織的利害關係人之種類與意義", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否清楚說明各種利害關係人議合之方法", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否清楚說明各種利害關係人關注之議題", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否清楚說明組織針對各項議題的因應之道", "max_score": 2.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "策略", "max_score": 12.0, "sub_criteria": [ {{"title": "報告書中是否有說明永續對組織的重要性與意義(價值鏈的呈現)", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "報告書中是否有揭露組織營運相關之內外部環境分析", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "報告書中是否有揭露組織對於環境、社會、治理等面向的發展原則與管理機制(長期策略)", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否有在各個面向或是各類重大性議題說明組織未來改善目標(中期策略)", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "針對各項重大性議題是否有設定隔年度之量化或是質化目標(短期策略)", "max_score": 2.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "組織介紹", "max_score": 2.0, "sub_criteria": [ {{"title": "揭露資訊：主要產品與服務、財務績效、地理分布、員工資訊、整體環境與組織營運之關聯性等", "max_score": 2.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "重大永續規範執行及資訊揭露", "max_score": 12.0, "sub_criteria": [ {{"title": "氣候相關財務揭露(TCFD)", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "永續會計準則委員會準則(SASB)", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "自然相關財務揭露(TNFD)", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "國際財務報導準則(IFRS) S1,S2揭露", "max_score": 3.0, "score": 0.0, "rationale": "..."}} ] }}
              ]
            }},
            {{
              "title": "可信度", "max_score": 35.0,
              "criteria": [
                {{ "title": "管理流程", "max_score": 10.0, "sub_criteria": [ {{"title": "報告揭露採用之指引與準則", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否揭露報告書主要負責單位", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "報告書的管理方式", "max_score": 4.0, "score": 0.0, "rationale": "..."}}, {{"title": "針對各項重大性議題皆說明管理方針", "max_score": 4.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "利害關係人回應", "max_score": 5.0, "sub_criteria": [ {{"title": "針對利害關係人關注之議題，組織是否實際回應議題，並提出相對應之作為、策略與規劃等政策", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "組織是否有針對組織鑑別出之實質性議題進行回應，並提出相對應之策略與作為", "max_score": 3.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "治理", "max_score": 5.0, "sub_criteria": [ {{"title": "是否有說明組織組織針對永續報告的責任單位", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "報告書是否有說明董事會的薪酬與永續績效的連結性", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "報告書是否有揭露組織組織的風險與可能之機會(因應之道)", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "組織績效指標管理方針是否與組織永續原則一致", "max_score": 1.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "績效", "max_score": 5.0, "sub_criteria": [ {{"title": "績效之揭露是否完整(重大性議題涵蓋經濟、環境與社會，是否有質化的說明與數據)", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "重大性議題是否有量化的圖表說明", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否有揭露過去負面訊息", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "績效的呈現是否易懂", "max_score": 1.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "保證/確信", "max_score": 10.0, "sub_criteria": [ {{"title": "是否已建立永續資訊編制內部控制制度及相關流程", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "永續資訊編制內部控制制度及其內部稽核執行情形說明", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否有外部第三方獨立保證/確信之佐證資料", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "外部保證是否有說明保證等級、範疇與方法(中度/有限等級者最多得2分，高度/合理等級者最多可得3分)", "max_score": 3.0, "score": 0.0, "rationale": "..."}} ] }}
              ]
            }},
            {{
              "title": "溝通性", "max_score": 25.0,
              "criteria": [
                {{ "title": "展現", "max_score": 10.0, "sub_criteria": [ {{"title": "版面是否圖表與文字說明比例恰當，內容清晰且易於閱讀", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "具有英文版報告書", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "展現創新的資訊呈現方式", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "報告書之份量是否適當(頁數120-150頁為參考範圍)", "max_score": 2.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "利害關係人共融", "max_score": 5.0, "sub_criteria": [ {{"title": "組織永續報告書是否公開下載", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "是否有說明利害關係人議合(溝通資訊)的方法", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "利害關係人議合的結果，組織是否公開揭露其相對應的回應與作為", "max_score": 2.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "架構", "max_score": 10.0, "sub_criteria": [ {{"title": "是否清楚整理並呈現本年度的亮點作為報告書的總結", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "完整的索引設計(包括GRI, SASB及其他重要規範等)", "max_score": 3.0, "score": 0.0, "rationale": "..."}}, {{"title": "報告書附有清楚的連結，使讀者可透過網頁的說明獲得更細節的資訊", "max_score": 2.0, "score": 0.0, "rationale": "..."}}, {{"title": "架構呈現完整易於查閱", "max_score": 2.0, "score": 0.0, "rationale": "..."}} ] }}
              ]
            }}
          ]
        }},
        {{
          "id": "media",
          "sections": [
            {{
              "title": "多元媒體應用及內容品質", "max_score": 19.0,
              "criteria": [
                {{ "title": "組織永續專區", "max_score": 3.0, "sub_criteria": [ {{"title": "設置組織永續專區", "max_score": 0.5, "score": 0.0, "rationale": "..."}}, {{"title": "是否將組織永續專區連結設於首頁", "max_score": 0.5, "score": 0.0, "rationale": "..."}}, {{"title": "是否提供報告書下載", "max_score": 0.5, "score": 0.0, "rationale": "..."}}, {{"title": "是否有網站地圖", "max_score": 0.5, "score": 0.0, "rationale": "..."}}, {{"title": "站內搜尋引擎", "max_score": 0.5, "score": 0.0, "rationale": "..."}}, {{"title": "是否將組織永續專區分類且內容充實", "max_score": 0.5, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "網頁管理與即時更新", "max_score": 4.0, "sub_criteria": [ {{"title": "判斷依據：由最新消息觀察網頁是否為最新訊息、是否即時更新", "max_score": 4.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "電子版報告書與關鍵資訊連結", "max_score": 4.0, "sub_criteria": [ {{"title": "按照永續報告定義，須符合環境、社會與治理(ESG)以及供應鏈管理等四項議題之揭露", "max_score": 4.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "多元媒體展現", "max_score": 4.0, "sub_criteria": [ {{"title": "文字說明", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "圖表說明", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "使用影片", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "互動式網頁", "max_score": 1.0, "score": 0.0, "rationale": "..."}} ] }},
                {{ "title": "溝通回饋管道與社群網絡互動", "max_score": 4.0, "sub_criteria": [ {{"title": "線上回饋機制之應用(網路填寫或連結至電子信箱)", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "線上互動式機制之應用", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "社交網站之應用", "max_score": 1.0, "score": 0.0, "rationale": "..."}}, {{"title": "提供訂閱電子報", "max_score": 1.0, "score": 0.0, "rationale": "..."}} ] }}
              ]
            }}
          ]
        }}
      ]
    }}
    ```
    """

def _calculate_final_scores(ai_data: Dict[str, Any]) -> Dict[str, Any]:
    """根據 AI 回傳的原始分數，計算加權後的最終分數"""
    try:
        report_breakdown = next((item for item in ai_data.get("breakdown", []) if item.get("id") == "report"), {})
        media_breakdown = next((item for item in ai_data.get("breakdown", []) if item.get("id") == "media"), {})
        
        report_raw_score = sum(s.get("score", 0) or 0 for s in report_breakdown.get("sections", []))
        report_raw_max = sum(s.get("max_score", 0) or 0 for s in report_breakdown.get("sections", []))
        
        media_raw_score = sum(s.get("score", 0) or 0 for s in media_breakdown.get("sections", []))
        media_raw_max = sum(s.get("max_score", 0) or 0 for s in media_breakdown.get("sections", []))
        
        report_scaled = (report_raw_score / report_raw_max) * 60 if report_raw_max > 0 else 0
        media_scaled = (media_raw_score / media_raw_max) * 40 if media_raw_max > 0 else 0
        
        ai_data["totals"] = {
            "report": round(report_scaled, 2),
            "media": round(media_scaled, 2),
            "final": round(report_scaled + media_scaled, 2)
        }
    except Exception as e:
        logger.error(f"🔴 計算最終分數時出錯: {e}", exc_info=True)
        ai_data["totals"] = None
    return ai_data

# --- 錯誤修正：增強 JSON 解析的穩定性 ---
def _parse_ai_response(response_text: str) -> Dict[str, Any]:
    """從 AI 的回應中解析出 JSON 物件，並增加對常見錯誤的容錯能力"""
    # 優先使用正規表示式尋找被 ```json ... ``` 包圍的區塊
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    if match:
        cleaned_text = match.group(1)
    else:
        # 如果沒有找到 markdown 區塊，則嘗試尋找第一個 '{' 到最後一個 '}' 的內容
        start = response_text.find('{')
        end = response_text.rfind('}')
        if start != -1 and end != -1:
            cleaned_text = response_text[start:end+1]
        else:
            # 如果連大括號都找不到，就直接使用原始文字
            cleaned_text = response_text
    
    # 讓呼叫者去處理解析錯誤，以便我們可以記錄清理後的文字
    return json.loads(cleaned_text)

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
    """
    同步地呼叫 Gemini AI 進行評分。
    此函式會依序嘗試 FALLBACK_MODELS 列表中的模型，直到成功為止。
    """
    prompt = _get_prompt(company_name, pdf_text, website_url)
    generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
    
    last_error = "未知的 AI 錯誤"
    raw_response_text_for_logging = ""

    for model_name in FALLBACK_MODELS:
        try:
            logger.info(f"ℹ️  正在嘗試使用模型: {model_name}...")
            # 關於 ALTS 的警告是正常的，在本地端開發時可忽略
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                contents=prompt,
                generation_config=generation_config
            )

            raw_response_text_for_logging = response.text
            ai_data = _parse_ai_response(raw_response_text_for_logging)
            ai_data_with_scores = _calculate_final_scores(ai_data)
            
            logger.info(f"✅ 模型 '{model_name}' 成功回傳並解析結果。")
            return ai_data_with_scores

        except json.JSONDecodeError as e:
            last_error = f"JSON 解析失敗 - {e}"
            logger.error(f"🔴 模型 '{model_name}' 回應的 JSON 格式錯誤: {e}")
            # --- 錯誤修正：優化日誌，印出清理後仍無法解析的文字 ---
            cleaned_text_for_log = raw_response_text_for_logging.strip().replace("```json", "").replace("```", "")
            logger.error(f"👇 嘗試解析的文字內容 (可能有問題) 👇\n{cleaned_text_for_log}")
        except Exception as e:
            last_error = str(e)
            logger.warning(f"⚠️ 模型 '{model_name}' 呼叫失敗: {e}。正在嘗試下一個備案模型...")

    final_error_message = f"所有備案 AI 模型皆嘗試失敗。最終錯誤: {last_error}"
    logger.error(f"🔴 {final_error_message}")
    return { "company": company_name, "overview_comment": final_error_message, "totals": None, "strengths": {}, "improvements": {}, "breakdown": [] }

# --- API 端點 ---

@app.get("/health", tags=["General"])
def health_check():
    """健康檢查端點，用於確認後端服務是否正常運行。"""
    return {"status": "ok", "message": "後端伺-服器運行中"}

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

@app.post("/scoring/batch", response_model=List[ScoringResult], tags=["Scoring"])
async def scoring_batch_endpoint(
    files: List[UploadFile] = File(...),
    company_names: List[str] = Form(...),
    website_urls: List[str] = Form(...)
):
    """
    接收多份 PDF 檔案及對應的公司資料，並行處理後回傳評分結果列表。
    """
    if not (len(files) == len(company_names) == len(website_urls)):
        raise HTTPException(status_code=400, detail="檔案、公司名稱和網站 URL 的數量必須一致。")

    tasks = []
    for i, file in enumerate(files):
        if file.content_type != 'application/pdf':
             logger.warning(f"⚠️  檔案 '{file.filename}' 不是 PDF，將略過處理。")
             continue
        task = process_single_file(await file.read(), file.filename, company_names[i], website_urls[i])
        tasks.append(task)
    
    if not tasks:
        raise HTTPException(status_code=400, detail="未提供任何有效的 PDF 檔案。")

    results = await asyncio.gather(*tasks)
    
    if not results:
        raise HTTPException(status_code=500, detail="所有檔案處理失敗，未產生任何結果。請檢查後端日誌。")
    
    return results

# --- 為了方便本地開發，可以直接執行此檔案 ---
if __name__ == "__main__":
    import uvicorn
    # 增加單元測試的說明
    print("\n--- 系統資訊 ---")
    print("若要執行單元測試，請安裝 pytest (`pip install pytest`)")
    print("然後在終端機中執行 `pytest` 指令。")
    print("----------------\n")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

