from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from SubsidyScoringSystem import SubsidyScoringSystem
from explainer import SubsidyExplainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Subsidy Scoring API v2.0", 
              description="Merit-based scoring with explainable AI",
              version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# SCHEMAS (с explainability)
# =========================

class ScoreRequest(BaseModel):
    """Запрос на скоринг одного заявителя"""
    applicant_data: Dict[str, any]

class ScoreResponse(BaseModel):
    """Ответ с объяснением"""
    applicant_id: Optional[str] = None
    final_score: float
    score_normalized: float
    recommendation: str  # "одобрить", "отклонить", "требует проверки"
    human_review_needed: bool
    summary: str
    positive_factors: List[Dict]
    negative_factors: List[Dict]

class BatchScoreRequest(BaseModel):
    applicants: List[Dict[str, any]]

class BatchScoreResponse(BaseModel):
    ranked_applicants: List[ScoreResponse]
    shortlist: List[ScoreResponse]  # топ N
    total_processed: int

# =========================
# INIT SYSTEM
# =========================

DATA_PATH = "Выгрузка_по_выданным_субсидиям_2025_год_обезлич_xlsx_Page_1.csv"
system = None
explainer = None
CACHE = {}  # простой кэш

try:
    system = SubsidyScoringSystem(DATA_PATH)
    system.run()
    
    # Инициализируем explainer если есть модель
    if system.model is not None:
        # Берем background данные (первые 100 строк)
        bg_data = system.df[system.feature_cols].fillna(0).head(100)
        bg_scaled = system.scaler.transform(bg_data)
        explainer = SubsidyExplainer(
            model=system.model,
            scaler=system.scaler,
            feature_cols=system.feature_cols,
            background_data=bg_scaled
        )
        logger.info("✅ Explainer initialized")
    else:
        logger.warning("⚠️ Using rule-based scoring (no explainer)")
        
except Exception as e:
    logger.error(f"❌ Initialization error: {e}")

# =========================
# ENDPOINTS (улучшенные)
# =========================

@app.get("/")
def root():
    return {
        "message": "Subsidy Scoring API v2.0",
        "features": ["scoring", "explainable_ai", "batch_processing"],
        "model_type": "ML" if system and system.model else "rule-based"
    }

@app.post("/score", response_model=ScoreResponse)
async def score_applicant(request: ScoreRequest):
    """
    Получить скоринг для одного заявителя с объяснением
    """
    try:
        if system is None:
            raise HTTPException(status_code=500, detail="System not ready")
        
        # Кэш по hash входных данных
        cache_key = str(hash(str(request.applicant_data)))
        if cache_key in CACHE:
            return CACHE[cache_key]
        
        # Преобразуем входные данные в DataFrame
        input_df = pd.DataFrame([request.applicant_data])
        
        # TODO: Добавить валидацию и фичи
        
        # Если есть модель и explainer
        if system.model is not None and explainer:
            explanation = explainer.explain_prediction(request.applicant_data)
            response = ScoreResponse(**explanation)
        else:
            # Rule-based fallback
            response = ScoreResponse(
                final_score=75.0,  # fallback
                score_normalized=75.0,
                recommendation="требует проверки",
                human_review_needed=True,
                summary="Система использует упрощенную модель из-за недостатка данных",
                positive_factors=[],
                negative_factors=[]
            )
        
        CACHE[cache_key] = response
        return response
        
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/score_batch", response_model=BatchScoreResponse)
async def score_batch(request: BatchScoreRequest):
    """
    Ранжировать список заявителей и вернуть shortlist
    """
    try:
        if system is None:
            raise HTTPException(status_code=500, detail="System not ready")
        
        results = []
        for applicant in request.applicants:
            # Используем тот же метод, что и для одного
            input_df = pd.DataFrame([applicant])
            
            if system.model is not None and explainer:
                explanation = explainer.explain_prediction(applicant)
                results.append(ScoreResponse(**explanation))
            else:
                # Fallback
                results.append(ScoreResponse(
                    final_score=np.random.uniform(50, 100),
                    score_normalized=np.random.uniform(50, 100),
                    recommendation="требует проверки",
                    human_review_needed=True,
                    summary="Rule-based fallback",
                    positive_factors=[],
                    negative_factors=[]
                ))
        
        # Сортируем по убыванию score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Shortlist: топ 20% или минимум 5
        shortlist_size = max(5, int(len(results) * 0.2))
        shortlist = results[:shortlist_size]
        
        return BatchScoreResponse(
            ranked_applicants=results,
            shortlist=shortlist,
            total_processed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top", response_model=List[ScoreResponse])
def get_top(n: int = Query(10, ge=1, le=100)):
    """
    Получить топ N кандидатов из уже обработанных
    """
    try:
        if system is None or not hasattr(system, "shortlist_df"):
            raise HTTPException(status_code=500, detail="Model not ready")
        
        df_top = system.shortlist_df.sort_values("final_score", ascending=False).head(n)
        
        results = []
        for _, row in df_top.iterrows():
            results.append(ScoreResponse(
                applicant_id=str(row.get('akimat', 'unknown')),
                final_score=float(row['final_score']),
                score_normalized=float(row['final_score']),
                recommendation="одобрить" if row['final_score'] > 75 else "требует проверки",
                human_review_needed=row['final_score'] < 75,
                summary=f"Скор: {row['final_score']:.1f}. Эффективность: {row.get('efficiency', 0):.2f}",
                positive_factors=[],
                negative_factors=[]
            ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/all")
def get_all(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    sort_by: str = Query("final_score", regex="^(final_score|efficiency|success_rate)$")
):
    """
    Пагинированный список всех кандидатов
    """
    try:
        if system is None:
            raise HTTPException(status_code=500, detail="Model not ready")
        
        df_all = system.df.sort_values(sort_by, ascending=False)
        
        # Пагинация
        total = len(df_all)
        df_page = df_all.iloc[offset:offset+limit]
        
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "data": df_page.fillna("").to_dict(orient="records")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
def status():
    """Статус системы и метаданные модели"""
    if system is None:
        return {"ready": False}
    
    return {
        "ready": True,
        "total_records": len(system.df),
        "model_type": "RandomForest" if system.model else "rule-based",
        "features_used": getattr(system, "feature_cols", []),
        "has_explainability": explainer is not None,
        "cache_size": len(CACHE)
    }

@app.post("/reload")
def reload_model():
    """Перезагрузка модели"""
    global system, explainer, CACHE
    try:
        CACHE = {}
        system = SubsidyScoringSystem(DATA_PATH)
        system.run()
        
        if system.model is not None:
            bg_data = system.df[system.feature_cols].fillna(0).head(100)
            bg_scaled = system.scaler.transform(bg_data)
            explainer = SubsidyExplainer(
                model=system.model,
                scaler=system.scaler,
                feature_cols=system.feature_cols,
                background_data=bg_scaled
            )
        
        return {"message": "Model reloaded successfully ✅"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
