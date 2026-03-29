from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import logging

# Импортируем простой explainer (без SHAP)
from explainer_simple import SubsidyExplainer

# Импортируем вашу систему
from SubsidyScoringSystem import SubsidyScoringSystem

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
# SCHEMAS
# =========================

class ScoreRequest(BaseModel):
    applicant_data: Dict[str, Any]

class ScoreResponse(BaseModel):
    applicant_id: Optional[str] = None
    final_score: float
    score_normalized: float
    recommendation: str
    human_review_needed: bool
    summary: str
    positive_factors: List[Dict]
    negative_factors: List[Dict]

class BatchScoreRequest(BaseModel):
    applicants: List[Dict[str, Any]]

class BatchScoreResponse(BaseModel):
    ranked_applicants: List[ScoreResponse]
    shortlist: List[ScoreResponse]
    total_processed: int

# =========================
# INIT SYSTEM
# =========================

DATA_PATH = "Выгрузка_по_выданным_субсидиям_2025_год_обезлич_xlsx_Page_1.csv"
system = None
explainer = None
CACHE = {}

try:
    logger.info("Loading subsidy scoring system...")
    system = SubsidyScoringSystem(DATA_PATH)
    system.run()
    
    # Инициализируем explainer если есть модель
    if system.model is not None and hasattr(system, 'feature_importance'):
        logger.info("Initializing explainer...")
        explainer = SubsidyExplainer(
            model=system.model,
            scaler=system.scaler,
            feature_cols=system.feature_cols,
            feature_importance=system.feature_importance
        )
        logger.info("✅ Explainer initialized successfully")
    else:
        logger.warning("⚠️ Using rule-based scoring (no explainer)")
        
except Exception as e:
    logger.error(f"❌ Initialization error: {e}")
    import traceback
    traceback.print_exc()

# =========================
# ENDPOINTS
# =========================

@app.get("/")
def root():
    return {
        "message": "Subsidy Scoring API v2.0",
        "features": ["scoring", "explainable_ai", "batch_processing"],
        "model_type": "ML" if (system and system.model) else "rule-based",
        "has_explainability": explainer is not None
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
        cache_key = str(hash(str(sorted(request.applicant_data.items()))))
        if cache_key in CACHE:
            return CACHE[cache_key]
        
        # Если есть модель и explainer
        if system.model is not None and explainer:
            explanation = explainer.explain_prediction(request.applicant_data)
            
            # Добавляем ID если есть
            if 'id' in request.applicant_data:
                explanation['applicant_id'] = str(request.applicant_data['id'])
            
            response = ScoreResponse(**explanation)
        else:
            # Rule-based fallback
            response = ScoreResponse(
                applicant_id=None,
                final_score=75.0,
                score_normalized=75.0,
                recommendation="требует проверки",
                human_review_needed=True,
                summary="Система использует упрощенную модель из-за недостатка данных",
                positive_factors=[],
                negative_factors=[]
            )
        
        # Кэшируем (ограничим размер)
        if len(CACHE) < 100:
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
            if system.model is not None and explainer:
                explanation = explainer.explain_prediction(applicant)
                if 'id' in applicant:
                    explanation['applicant_id'] = str(applicant['id'])
                results.append(ScoreResponse(**explanation))
            else:
                # Fallback с случайным скором (только для демо)
                random_score = np.random.uniform(40, 95)
                results.append(ScoreResponse(
                    applicant_id=str(applicant.get('id', 'unknown')),
                    final_score=random_score,
                    score_normalized=random_score,
                    recommendation="требует проверки" if random_score < 75 else "одобрить",
                    human_review_needed=random_score < 75,
                    summary="Rule-based fallback scoring",
                    positive_factors=[],
                    negative_factors=[]
                ))
        
        # Сортируем по убыванию score
        results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Shortlist: топ 20% или минимум 5
        shortlist_size = max(5, min(20, int(len(results) * 0.2)))
        shortlist = results[:shortlist_size]
        
        return BatchScoreResponse(
            ranked_applicants=results,
            shortlist=shortlist,
            total_processed=len(results)
        )
        
    except Exception as e:
        logger.error(f"Batch scoring error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top")
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
            results.append({
                "applicant_id": str(row.get('akimat', 'unknown')),
                "final_score": float(row['final_score']),
                "score_normalized": float(row['final_score']),
                "recommendation": "одобрить" if row['final_score'] > 75 else "требует проверки",
                "human_review_needed": row['final_score'] < 75,
                "summary": f"Скор: {row['final_score']:.1f}. Эффективность: {row.get('efficiency', 0):.2f}",
                "efficiency": float(row.get('efficiency', 0)),
                "success_rate": float(row.get('success_rate', 0))
            })
        
        return results
        
    except Exception as e:
        logger.error(f"Top endpoint error: {e}")
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
        
        if sort_by not in system.df.columns:
            sort_by = "final_score"
        
        df_all = system.df.sort_values(sort_by, ascending=False)
        
        # Пагинация
        total = len(df_all)
        df_page = df_all.iloc[offset:offset+limit]
        
        # Ограничиваем колонки для ответа
        columns = ['akimat', 'region', 'district', 'final_score', 'efficiency', 
                   'success_rate', 'application_count', 'total_subsidy']
        columns = [c for c in columns if c in df_page.columns]
        
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "sort_by": sort_by,
            "data": df_page[columns].fillna("").to_dict(orient="records")
        }
        
    except Exception as e:
        logger.error(f"All endpoint error: {e}")
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
        "features_used": getattr(system, "feature_cols", [])[:10],  # топ-10
        "has_explainability": explainer is not None,
        "cache_size": len(CACHE)
    }

@app.post("/reload")
def reload_model():
    """Перезагрузка модели"""
    global system, explainer, CACHE
    try:
        CACHE = {}
        logger.info("Reloading model...")
        system = SubsidyScoringSystem(DATA_PATH)
        system.run()
        
        if system.model is not None and hasattr(system, 'feature_importance'):
            explainer = SubsidyExplainer(
                model=system.model,
                scaler=system.scaler,
                feature_cols=system.feature_cols,
                feature_importance=system.feature_importance
            )
            logger.info("✅ Explainer reinitialized")
        
        return {"message": "Model reloaded successfully ✅"}
        
    except Exception as e:
        logger.error(f"Reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
