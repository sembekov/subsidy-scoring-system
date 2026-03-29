from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from SubsidyScoringSystem import SubsidyScoringSystem
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")


app = FastAPI(title="Subsidy Scoring API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для демонстрации, на проде лучше указать домен
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# INIT SYSTEM
# =========================
DATA_PATH = "Выгрузка_по_выданным_субсидиям_2025_год_обезлич_xlsx_Page_1.csv"

system = None
try:
    system = SubsidyScoringSystem(DATA_PATH)
    system.run()
except Exception as e:
    print("❌ Error during initialization:", e)


# =========================
# SCHEMAS
# =========================
class TopRequest(BaseModel):
    n: int = 10


# =========================
# ENDPOINTS
# =========================

@app.get("/")
def root():
    return {"message": "Subsidy Scoring API is running 🚀"}


@app.get("/top")
def get_top(n: int = 10):
    """
    Получить топ N кандидатов
    """
    try:
        if system is None or not hasattr(system, "shortlist_df"):
            raise HTTPException(status_code=500, detail="Model not ready")

        df_top = system.shortlist_df.sort_values("final_score", ascending=False).head(n)
        df_top = df_top.fillna("").to_dict(orient="records")
        return df_top

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/all")
def get_all(limit: int = 100):
    """
    Все кандидаты с оценками
    """
    try:
        if system is None:
            raise HTTPException(status_code=500, detail="Model not ready")
        df_all = system.df.sort_values("final_score", ascending=False).head(limit)
        df_all = df_all.fillna("").to_dict(orient="records")
        return df_all

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
def status():
    if system is None:
        return {"ready": False}
    return {"ready": True, "total_records": len(system.df)}


@app.get("/reload")
def reload_model():
    """
    Перезапуск модели (если обновились данные)
    """
    global system
    try:
        system = SubsidyScoringSystem(DATA_PATH)
        system.run()
        return {"message": "Model reloaded successfully ✅"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
