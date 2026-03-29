import pandas as pd
import numpy as np
import shap

class SubsidyExplainer:
    def __init__(self, model, scaler, feature_cols, background_data):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        # SHAP explainer на подвыборке
        self.explainer = shap.TreeExplainer(model, background_data)
    
    def explain_prediction(self, features_dict):
        """
        Возвращает human-readable explanation для одного заявителя
        """
        # Преобразуем вход в DataFrame
        input_df = pd.DataFrame([features_dict])[self.feature_cols].fillna(0)
        input_scaled = self.scaler.transform(input_df)
        
        # Предсказание
        score = self.model.predict(input_scaled)[0]
        
        # SHAP значения
        shap_values = self.explainer.shap_values(input_scaled)
        
        # Формируем объяснение
        explanation = {
            "final_score": float(score),
            "score_normalized": float(100 * (score - 0) / (100 - 0)),  # адаптируйте под свой диапазон
            "positive_factors": [],
            "negative_factors": [],
            "recommendation": "одобрить" if score > 75 else "требует проверки"
        }
        
        # Анализируем SHAP значения
        for i, col in enumerate(self.feature_cols):
            impact = shap_values[0][i]
            value = input_df[col].values[0]
            
            factor_text = self._factor_description(col, value)
            
            if impact > 0:
                explanation["positive_factors"].append({
                    "factor": factor_text,
                    "impact": float(impact),
                    "value": float(value)
                })
            elif impact < 0:
                explanation["negative_factors"].append({
                    "factor": factor_text,
                    "impact": float(impact),
                    "value": float(value)
                })
        
        # Сортируем по важности
        explanation["positive_factors"].sort(key=lambda x: x["impact"], reverse=True)
        explanation["negative_factors"].sort(key=lambda x: x["impact"])
        
        # Добавляем human-readable резюме
        explanation["summary"] = self._generate_summary(explanation)
        explanation["human_review_needed"] = score < 75  # порог для человека
        
        return explanation
    
    def _factor_description(self, col, value):
        """Превращает техническое название в понятное"""
        descriptions = {
            "efficiency": "эффективность использования субсидий",
            "success_rate": "процент успешно исполненных заявок",
            "application_count": "количество поданных заявок",
            "total_subsidy": "общая сумма полученных субсидий",
            "stability": "стабильность результатов",
            "eff_vs_region": "эффективность относительно региона"
        }
        return descriptions.get(col, col)
    
    def _generate_summary(self, explanation):
        """Генерирует текст на русском"""
        positive = explanation["positive_factors"][:2]
        negative = explanation["negative_factors"][:2]
        
        summary = ""
        if positive:
            summary += f"✅ Положительные факторы: {', '.join([p['factor'] for p in positive])}. "
        if negative:
            summary += f"⚠️ Отрицательные факторы: {', '.join([n['factor'] for n in negative])}. "
        
        if explanation["final_score"] >= 75:
            summary += "Рекомендуется одобрение субсидии."
        else:
            summary += "Рекомендуется дополнительная проверка комиссией."
        
        return summary
