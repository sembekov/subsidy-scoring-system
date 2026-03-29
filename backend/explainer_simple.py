import pandas as pd
import numpy as np

class SubsidyExplainer:
    def __init__(self, model, scaler, feature_cols, feature_importance):
        """
        Простой explainer на основе важности признаков из RandomForest
        """
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.feature_importance = feature_importance  # DataFrame с колонками feature, importance
        
    def explain_prediction(self, input_data):
        """
        Возвращает объяснение для одного заявителя
        """
        # Преобразуем вход в DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Заполняем缺失ные фичи нулями
        for col in self.feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[self.feature_cols].fillna(0)
        
        # Масштабируем
        input_scaled = self.scaler.transform(input_df)
        
        # Предсказание
        score = self.model.predict(input_scaled)[0]
        
        # Нормализуем скор в 0-100
        score_normalized = min(100, max(0, score * 100))
        
        # Определяем пороги
        if score_normalized >= 75:
            recommendation = "одобрить"
            human_review_needed = False
        elif score_normalized >= 50:
            recommendation = "требует проверки"
            human_review_needed = True
        else:
            recommendation = "отклонить"
            human_review_needed = True
        
        # Анализируем вклад каждого признака (на основе важности и значения)
        positive_factors = []
        negative_factors = []
        
        # Получаем важность признаков из обученной модели
        importance_dict = dict(zip(self.feature_importance['feature'], 
                                   self.feature_importance['importance']))
        
        for i, col in enumerate(self.feature_cols):
            value = input_df[col].values[0]
            importance = importance_dict.get(col, 0)
            
            # Определяем, хорошее или плохое значение
            is_positive = self._is_positive_factor(col, value, importance)
            
            factor_text = self._factor_description(col, value)
            
            if is_positive and importance > 0.01:
                positive_factors.append({
                    "factor": factor_text,
                    "importance": float(importance),
                    "value": float(value)
                })
            elif not is_positive and importance > 0.01:
                negative_factors.append({
                    "factor": factor_text,
                    "importance": float(importance),
                    "value": float(value)
                })
        
        # Сортируем по важности
        positive_factors.sort(key=lambda x: x["importance"], reverse=True)
        negative_factors.sort(key=lambda x: x["importance"], reverse=True)
        
        # Ограничиваем топ-3
        positive_factors = positive_factors[:3]
        negative_factors = negative_factors[:3]
        
        # Генерируем суммари
        summary = self._generate_summary(positive_factors, negative_factors, recommendation)
        
        return {
            "final_score": float(score_normalized),
            "score_normalized": float(score_normalized),
            "recommendation": recommendation,
            "human_review_needed": human_review_needed,
            "summary": summary,
            "positive_factors": positive_factors,
            "negative_factors": negative_factors
        }
    
    def _is_positive_factor(self, col, value, importance):
        """
        Определяет, является ли значение признака положительным фактором
        """
        # Пороги для разных признаков (на основе медианы или среднего)
        thresholds = {
            "efficiency": 1.0,  # выше 1.0 - хорошо
            "success_rate": 0.7,  # выше 70% - хорошо
            "application_count": 5,  # больше 5 заявок - хорошо
            "total_subsidy": 1000000,  # больше 1M - хорошо
            "stability": 0.5,  # выше 0.5 - стабильно
            "eff_vs_region": 0,  # выше среднего по региону - хорошо
            "success_vs_region": 0,
            "avg_efficiency": 1.0,
            "efficiency_std": 0.3,  # низкое стандартное отклонение - хорошо
            "amount_log": 10,
            "normative_log": 10
        }
        
        if col in thresholds:
            if col in ["efficiency_std"]:
                return value < thresholds[col]  # чем меньше std, тем лучше
            else:
                return value > thresholds[col]
        
        # По умолчанию: чем больше значение, тем лучше
        return value > 0
    
    def _factor_description(self, col, value):
        """Человеко-читаемое описание фактора"""
        descriptions = {
            "efficiency": f"эффективность использования субсидий ({value:.2f})",
            "success_rate": f"процент успешных заявок ({value:.1%})",
            "application_count": f"количество поданных заявок ({int(value)})",
            "total_subsidy": f"общая сумма полученных субсидий ({value:,.0f} ₸)",
            "stability": f"стабильность результатов ({value:.2f})",
            "eff_vs_region": f"эффективность относительно региона ({value:+.2f})",
            "success_vs_region": f"успешность относительно региона ({value:+.2f})",
            "avg_efficiency": f"средняя эффективность ({value:.2f})",
            "efficiency_std": f"вариативность эффективности ({value:.2f})",
            "amount_log": f"логарифм суммы субсидий ({value:.1f})",
            "normative_log": f"логарифм норматива ({value:.1f})"
        }
        return descriptions.get(col, f"{col}: {value:.2f}")
    
    def _generate_summary(self, positive, negative, recommendation):
        """Генерирует текст на русском"""
        summary_parts = []
        
        if positive:
            pos_text = ", ".join([p['factor'] for p in positive[:2]])
            summary_parts.append(f"✅ Сильные стороны: {pos_text}")
        
        if negative:
            neg_text = ", ".join([n['factor'] for n in negative[:2]])
            summary_parts.append(f"⚠️ Слабые стороны: {neg_text}")
        
        if recommendation == "одобрить":
            summary_parts.append("Рекомендуется одобрение субсидии.")
        elif recommendation == "требует проверки":
            summary_parts.append("Требуется дополнительная проверка комиссией.")
        else:
            summary_parts.append("Рекомендуется отклонить заявку.")
        
        return " ".join(summary_parts)
