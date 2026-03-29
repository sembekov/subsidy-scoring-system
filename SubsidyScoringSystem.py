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


class SubsidyScoringSystem:

    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
    def load_data(self):
        # Read raw CSV to detect header row
        df_raw = pd.read_csv(self.file_path, header=None, engine="python")
    
        header_row = None
        for i, row in df_raw.iterrows():
            row_str = " ".join(map(str, row.values))
            if "Дата" in row_str or "Область" in row_str:
                header_row = i
                break
    
        if header_row is None:
            raise ValueError("Header not found")
    
        # ✅ Correct: use header_row as the header
        self.df = pd.read_csv(
                self.file_path,
                header=header_row,
                engine="python"
                )
    
        # Clean column names immediately
        self.df.columns = self.df.columns.str.strip().str.replace('\n',' ').str.replace('\xa0',' ')
    
        print("✅ Data loaded:", self.df.shape)
        return self

    # =========================
    # 2. CLEAN
    # =========================
    def clean_data(self):
        df = self.df.copy()

        df = df.dropna(how='all')

        df.columns = (
                df.columns
                .str.strip()
                .str.replace('\n', ' ')
                )

        # Rename columns
        mapping = {
                'Дата поступления': 'date',
                'Область': 'region',
                'Акимат': 'akimat',
                'Номер заявки': 'request_id',
                'Направление водства': 'direction',
                'Наименование субсидирования': 'subsidy_type',
                'Статус заявки': 'status',
                'Норматив': 'normative',
                'Причитающая сумма': 'amount',
                'Район хозяйства': 'district'
                }

        for k, v in mapping.items():
            for col in df.columns:
                if k.lower() in col.lower():
                    df.rename(columns={col: v}, inplace=True)

        for col in ['normative', 'amount']:
            if col in df.columns:
                df[col] = (
                        df[col].astype(str)
                        .str.replace(r'[^\d,\.]', '', regex=True)
                        .str.replace(',', '.', regex=False)
                        )
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        if 'status' in df.columns:
            df['is_executed'] = (df['status'] == 'Исполнена').astype(int)

# Only filter if columns exist
        if 'amount' in df.columns and 'normative' in df.columns:
            df = df[(df['amount'] > 0) & (df['normative'] > 0)]
        else:
            print("⚠️ 'amount' or 'normative' column not found! Available columns:", df.columns.tolist())
        
        self.df = df
        print("✅ Cleaned:", len(df))
        return self
    def feature_engineering(self):
        df = self.df.copy()
        df['efficiency'] = df['amount'] / (df['normative'] + 1e-6)
        df['efficiency_ratio'] = df['amount'] / df['normative']
        df['amount_log'] = np.log1p(df['amount'])
        df['normative_log'] = np.log1p(df['normative'])
        historical = df.groupby('akimat').agg({
            'is_executed': 'mean',  # success rate
            'efficiency': ['mean', 'std', 'count'],  # efficiency history
            'amount': 'sum'  # total subsidies received
            }).fillna(0)    
        historical.columns = ['success_rate', 'avg_efficiency', 'efficiency_std', 
                              'application_count', 'total_subsidy']

        df = df.merge(historical, on='akimat', how='left')
        if 'date' in df.columns:
            df = df.sort_values(['akimat', 'date'])
            df['efficiency_lag1'] = df.groupby('akimat')['efficiency'].shift(1)
            df['efficiency_trend'] = df['efficiency'] - df['efficiency_lag1']
        region_metrics = df.groupby('region').agg({
            'efficiency': 'mean',
            'success_rate': 'mean'
            }).rename(columns={'efficiency': 'region_avg_efficiency', 
                               'success_rate': 'region_success_rate'})

        df = df.merge(region_metrics, on='region', how='left')
        df['eff_vs_region'] = df['efficiency'] - df['region_avg_efficiency']
        df['success_vs_region'] = df['success_rate'] - df['region_success_rate']
        if 'date' in df.columns:
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['day_of_week'] = df['date'].dt.dayofweek
        df['stability'] = 1 / (df['efficiency_std'] + 1e-6)
        df['stability'] = df['stability'].clip(upper=df['stability'].quantile(0.99))
        df['scale_score'] = np.log1p(df['total_subsidy'])
        df['activity'] = np.log1p(df['application_count'])

        self.df = df
        print("✅ Features created")
        return self
    def train_predictive_model(self):
        feature_cols = [
                'efficiency', 'amount_log', 'normative_log',
                'avg_efficiency', 'efficiency_std', 'success_rate',
                'eff_vs_region', 'success_vs_region',
                'stability', 'scale_score', 'activity'
                ]
        if 'month' in self.df.columns:
            feature_cols.extend(['month', 'quarter', 'day_of_week'])
        self.df['target'] = self.df.groupby('akimat')['efficiency'].shift(-1)
        df_model = self.df.dropna(subset=['target'] + feature_cols).copy()

        if len(df_model) < 50:
            print("⚠️ Not enough data for predictive model, using rule-based scoring")
            self.use_rule_based_scoring()
            return self

        X = df_model[feature_cols].fillna(0)
        y = df_model['target']
        X = X.replace([np.inf, -np.inf], 0)
        X_scaled = self.scaler.fit_transform(X)
        self.model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                           random_state=42, n_jobs=-1)
        self.model.fit(X_scaled, y)
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')

        print(f"Predictive Model Performance:")
        print(f"CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

        print("📈 Feature Importance (Top 10):")
        print(self.feature_importance.head(10))

        # Store feature columns for prediction
        self.feature_cols = feature_cols

        return self

    def use_rule_based_scoring(self):
        """Fallback rule-based scoring when data is insufficient"""
        df = self.df.copy()

        # Normalize components
        def norm(x):
            return (x - x.min()) / (x.max() - x.min() + 1e-6)

        df['eff_score'] = norm(df['efficiency'])
        df['amount_score'] = norm(df['amount'])
        df['success_score'] = norm(df['success_rate'])
        df['stability_score'] = norm(df['stability'])
        df['region_score'] = norm(df['eff_vs_region'])

        # Rule-based weights (based on domain knowledge)
        df['final_score'] = (
                0.30 * df['eff_score'] +
                0.20 * df['amount_score'] +
                0.20 * df['success_score'] +
                0.15 * df['stability_score'] +
                0.15 * df['region_score']
                )

        self.df = df
        self.model = None
        return self
    def score_applicants(self):
        df = self.df.copy()

        if self.model is not None:
            # Use predictive model
            X_pred = df[self.feature_cols].fillna(0)
            X_pred = X_pred.replace([np.inf, -np.inf], 0)
            X_scaled = self.scaler.transform(X_pred)
            df['predicted_efficiency'] = self.model.predict(X_scaled)
            df['historical_score'] = (df['efficiency'] - df['efficiency'].min()) / (df['efficiency'].max() - df['efficiency'].min() + 1e-6)
            df['future_score'] = (df['predicted_efficiency'] - df['predicted_efficiency'].min()) / (df['predicted_efficiency'].max() - df['predicted_efficiency'].min() + 1e-6)
            df['final_score'] = 0.6 * df['historical_score'] + 0.4 * df['future_score']
        else:
            if 'final_score' not in df.columns:
                print("⚠️ Using rule-based scores")
                self.use_rule_based_scoring()
                df = self.df.copy()
        df['final_score'] = 100 * (df['final_score'] - df['final_score'].min()) / (df['final_score'].max() - df['final_score'].min() + 1e-6)

        self.df = df
        print("✅ Scores calculated")
        return self
    def shortlist(self, top_n=50):
        df = self.df.copy()
        df_executed = df[df['is_executed'] == 1].copy()

        top = df_executed.sort_values('final_score', ascending=False).head(top_n)

        print(f"\n{'='*70}")
        print(f"🏆 TOP {top_n} APPLICANTS - RECOMMENDED FOR PRIORITY FUNDING")
        print(f"{'='*70}")

        display_cols = ['akimat', 'region', 'final_score', 'efficiency', 'success_rate', 
                        'application_count', 'total_subsidy']
        display_cols = [c for c in display_cols if c in top.columns]

        print(top[display_cols].head(15).to_string(index=False))
        total_score = top['final_score'].sum()
        top['recommended_percent'] = (top['final_score'] / total_score) * 100
        top['recommended_allocation'] = top['recommended_percent'] / 100

        self.shortlist_df = top
        self.recommendations = top[['akimat', 'region', 'final_score', 
                                    'recommended_percent', 'efficiency']].copy()

        return self
    def explain(self, n_candidates=5):
        df = self.shortlist_df.copy()

        print(f"\n{'='*70}")
        print("🔍 DETAILED ANALYSIS FOR TOP CANDIDATES")
        print(f"{'='*70}")

        for idx, (_, row) in enumerate(df.head(n_candidates).iterrows(), 1):
            print(f"\n{idx}. {row['akimat']} ({row['region']})")
            print(f"   Score: {row['final_score']:.1f}/100")
            print(f"   {row['score_icon'] if 'score_icon' in row else '⭐'}")
            print(f"\n   📊 Key Metrics:")
            print(f"      - Efficiency (output per input): {row['efficiency']:.2f}")
            print(f"      - Success rate: {row['success_rate']:.1%}")
            print(f"      - Applications submitted: {row['application_count']:.0f}")
            print(f"      - Total subsidies received: {row['total_subsidy']:,.0f} KZT")

            if 'predicted_efficiency' in row:
                print(f"      - Predicted future efficiency: {row['predicted_efficiency']:.2f}")
            strengths = []
            if row['efficiency'] > df['efficiency'].median():
                strengths.append("High operational efficiency")
            if row['success_rate'] > 0.8:
                strengths.append("Excellent track record")
            if row['application_count'] > df['application_count'].median():
                strengths.append("Active participant")
            if 'stability' in row and row['stability'] > df['stability'].median():
                strengths.append("Consistent performance")

            if strengths:
                print(f"\n   ✅ Strengths: {', '.join(strengths)}")

            weaknesses = []
            if row['efficiency'] < df['efficiency'].median():
                weaknesses.append("Below-average efficiency")
            if row['success_rate'] < 0.5:
                weaknesses.append("Low success rate")
            if 'stability' in row and row['stability'] < df['stability'].median():
                weaknesses.append("Inconsistent performance")

            if weaknesses:
                print(f"\n   ⚠️ Areas for Improvement: {', '.join(weaknesses)}")
                print(f"      Recommendation: {'; '.join(self._get_recommendations(weaknesses))}")

            print("-" * 60)

        return self

    def _get_recommendations(self, weaknesses):
        """Generate actionable recommendations based on weaknesses"""
        recommendations = []
        for w in weaknesses:
            if "efficiency" in w.lower():
                recommendations.append("Focus on yield optimization / cost reduction")
            elif "success rate" in w.lower():
                recommendations.append("Review application quality and documentation")
            elif "inconsistent" in w.lower():
                recommendations.append("Develop standardized operating procedures")
        return recommendations if recommendations else ["Regular performance review"]
    def visualize(self):
        df = self.df
        shortlist = self.shortlist_df

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes[0, 0].hist(df['final_score'], bins=30, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['final_score'].median(), color='red', linestyle='--', 
                           label=f'Median: {df["final_score"].median():.1f}')
        axes[0, 0].set_title('Distribution of Applicant Scores')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        region_scores = df.groupby('region')['final_score'].mean().sort_values(ascending=False).head(10)
        axes[0, 1].barh(range(len(region_scores)), region_scores.values)
        axes[0, 1].set_yticks(range(len(region_scores)))
        axes[0, 1].set_yticklabels(region_scores.index, fontsize=10)
        axes[0, 1].set_title('Top 10 Regions by Average Score')
        axes[0, 1].set_xlabel('Average Score')
        axes[0, 2].scatter(df['efficiency'], df['final_score'], alpha=0.5)
        axes[0, 2].set_title('Efficiency vs Score')
        axes[0, 2].set_xlabel('Efficiency (Amount/Normative)')
        axes[0, 2].set_ylabel('Score')
        if len(shortlist) > 0:
            top_candidates = shortlist.head(15)
            axes[1, 0].barh(range(len(top_candidates)), top_candidates['final_score'].values)
            axes[1, 0].set_yticks(range(len(top_candidates)))
            axes[1, 0].set_yticklabels([f"{row['akimat'][:30]}" for _, row in top_candidates.iterrows()], fontsize=8)
            axes[1, 0].set_title('Top 15 Candidates by Score')
            axes[1, 0].set_xlabel('Score')
        if 'success_rate' in df.columns:
            axes[1, 1].hist(df['success_rate'].dropna(), bins=20, edgecolor='black', alpha=0.7)
            axes[1, 1].set_title('Distribution of Success Rates')
            axes[1, 1].set_xlabel('Success Rate')
            axes[1, 1].set_ylabel('Frequency')
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1, 2].barh(range(len(top_features)), top_features['importance'].values)
            axes[1, 2].set_yticks(range(len(top_features)))
            axes[1, 2].set_yticklabels(top_features['feature'].values, fontsize=9)
            axes[1, 2].set_title('Top 10 Features by Importance')
            axes[1, 2].set_xlabel('Importance')

        plt.tight_layout()
        plt.savefig('subsidy_scoring_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

        print("\n✅ Visualizations saved to 'subsidy_scoring_analysis.png'")
        return self
    def export_results(self):
        results = self.df[['akimat', 'region', 'district', 'final_score', 
                           'efficiency', 'success_rate', 'application_count']].copy()
        results = results.sort_values('final_score', ascending=False)
        results.to_csv('full_scores.csv', index=False)
        if hasattr(self, 'shortlist_df'):
            self.shortlist_df.to_csv('shortlist_recommendations.csv', index=False)
        regional_summary = self.df.groupby('region').agg({
            'final_score': 'mean',
            'akimat': 'count',
            'efficiency': 'mean',
            'success_rate': 'mean'
            }).round(3).sort_values('final_score', ascending=False)
        regional_summary.to_csv('regional_summary.csv')
        print("\n✅ Results exported:")
        print("   - full_scores.csv (all applicants with scores)")
        print("   - shortlist_recommendations.csv (top candidates)")
        print("   - regional_summary.csv (regional statistics)")
        print(self.df['final_score'])
        return self
    def run(self):
        print("="*70)
        print("MERIT-BASED SUBSIDY SCORING SYSTEM".center(70))
        print("="*70)
    
        return (
            self.load_data()
                .clean_data()
                .feature_engineering()
                .train_predictive_model()
                .score_applicants()
                .shortlist()
                .explain()
                .visualize()
                .export_results()
        )
if __name__ == "__main__":
    system = SubsidyScoringSystem(
            "Выгрузка по выданным субсидиям 2025 год (обезлич).xlsx - Page 1.csv"
            )
    system.run()

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
