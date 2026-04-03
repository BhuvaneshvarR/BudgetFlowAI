import json
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import numpy as np
import kagglehub
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from xgboost import XGBRegressor
import joblib
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Re-define BudgetAnalysisAPI class
class BudgetAnalysisAPI:
    def __init__(self, budget_ai_df, optimization_summary_df):
        self.budget_ai = budget_ai_df
        self.optimization_summary = optimization_summary_df

    def get_overall_health_summary(self):
        total_anomalies = (self.budget_ai['anomaly'] == 'overspending').sum()
        avg_risk = self.budget_ai['budget_risk'].mean()

        if avg_risk > 0.05 or total_anomalies > 200:
            status = 'Critical'
        elif avg_risk > 0.01 or total_anomalies > 50:
            status = 'At Risk'
        else:
            status = 'Stable'

        return {
            "status": status,
            "total_anomalies_detected": int(total_anomalies),
            "average_budget_risk_score": round(float(avg_risk), 4)
        }

    def get_top_optimization_recommendations(self, num_recommendations=5):
        top_recommendations = self.optimization_summary.sort_values(
            by=['budget_risk', 'anomaly_count'], ascending=False
        ).head(num_recommendations)

        recommendations_list = []
        for _, row in top_recommendations.iterrows():
            recommendations_list.append({
                "category": row['category'],
                "cluster": int(row['cluster']),
                "average_spend": round(float(row['spend']), 2),
                "suggested_reduction": round(float(row['suggested_reduction']), 2),
                "budget_risk": round(float(row['budget_risk']), 4),
                "priority": row['optimization_priority'],
                "advice": row['advice']
            })
        return recommendations_list

    def generate_analysis_report(self):
        report = {
            "overall_health": self.get_overall_health_summary(),
            "optimization_priorities": self.get_top_optimization_recommendations(),
            "metadata": {
                "report_type": "BudgetAnalysis",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        return json.dumps(report, indent=4)

# Re-define ChatAdviserAPI class
class ChatAdviserAPI:
    def __init__(self, budget_ai_df, optimization_summary_df):
        self.budget_ai = budget_ai_df
        self.optimization_summary = optimization_summary_df

    def get_financial_health_summary(self):
        total_anomalies = (self.budget_ai['anomaly'] == 'overspending').sum()
        avg_risk = self.budget_ai['budget_risk'].mean()

        if avg_risk > 0.05 or total_anomalies > 200:
            status = 'Critical'
            summary_text = "Your financial health is in a **critical state**. Immediate action is required to address significant overspending and high budget risks."
        elif avg_risk > 0.01 or total_anomalies > 50:
            status = 'At Risk'
            summary_text = "Your financial health is **at risk**. There are notable overspending incidents and areas that require attention to avoid further issues."
        else:
            status = 'Stable'
            summary_text = "Your financial health appears **stable**. Spending is mostly within predicted limits, with minimal anomalies."

        return {
            "status": status,
            "total_anomalies": int(total_anomalies),
            "average_budget_risk": round(float(avg_risk), 4),
            "summary_text": summary_text
        }

    def get_natural_language_recommendations(self, num_recommendations=3):
        high_priority = self.optimization_summary[self.optimization_summary['optimization_priority'] == 'High'].sort_values(by='budget_risk', ascending=False).head(num_recommendations)

        recommendations_list = []
        if not high_priority.empty:
            recommendations_list.append("Here are the top areas where you can optimize your budget:")
            for _, row in high_priority.iterrows():
                recommendations_list.append(f"- **{row['category']}**: You are advised to {row['advice']}. This category has a budget risk of {row['budget_risk']:.2f}.")
        else:
            recommendations_list.append("No high-priority budget reductions are identified at this time. Keep up the good work!")

        return recommendations_list

    def generate_chat_advice(self):
        health_data = self.get_financial_health_summary()
        recommendations = self.get_natural_language_recommendations()

        advice_report = {
            "overall_health": health_data["summary_text"],
            "details": {
                "status": health_data["status"],
                "total_anomalies_detected": health_data["total_anomalies"],
                "average_budget_risk_score": health_data["average_budget_risk"]
            },
            "recommendations": recommendations,
            "metadata": {
                "report_type": "FinancialChatAdvice",
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        return json.dumps(advice_report, indent=4)

# Define and apply the ai_budget_optimizer function to ensure optimization_summary is available
def ai_budget_optimizer(df):
    summary = df.groupby(['category', 'cluster']).agg({
        'spend': 'mean',
        'predicted_spend': 'mean',
        'budget_risk': 'mean',
        'anomaly': lambda x: (x == 'overspending').sum()
    }).reset_index()

    summary.rename(columns={'anomaly': 'anomaly_count'}, inplace=True)

    summary['is_high_risk'] = (summary['spend'] > summary['predicted_spend'] * 1.2) | (summary['anomaly_count'] > 0)

    summary['suggested_reduction'] = summary.apply(
        lambda x: x['spend'] * 0.15 if x['is_high_risk'] else 0.0, axis=1
    )

    def assign_priority(row):
        if row['budget_risk'] > 0.01 or row['anomaly_count'] > 5:
            return 'High'
        elif row['is_high_risk']:
            return 'Medium'
        else:
            return 'Low'

    summary['optimization_priority'] = summary.apply(assign_priority, axis=1)

    summary['advice'] = summary.apply(
        lambda x: f"Reduce spending by {x['suggested_reduction']:.2f} in {x['category']}" if x['is_high_risk'] else "Spending within limits", axis=1
    )

    return summary.sort_values(by=['budget_risk', 'anomaly_count'], ascending=False)

def prepare_and_train_financial_models(dataset_name="ismetsemedov/personal-budget-transactions-dataset"):
    """
    Consolidates data loading, preprocessing, feature engineering, clustering,
    anomaly detection, and spending prediction into a single function.
    Now with model persistence for faster startups.
    """
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    budget_ai_path = os.path.join(model_dir, "budget_ai.joblib")
    iso_forest_path = os.path.join(model_dir, "iso_forest.joblib")
    xgb_model_path = os.path.join(model_dir, "xgb_model.joblib")
    scaled_features_path = os.path.join(model_dir, "scaled_features.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    le_path = os.path.join(model_dir, "le.joblib")
    kmeans_path = os.path.join(model_dir, "kmeans.joblib")

    # Check if all models exist
    if all(os.path.exists(p) for p in [budget_ai_path, iso_forest_path, xgb_model_path, scaled_features_path, scaler_path, le_path, kmeans_path]):
        logger.info("Loading models from disk...")
        budget_ai = joblib.load(budget_ai_path)
        iso_forest = joblib.load(iso_forest_path)
        xgb_model = joblib.load(xgb_model_path)
        scaled_features = joblib.load(scaled_features_path)
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        kmeans = joblib.load(kmeans_path)
        return budget_ai, iso_forest, xgb_model, scaled_features, scaler, le, kmeans

    # 1. Download and load dataset
    logger.info("Retraining models: downloading dataset...")
    path = kagglehub.dataset_download(dataset_name)
    file_path = os.path.join(path, "budget_data.csv")
    budget_df = pd.read_csv(file_path)

    # 2. Initialize budget_ai and rename amount to spend
    budget_ai = budget_df[['date', 'category', 'amount']].copy()
    budget_ai.rename(columns={'amount': 'spend'}, inplace=True)

    # 3. Cleaning
    budget_ai['date'] = pd.to_datetime(budget_ai['date'])
    budget_ai = budget_ai[budget_ai['spend'] > 0]
    budget_ai = budget_ai.dropna().drop_duplicates().reset_index(drop=True)

    # 4. Feature Extraction
    budget_ai['month'] = budget_ai['date'].dt.month
    budget_ai['day'] = budget_ai['date'].dt.day 
    budget_ai['hour'] = budget_ai['date'].dt.hour

    # 5. Encoding
    le = LabelEncoder()
    budget_ai['category_encoded'] = le.fit_transform(budget_ai['category'])

    # 6. Scaling
    feature_cols = ['spend', 'month', 'hour', 'category_encoded']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(budget_ai[feature_cols])

    # 7. K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    budget_ai['cluster'] = kmeans.fit_predict(scaled_features)

    # 8. Isolation Forest for Overspending Detection
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    budget_ai['anomaly_score'] = iso_forest.fit_predict(scaled_features)
    budget_ai['anomaly'] = budget_ai['anomaly_score'].map({1: 'normal', -1: 'overspending'})

    # 9. XGBoost Regressor for Spending Prediction
    X_model = budget_ai[['month', 'hour', 'category_encoded', 'cluster']]
    y_model = budget_ai['spend']

    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_model.fit(X_model, y_model)

    # 10. Generate predictions for the entire dataset
    budget_ai['predicted_spend'] = xgb_model.predict(X_model)

    # 11. Calculate budget_risk score (normalized absolute deviation)
    deviation = np.abs(budget_ai['spend'] - budget_ai['predicted_spend'])
    budget_ai['budget_risk'] = (deviation - deviation.min()) / (deviation.max() - deviation.min())

    logger.info("Saving models to disk...")
    joblib.dump(budget_ai, budget_ai_path)
    joblib.dump(iso_forest, iso_forest_path)
    joblib.dump(xgb_model, xgb_model_path)
    joblib.dump(scaled_features, scaled_features_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(le, le_path)
    joblib.dump(kmeans, kmeans_path)

    logger.info("Data preparation and model training complete.")
    return budget_ai, iso_forest, xgb_model, scaled_features, scaler, le, kmeans

# Prepare data and models
budget_ai, iso_forest, xgb_model, scaled_features, scaler, le, kmeans = prepare_and_train_financial_models()

# Apply the optimizer to create optimization_summary
optimization_summary = ai_budget_optimizer(budget_ai)

# Initialize your API classes globally
budget_analysis_service = BudgetAnalysisAPI(budget_ai, optimization_summary)
chat_adviser_service = ChatAdviserAPI(budget_ai, optimization_summary)

# Initialize FastAPI app
app = FastAPI()

# Define a request body model if you expect dynamic input for suggestions
class SuggestionInput(BaseModel):
    category_ids: list[str] = [] # Example of filtering recommendations
    min_risk: float = 0.0 # Example of filtering recommendations

@app.get("/health")
def health_check():
    """Health check endpoint for Render.com."""
    return {"status": "healthy", "model_loaded": True}

@app.get("/budget-analysis")
def get_budget_analysis_report():
    """Endpoint for the Budget Suggestions API."""
    return json.loads(budget_analysis_service.generate_analysis_report())

@app.get("/chat-advice")
def get_chat_advice_report():
    """Endpoint for the Chat Adviser API."""
    return json.loads(chat_adviser_service.generate_chat_advice())

@app.post("/dynamic-suggestions")
def get_dynamic_suggestions(input_data: SuggestionInput):
    """Example of an endpoint that could take dynamic input."""
    filtered_recs = [rec for rec in json.loads(budget_analysis_service.generate_analysis_report())['optimization_priorities']
                     if rec['budget_risk'] >= input_data.min_risk]
    if input_data.category_ids: # Example of using input to filter
        filtered_recs = [rec for rec in filtered_recs if rec['category'] in input_data.category_ids]

    return {"filtered_recommendations": filtered_recs}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
