"""Stage 3: baseline regressor - XGBoost predicting the composite score."""

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr

FEATURES = [
    "Total_Orders",
    "On_Time_Pct",
    "Rejection_Pct",
    "Quality_Pct",
    "Avg_Delivery_Days",
    "Avg_Price_Per_Ton",
    "Total_Purchase_Value",
]
TARGET = "Composite_Score"


def train_baseline_regressor(train_df, val_df):
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
    )
    model.fit(
        train_df[FEATURES],
        train_df[TARGET],
        eval_set=[(val_df[FEATURES], val_df[TARGET])],
        verbose=False,
    )
    return model


def evaluate_baseline_regressor(model, df):
    preds = model.predict(df[FEATURES])
    mae = mean_absolute_error(df[TARGET], preds)
    r2 = r2_score(df[TARGET], preds)
    rho, _ = spearmanr(df[TARGET], preds)
    return {"MAE": float(mae), "R2": float(r2), "Spearman": float(rho)}, preds  # type: ignore[arg-type]
