"""Stage 5 sub-models: price, delivery, quality, risk, with a shared registry."""

from dataclasses import dataclass

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, f1_score

from supplier_ranking.features.order_history_features import ORDER_HISTORY_FEATURES

FEATURES = ORDER_HISTORY_FEATURES


@dataclass(frozen=True)
class SubModel:
    """One per-order forecasting model."""
    target: str
    kind: str  # "regression" or "classification"


SUBMODELS = {
    "Price": SubModel(target="Target_Price", kind="regression"),
    "Delivery": SubModel(target="Target_Delivery_Days", kind="regression"),
    "Quality": SubModel(target="Target_Quality", kind="regression"),
    "Risk": SubModel(target="Target_Risk", kind="classification"),
}


def train_submodel(spec: SubModel, train_df, val_df):
    """Train the sub-model for a spec; dispatches on its kind."""
    if spec.kind == "classification":
        return _train_classifier(train_df, val_df, spec.target)
    return _train_regressor(train_df, val_df, spec.target)


def evaluate_submodel(spec: SubModel, model, df, threshold=0.5):
    """Evaluate a sub-model; returns (metrics dict, predictions)."""
    if spec.kind == "classification":
        return _evaluate_classifier(model, df, spec.target, threshold)
    return _evaluate_regressor(model, df, spec.target)


def find_best_threshold(model, val_df):
    """Pick the classification threshold on validation that maximizes F1."""
    probs = model.predict_proba(val_df[FEATURES])[:, 1]
    best_t, best_f1 = 0.5, -1.0
    for t in [i / 100 for i in range(5, 96, 5)]:
        preds = (probs >= t).astype(int)
        f1 = f1_score(val_df["Target_Risk"], preds, zero_division=0)  # type: ignore[arg-type]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def _train_regressor(train_df, val_df, target):
    model = xgb.XGBRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42
    )
    model.fit(train_df[FEATURES], train_df[target], eval_set=[(val_df[FEATURES], val_df[target])], verbose=False)
    return model


def _evaluate_regressor(model, df, target):
    preds = model.predict(df[FEATURES])
    mae = mean_absolute_error(df[target], preds)
    r2 = r2_score(df[target], preds)
    return {"MAE": float(mae), "R2": float(r2)}, preds


def _train_classifier(train_df, val_df, target):
    pos = train_df[target].sum()
    neg = len(train_df) - pos
    model = xgb.XGBClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", scale_pos_weight=(neg / pos) if pos > 0 else 1.0,
    )
    model.fit(train_df[FEATURES], train_df[target], eval_set=[(val_df[FEATURES], val_df[target])], verbose=False)
    return model


def _evaluate_classifier(model, df, target, threshold):
    probs = model.predict_proba(df[FEATURES])[:, 1]
    preds = (probs >= threshold).astype(int)
    auc = roc_auc_score(df[target], probs) if df[target].nunique() > 1 else float("nan")
    f1 = f1_score(df[target], preds, zero_division=0)  # type: ignore[arg-type]
    return {"AUC": float(auc), "F1": float(f1), "Threshold": threshold}, probs