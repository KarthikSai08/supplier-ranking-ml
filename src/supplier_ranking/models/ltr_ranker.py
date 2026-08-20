"""Stage 4: learning-to-rank (LambdaMART, XGBRanker rank:ndcg)."""

import numpy as np
import xgboost as xgb

from supplier_ranking.models.baseline_regressor import FEATURES


def make_relevance_grades(train_scores, target_scores, n_grades=5):
    edges = np.quantile(train_scores, np.linspace(0, 1, n_grades + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(len(target_scores), dtype=int)
    inner_edges = edges[1:-1]
    grades = np.digitize(target_scores, inner_edges, right=True)
    return grades


def train_ltr_ranker(train_df, val_df):
    train_sorted = train_df.sort_values("Query_ID").reset_index(drop=True)
    val_sorted = val_df.sort_values("Query_ID").reset_index(drop=True)
    train_group = train_sorted.groupby("Query_ID").size().values
    val_group = val_sorted.groupby("Query_ID").size().values

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(
        train_sorted[FEATURES],
        train_sorted["Relevance"],
        group=train_group,
        eval_set=[(val_sorted[FEATURES], val_sorted["Relevance"])],
        eval_group=[val_group],
        verbose=False,
    )
    return model


def score_ranker(model, df):
    return model.predict(df[FEATURES])
