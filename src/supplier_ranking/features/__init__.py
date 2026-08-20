"""Supplier feature engineering: eligibility, aggregation, scoring, and order history."""

from supplier_ranking.features.order_history_features import (
    ORDER_HISTORY_FEATURES,
    build_order_level_history_features,
)
from supplier_ranking.features.scoring import DEFAULT_WEIGHTS, compute_composite_score
from supplier_ranking.features.supplier_features import (
    FEATURE_COLUMNS,
    compute_all_supplier_features,
    compute_supplier_features,
    get_eligible_suppliers,
    get_eligible_suppliers_by_category,
)

__all__ = [
    "DEFAULT_WEIGHTS",
    "FEATURE_COLUMNS",
    "ORDER_HISTORY_FEATURES",
    "build_order_level_history_features",
    "compute_all_supplier_features",
    "compute_composite_score",
    "compute_supplier_features",
    "get_eligible_suppliers",
    "get_eligible_suppliers_by_category",
]