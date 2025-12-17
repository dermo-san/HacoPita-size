"""
Application constants such as the ordered list of feature columns used for inference.
"""

from typing import Final, List

# Ordered feature columns generated from Azure AutoML scoring script.
FEATURE_COLUMNS: Final[List[str]] = [
    "total_items",
    "bonsai",
    "other",
    "plastic_pots_trays",
    "single_flower_vase",
    "decorative_sand",
    "saucers_mats",
    "books",
    "suiban",
    "bonsai_seeds",
    "for_bonsai_classes",
    "bonsai_soil",
    "bonsai_tools",
    "bonsai_pots",
    "bonsai_decorations",
    "lucky_bag",
    "moss",
    "moss_bonsai",
    "chemicals_fertilizer",
    "wire",
    "decorative_stones",
    "accessories",
    "max_item_long_cm",
    "sum_item_volume_cm3",
]

# Output column names used throughout the application.
BOX_ID_COLUMN: Final[str] = "box_id"
BOX_ID_PRED_COLUMN: Final[str] = "box_id_pred"
