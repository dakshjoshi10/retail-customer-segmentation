"""
Data loading utilities for Kaggle E-Commerce Data.
"""

import logging
from pathlib import Path
from typing import Optional
import pandas as pd
from src.config import RAW_DATA_FILE, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_raw_data(filepath: Optional[Path] = None, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    """
    Loads raw e-commerce dataset from CSV with specified encoding.
    """
    target_path = filepath or RAW_DATA_FILE
    if not target_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {target_path}. Please download it via Kaggle CLI or manually."
        )
    logger.info(f"Loading raw data from {target_path}...")
    df = pd.read_csv(target_path, encoding=encoding)
    logger.info(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns.")
    return df
