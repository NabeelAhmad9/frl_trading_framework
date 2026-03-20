"""Data workflow — orchestrates data loading, preprocessing, feature engineering, and persistence."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.data.loader import load_raw_pair
from src.data.preprocessing import preprocess_pair
from src.utils.config_loader import resolve_config, save_yaml
from src.utils.logger import get_logger
from src.utils.progress import (
    get_progress_colour,
    progress_iter,
    progress_logging_redirect,
    set_bar_metrics,
)

logger = get_logger(__name__)


def run_data_workflow(
    config: Dict[str, Any],
    pairs: Optional[List[str]] = None,
    root: Optional[str] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Build processed train/test datasets for the given pairs.

    Parameters
    ----------
    config : dict
        Resolved configuration.
    pairs : list of str, optional
        Pairs to process. Defaults to all pairs in config.
    root : str, optional
        Repository root for resolving paths.

    Returns
    -------
    dict
        Mapping of pair -> {'train': df, 'test': df}.
    """
    data_cfg = config.get("data", {})
    all_pairs = data_cfg.get("pairs", [])
    if pairs is None:
        pairs = all_pairs
    pairs = list(pairs)

    raw_dir = Path(root or ".") / data_cfg.get("raw_data_dir", "data/raw")
    processed_dir = Path(root or ".") / data_cfg.get("processed_data_dir", "data/processed")

    results = {}

    with progress_logging_redirect(config):
        pair_pbar = progress_iter(
            pairs,
            config,
            desc="Data pairs",
            total=len(pairs),
            position=0,
            leave=True,
            unit="pair",
            colour=get_progress_colour(config, "pair"),
        )

        for pair in pair_pbar:
            logger.info("=" * 60)
            logger.info("Processing pair: %s", pair)

            # Load raw
            raw_df = load_raw_pair(pair, raw_dir=raw_dir)

            # Preprocess and split
            train_df, test_df = preprocess_pair(raw_df, config)

            # Feature engineering (if available, will be plugged in Phase 3)
            try:
                from src.features.pipeline import run_feature_pipeline
                train_df, test_df = run_feature_pipeline(train_df, test_df, config)
            except ImportError:
                logger.info("Feature pipeline not yet available, saving raw-cleaned splits.")

            split_items = [("train", train_df), ("test", test_df)]
            split_pbar = progress_iter(
                split_items,
                config,
                desc=f"{pair} splits",
                total=len(split_items),
                position=1,
                leave=False,
                unit="split",
                colour=get_progress_colour(config, "workflow"),
            )

            # Save processed
            for split_name, split_df in split_pbar:
                out_path = processed_dir / pair / f"{split_name}.parquet"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                split_df.to_parquet(out_path, index=False)
                logger.info("Saved %s/%s: %d rows -> %s", pair, split_name, len(split_df), out_path)

            set_bar_metrics(
                pair_pbar,
                metrics={
                    "pair": pair,
                    "train_rows": len(train_df),
                    "test_rows": len(test_df),
                },
                refresh=False,
            )
            results[pair] = {"train": train_df, "test": test_df}

    return results
