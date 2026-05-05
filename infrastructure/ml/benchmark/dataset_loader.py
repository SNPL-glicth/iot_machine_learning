"""Dataset loader — load labeled time series for benchmarking.

Single responsibility: parse CSV files in NAB or Yahoo format and
return structured samples with timestamps, values, and labels.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetSample:
    """Single sample from a labeled dataset.
    
    Attributes:
        timestamp: Unix timestamp or sequential index.
        value: Time series value.
        is_anomaly: Ground truth label (True = anomaly).
    """
    timestamp: float
    value: float
    is_anomaly: bool


class DatasetLoader:
    """Loads labeled time series datasets for benchmarking.
    
    Supports:
    - NAB format: timestamp,value,label (label ∈ {0, 1})
    - Yahoo format: timestamp,value,is_anomaly (is_anomaly ∈ {0, 1})
    
    Both formats are equivalent, just different column names.
    """
    
    @staticmethod
    def load_csv(
        file_path: str,
        timestamp_col: str = "timestamp",
        value_col: str = "value",
        label_col: str = "label",
        skip_header: bool = True,
    ) -> List[DatasetSample]:
        """Load dataset from CSV file.
        
        Args:
            file_path: Path to CSV file.
            timestamp_col: Name of timestamp column.
            value_col: Name of value column.
            label_col: Name of label column (0=normal, 1=anomaly).
            skip_header: Whether to skip first row as header.
        
        Returns:
            List of DatasetSample instances.
        
        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If CSV format is invalid.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        samples: List[DatasetSample] = []
        
        try:
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                
                for row_idx, row in enumerate(reader, start=1):
                    try:
                        # Parse timestamp
                        timestamp_str = row.get(timestamp_col)
                        if timestamp_str is None:
                            # Fallback: use row index as timestamp
                            timestamp = float(row_idx)
                        else:
                            timestamp = float(timestamp_str)
                        
                        # Parse value
                        value_str = row.get(value_col)
                        if value_str is None:
                            raise ValueError(f"Missing '{value_col}' column at row {row_idx}")
                        value = float(value_str)
                        
                        # Parse label
                        label_str = row.get(label_col)
                        if label_str is None:
                            # Try alternative column names
                            label_str = row.get("is_anomaly") or row.get("anomaly") or "0"
                        
                        is_anomaly = bool(int(label_str))
                        
                        samples.append(DatasetSample(
                            timestamp=timestamp,
                            value=value,
                            is_anomaly=is_anomaly,
                        ))
                    
                    except (ValueError, KeyError) as e:
                        logger.warning(
                            "dataset_loader_skip_row",
                            extra={
                                "event": "WARNING",
                                "row": row_idx,
                                "error": str(e),
                                "action_taken": "skip_row",
                            },
                        )
                        continue
            
            logger.info(
                "dataset_loaded",
                extra={
                    "event": "DATASET_LOADED",
                    "file": str(path),
                    "n_samples": len(samples),
                    "n_anomalies": sum(1 for s in samples if s.is_anomaly),
                },
            )
            
            return samples
        
        except Exception as e:
            logger.error(
                "dataset_load_failed",
                extra={
                    "event": "LOAD_ERROR",
                    "file": str(path),
                    "error": str(e),
                },
            )
            raise ValueError(f"Failed to load dataset: {e}") from e
    
    @staticmethod
    def load_nab(file_path: str) -> List[DatasetSample]:
        """Load NAB format dataset.
        
        NAB format: timestamp,value,label
        
        Args:
            file_path: Path to NAB CSV file.
        
        Returns:
            List of DatasetSample instances.
        """
        return DatasetLoader.load_csv(
            file_path=file_path,
            timestamp_col="timestamp",
            value_col="value",
            label_col="label",
        )
    
    @staticmethod
    def load_yahoo(file_path: str) -> List[DatasetSample]:
        """Load Yahoo format dataset.
        
        Yahoo format: timestamp,value,is_anomaly
        
        Args:
            file_path: Path to Yahoo CSV file.
        
        Returns:
            List of DatasetSample instances.
        """
        return DatasetLoader.load_csv(
            file_path=file_path,
            timestamp_col="timestamp",
            value_col="value",
            label_col="is_anomaly",
        )
