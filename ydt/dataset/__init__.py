"""
Dataset operations module

Provides dataset manipulation operations:
- Splitting (train/val)
- Merging multiple datasets
- Synthetic dataset generation
"""

from .split import split_dataset, merge_datasets
from .synthesize import DatasetSynthesizer

__all__ = [
    "split_dataset",
    "merge_datasets",
    "DatasetSynthesizer",
]
