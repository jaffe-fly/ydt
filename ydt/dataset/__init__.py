"""
Dataset operations module

Provides dataset manipulation operations:
- Splitting (train/val)
- Merging multiple datasets
- Synthetic dataset generation
"""

from .split import merge_datasets, split_dataset
from .synthesize import DatasetSynthesizer

__all__ = [
    "split_dataset",
    "merge_datasets",
    "DatasetSynthesizer",
]
