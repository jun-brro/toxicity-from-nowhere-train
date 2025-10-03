"""Dataset registry and factory functions."""

from __future__ import annotations

from typing import Iterable, Optional

from ..utils.logging import get_logger
from .memesafety import MemeSafetyBenchAdapter, Sample, Labeler, load_label_mapping

LOGGER = get_logger(__name__)


def get_dataset(name: str, split: str, root: Optional[str] = None) -> Iterable[Sample]:
    """Get dataset iterator by name.
    
    Args:
        name: Dataset name (e.g., 'memesafety_test')
        split: Dataset split (e.g., 'test', 'train')
        root: Optional root directory (unused for HF datasets)
        
    Returns:
        Iterable of Sample objects
        
    Raises:
        ValueError: If dataset name is not recognized
    """
    if name == "memesafety_test" or name.startswith("oneonlee/Meme-Safety-Bench"):
        # Handle both short name and full HF name
        hf_name = "oneonlee/Meme-Safety-Bench" if name == "memesafety_test" else name
        
        # Create adapter
        adapter = MemeSafetyBenchAdapter(
            name=hf_name,
            split=split,
            image_column="meme_image",
            instruction_column="instruction", 
            streaming=False
        )
        
        # Create labeler with default mapping
        label_mapping = load_label_mapping(None)  # Use default mapping
        labeler = Labeler(label_mapping, "sentiment")  # Use sentiment column for labeling
        
        # Default prompt template
        prompt_template = "<image>\n{instruction}"
        
        # Return samples with proper ID generation
        for idx, sample in enumerate(adapter.iter_samples(prompt_template, labeler)):
            # Convert to match specification: add id, change prompt to instruction
            yield Sample(
                id=f"{name}_{split}_{idx}",
                image=sample.image,
                instruction=sample.prompt.replace("<image>\n", ""),  # Remove image token from instruction
                label=sample.label,
                meta=sample.metadata
            )
    
    else:
        raise ValueError(f"Unknown dataset name: {name}")


# Registry of available datasets
DATASET_REGISTRY = {
    "memesafety_test": {
        "hf_name": "oneonlee/Meme-Safety-Bench",
        "splits": ["test", "train"],
        "description": "Meme Safety Benchmark dataset"
    }
}


def list_datasets() -> dict:
    """List all available datasets in the registry.
    
    Returns:
        Dictionary of dataset information
    """
    return DATASET_REGISTRY.copy()


__all__ = ["get_dataset", "list_datasets", "DATASET_REGISTRY"]