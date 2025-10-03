"""Collate functions for batching, padding, and masks."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image

from ..models.llava_guard import ModelInputs
from ..utils.logging import get_logger
from .memesafety import Sample

LOGGER = get_logger(__name__)


class DataCollator:
    """Collator for batching samples with proper padding and masks."""
    
    def __init__(self, processor, tokenizer, max_length: int = 512):
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples.
        
        Args:
            samples: List of Sample objects
            
        Returns:
            Dictionary with batched tensors
        """
        batch_size = len(samples)
        
        # Separate images and texts
        images = []
        texts = []
        labels = []
        sample_ids = []
        
        for sample in samples:
            # Load image if it's a path/reference
            if isinstance(sample.image, str):
                try:
                    image = Image.open(sample.image).convert("RGB")
                except Exception as e:
                    LOGGER.warning("Failed to load image %s: %s", sample.image, e)
                    # Create a dummy image
                    image = Image.new("RGB", (224, 224), color="white")
            elif isinstance(sample.image, dict) and "path" in sample.image:
                try:
                    image = Image.open(sample.image["path"]).convert("RGB")
                except Exception as e:
                    LOGGER.warning("Failed to load image %s: %s", sample.image["path"], e)
                    image = Image.new("RGB", (224, 224), color="white")
            else:
                image = sample.image
            
            images.append(image)
            texts.append(f"<image>\n{sample.instruction}")
            labels.append(sample.label if sample.label is not None else -1)
            sample_ids.append(sample.id)
        
        # Process batch
        try:
            batch_inputs = self.processor(
                images=images,
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length
            )
        except Exception as e:
            LOGGER.error("Failed to process batch: %s", e)
            # Fallback: process individually and pad manually
            batch_inputs = self._process_individually(images, texts)
        
        # Add labels and metadata
        result = {
            "input_ids": batch_inputs["input_ids"],
            "attention_mask": batch_inputs["attention_mask"],
            "pixel_values": batch_inputs["pixel_values"],
            "labels": torch.tensor(labels, dtype=torch.long),
            "sample_ids": sample_ids,
        }
        
        if "image_sizes" in batch_inputs:
            result["image_sizes"] = batch_inputs["image_sizes"]
        
        return result
    
    def _process_individually(self, images: List[Image.Image], texts: List[str]) -> Dict[str, torch.Tensor]:
        """Fallback processing when batch processing fails."""
        all_input_ids = []
        all_attention_masks = []
        all_pixel_values = []
        
        for image, text in zip(images, texts):
            try:
                inputs = self.processor(
                    images=image,
                    text=text,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True
                )
                all_input_ids.append(inputs["input_ids"])
                all_attention_masks.append(inputs["attention_mask"])
                all_pixel_values.append(inputs["pixel_values"])
            except Exception as e:
                LOGGER.warning("Failed to process individual sample: %s", e)
                # Create dummy tensors
                dummy_ids = torch.zeros((1, 10), dtype=torch.long)
                dummy_mask = torch.ones((1, 10), dtype=torch.long)
                dummy_pixels = torch.zeros((1, 3, 224, 224))
                all_input_ids.append(dummy_ids)
                all_attention_masks.append(dummy_mask)
                all_pixel_values.append(dummy_pixels)
        
        # Pad sequences to same length
        max_len = max(ids.shape[1] for ids in all_input_ids)
        
        padded_ids = []
        padded_masks = []
        
        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - ids.shape[1]
            if pad_len > 0:
                pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0
                padded_ids.append(torch.cat([ids, torch.full((1, pad_len), pad_token_id, dtype=torch.long)], dim=1))
                padded_masks.append(torch.cat([mask, torch.zeros((1, pad_len), dtype=torch.long)], dim=1))
            else:
                padded_ids.append(ids)
                padded_masks.append(mask)
        
        return {
            "input_ids": torch.cat(padded_ids, dim=0),
            "attention_mask": torch.cat(padded_masks, dim=0),
            "pixel_values": torch.cat(all_pixel_values, dim=0),
        }


def create_data_loader(samples: List[Sample], processor, tokenizer, batch_size: int = 8, max_length: int = 512):
    """Create a DataLoader with proper collation.
    
    Args:
        samples: List of Sample objects
        processor: HuggingFace processor
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        
    Returns:
        DataLoader with collated batches
    """
    from torch.utils.data import DataLoader, Dataset
    
    class SampleDataset(Dataset):
        def __init__(self, samples: List[Sample]):
            self.samples = samples
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
    
    dataset = SampleDataset(samples)
    collator = DataCollator(processor, tokenizer, max_length)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0  # Avoid multiprocessing issues with images
    )


__all__ = ["DataCollator", "create_data_loader"]