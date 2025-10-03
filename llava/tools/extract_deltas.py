import argparse
import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration


def load_siuo(data_dir: str, data_type: str = "gen", limit: int = None):
    data_dir = Path(data_dir)
    meta = data_dir / f"siuo_{data_type}.json"
    with open(meta, "r") as f:
        items = json.load(f)
    if limit:
        items = items[:limit]
    out = []
    for it in items:
        image_path = data_dir / "images" / it["image"]
        out.append(
            {
                "text": it["question"],
                "image": str(image_path),
                "category": it.get("category", "unknown"),
            }
        )
    return out


def prepare_inputs(processor, image, text, device):
    conversation = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    batch = processor(text=text_prompt, images=image, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def prepare_text_only_inputs(processor, text, device):
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": text}]}
    ]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    batch = processor(text=text_prompt, images=None, return_tensors="pt")
    return {k: v.to(device) for k, v in batch.items()}


def register_layer_hooks(model, target_layers: List[int]):
    activations: Dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, output):
            # Support tuple outputs
            out = output[0] if isinstance(output, (tuple, list)) else output
            # Mean over sequence to get [batch, hidden]
            if out.dim() > 2:
                out = out.mean(dim=1)
            activations[layer_idx] = out.detach().cpu()
        return hook

    hooks = []
    # Language model layers
    if hasattr(model, "language_model"):
        layers = None
        if hasattr(model.language_model, "layers"):
            layers = model.language_model.layers
        elif hasattr(model.language_model, "model") and hasattr(model.language_model.model, "layers"):
            layers = model.language_model.model.layers
        if layers is None:
            raise RuntimeError("Could not locate language layers on model")
        for idx in target_layers:
            if idx < len(layers):
                hooks.append(layers[idx].register_forward_hook(make_hook(idx)))
    else:
        raise RuntimeError("Model missing language_model component")

    return activations, hooks


def compute_deltas_for_sample(model, processor, image_or_path, text: str, target_layers: List[int], device: str):
    # Load image from PIL.Image, local path, or URL
    if isinstance(image_or_path, Image.Image):
        image = image_or_path
    elif isinstance(image_or_path, str):
        if image_or_path.startswith("http"):
            import requests
            from io import BytesIO
            resp = requests.get(image_or_path, timeout=10)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content))
        else:
            p = Path(image_or_path)
            image = Image.open(p) if p.exists() else Image.new("RGB", (224, 224), color="white")
    else:
        # Fallback
        try:
            p = Path(image_or_path)
            image = Image.open(p) if p.exists() else Image.new("RGB", (224, 224), color="white")
        except Exception:
            image = Image.new("RGB", (224, 224), color="white")

    # h_on: normal multimodal forward
    acts_on, hooks_on = register_layer_hooks(model, target_layers)
    inputs_on = prepare_inputs(processor, image, text, device)
    with torch.no_grad():
        _ = model(**inputs_on)
    for h in hooks_on:
        h.remove()

    # h_off: text-only forward (no cross-modal evidence)
    acts_off, hooks_off = register_layer_hooks(model, target_layers)
    inputs_off = prepare_text_only_inputs(processor, text, device)
    with torch.no_grad():
        _ = model(**inputs_off)
    for h in hooks_off:
        h.remove()

    deltas = {}
    for layer in target_layers:
        if layer in acts_on and layer in acts_off:
            on = acts_on[layer]
            off = acts_off[layer]
            # Shapes: [1, hidden]
            if on.shape != off.shape:
                continue
            deltas[layer] = (on - off).squeeze(0).numpy()
    return deltas


def main():
    parser = argparse.ArgumentParser(description="Extract h_on - h_off deltas for LlavaGuard (SIUO)")
    parser.add_argument("--model", default="AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--siuo_data_path", default="/scratch2/pljh0906/tcav/datasets/SIUO/data")
    parser.add_argument("--data_type", default="gen", choices=["gen", "mcqa"]) 
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--layers", default="20,23,25")
    parser.add_argument("--output", default="artifacts/deltas/siuo_gen")

    args = parser.parse_args()
    target_layers = [int(x.strip()) for x in args.layers.split(",")]

    print(f"Loading model {args.model} on {args.device}...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    items = load_siuo(args.siuo_data_path, args.data_type, args.max_samples)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_buffers: Dict[int, List[np.ndarray]] = {l: [] for l in target_layers}
    categories: List[str] = []

    for it in tqdm(items, desc="Extracting deltas"):
        deltas = compute_deltas_for_sample(model, processor, it["image"], it["text"], target_layers, args.device)
        if not deltas:
            continue
        for l in target_layers:
            if l in deltas:
                layer_buffers[l].append(deltas[l])
        categories.append(it.get("category", "unknown"))

    # Save per-layer matrices and categories
    for l in target_layers:
        arr = np.stack(layer_buffers[l]) if layer_buffers[l] else np.empty((0,))
        np.save(output_dir / f"layer_{l}_delta.npy", arr)
        print(f"Saved layer {l} deltas: {arr.shape}")

    with open(output_dir / "categories.json", "w") as f:
        json.dump(categories, f)
    meta = {"layers": target_layers, "count": len(categories), "data_type": args.data_type, "model": args.model}
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Done. Saved to {output_dir}")


if __name__ == "__main__":
    main()


