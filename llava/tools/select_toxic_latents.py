import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools.extract_deltas import compute_deltas_for_sample
from tools.analyze_sae_features import load_sae_for_layer  # returns (sae, mean, std)

from datasets import load_dataset
import random

def load_manifest(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        lines = p.read_text(encoding="utf-8").splitlines()
        data = [json.loads(l) for l in lines if l.strip()]
    else:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
    return data


def load_meme_safety_bench(dataset_path: str, split: str = "test", max_samples: int = 0, balanced: bool = False):
    
    ds = load_dataset("oneonlee/Meme-Safety-Bench", "full")
    ds = ds[split]
    
    positive_records = []  # sentiment == "positive" (benign, label=0)
    negative_records = []  # sentiment == "negative" (toxic, label=1)
    sentiment_values = set()
    
    for ex in ds:
        img = ex["meme_image"]
        txt = ex["instruction"]
        sentiment = str(ex["sentiment"]).strip().lower()
        sentiment_values.add(sentiment)
        
        # sentiment mapping
        lbl = 1 if sentiment == "negative" else 0
        record = {"image": img, "text": txt, "label": lbl}
        
        if lbl == 1:  # negative (toxic)
            negative_records.append(record)
        else:  # positive (benign)
            positive_records.append(record)
    
    print(f"Sentiment values found: {sentiment_values}")
    print(f"Raw data: {len(negative_records)} negative (toxic), {len(positive_records)} positive (benign)")
    
    if balanced and max_samples > 0:
        per_class = max_samples // 2
        
        selected_negative = random.sample(negative_records, min(per_class, len(negative_records)))
        selected_positive = random.sample(positive_records, min(per_class, len(positive_records)))
        
        records = selected_negative + selected_positive
        random.shuffle(records)
        
        print(f"Balanced sampling: {len(selected_negative)} negative + {len(selected_positive)} positive = {len(records)} total")
    else:
        records = negative_records + positive_records
        if max_samples and len(records) > max_samples:
            records = records[:max_samples]
    
    pos_count = sum(1 for r in records if r["label"] == 1)
    neg_count = sum(1 for r in records if r["label"] == 0)
    
    print(f"Final dataset: {pos_count} negative (toxic), {neg_count} positive (benign)")
    if pos_count == 0 or neg_count == 0:
        print(f"Only one class present! pos={pos_count}, neg={neg_count}")
    
    return records


def resolve_image(img, root):
    img = str(img)
    if img.startswith("/") or img.startswith("http"):
        return img
    return str(Path(root) / img)


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    std = x.std()
    if std < 1e-8:
        return np.zeros_like(x)
    return (x - x.mean()) / std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_type", choices=["manifest", "meme_safety_bench"], default="manifest")
    ap.add_argument("--manifest", help="JSON/JSONL with fields: image, text, label (0/1) [for dataset_type=manifest]")
    ap.add_argument("--images_root", help="Root dir to resolve relative image paths [for dataset_type=manifest]")
    ap.add_argument("--dataset_path", help="Local HF dataset path or HF hub (used if local not found) for Meme-Safety-Bench")
    ap.add_argument("--split", default="train")

    ap.add_argument("--sae_dir", default="/home/pljh0906/tcav/llava/artifacts/sae")
    ap.add_argument("--layers", default="20,23,25")
    ap.add_argument("--model", default="AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--output", default="toxic_latents_meme_safety_bench.json")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode (limits to 5 samples and verbose output)")
    args = ap.parse_args()
    
    if args.debug:
        args.max_samples = 10
        print("DEBUG MODE ENABLED - Processing 5 positive + 5 negative samples with verbose output")

    target_layers = [int(x.strip()) for x in args.layers.split(",")]

    # Concise run configuration
    print(f"Config: model={args.model}, device={args.device}, layers={target_layers}, top_k={args.top_k}")
    data_path = args.dataset_path if args.dataset_type == "meme_safety_bench" else (args.manifest or "")
    print(f"Data: type={args.dataset_type}, split={args.split}, max_samples={args.max_samples}, path={data_path}")

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=args.device, trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    # Load SAE per layer
    layer_cfg = {}
    for L in target_layers:
        sae, mean, std = load_sae_for_layer(Path(args.sae_dir) / f"layer_{L}", args.device)
        layer_cfg[L] = {"sae": sae.eval(), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}
        try:
            latent_dim = getattr(sae, "latent_dim", None)
            input_dim = len(mean)
            print(f"Loaded SAE layer {L}: input_dim={input_dim}, latent_dim={latent_dim}")
        except Exception:
            pass

    # Load dataset
    if args.dataset_type == "meme_safety_bench":
        records = load_meme_safety_bench(
            args.dataset_path or "", 
            args.split, 
            args.max_samples, 
            balanced=args.debug
        )
        images_root = None
    else:
        records = load_manifest(args.manifest)
        if args.max_samples and len(records) > args.max_samples:
            records = records[:args.max_samples]
        images_root = args.images_root

    print(f"Processing {len(records)} samples across {len(target_layers)} layers...")

    # Accumulate z per layer
    Z = {L: [] for L in target_layers}
    y = []

    for i, r in enumerate(records):
        if args.debug or i % 50 == 0:
            print(f"Progress: {i+1}/{len(records)} samples processed")
        
        # r["image"] can be PIL.Image or str path; handle both
        if isinstance(r["image"], str):
            img = resolve_image(r["image"], images_root) if images_root else r["image"]
        else:
            img = r["image"]
        txt = r["text"]
        lbl = int(r["label"])
        
        if args.debug:
            print(f"  Sample {i+1}: label={lbl}, text={txt[:50]}...")
            if isinstance(img, str):
                print(f"  Image: {img}")

        deltas = compute_deltas_for_sample(model, processor, img, txt, target_layers, args.device)
        any_ok = False
        for L in target_layers:
            if L in deltas:
                delta = deltas[L].astype(np.float32)
                mean = layer_cfg[L]["mean"]
                std = layer_cfg[L]["std"]
                x = (delta - mean) / (std + 1e-6)
                t = torch.from_numpy(x).unsqueeze(0).to(args.device)
                with torch.no_grad():
                    _, z = layer_cfg[L]["sae"](t)
                Z[L].append(z.squeeze(0).detach().cpu().numpy())
                any_ok = True
                if args.debug:
                    print(f"    Layer {L}: delta shape={delta.shape}, z shape={z.shape}")
        if any_ok:
            y.append(lbl)
            if args.debug:
                print(f"    Added to dataset with label {lbl}")

    print(f"Completed processing {len(y)} valid samples")
    
    y = np.array(y, dtype=np.int64)
    
    unique_labels, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique_labels, counts))
    print(f"Final class distribution: {class_dist}")
    
    if args.debug:
        print("DEBUG: Class distribution analysis")
        for label, count in class_dist.items():
            print(f"  Class {label}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Check if we have both classes for ROC AUC calculation
    has_both_classes = len(unique_labels) > 1
    if not has_both_classes:
        print("WARNING: Only one class present in final dataset. ROC AUC will be set to 0.5.")
    
    print("Computing statistics and rankings...")
    results = {"layers": target_layers, "counts": int(len(y)), "top_k": args.top_k, "per_layer": {}}

    for L in target_layers:
        if len(Z[L]) == 0:
            if args.debug:
                print(f"  Layer {L}: No data, skipping")
            continue
        ZL = np.vstack(Z[L])  # [N, latent_dim]
        pos = y == 1
        neg = y == 0
        
        if args.debug:
            print(f"  Layer {L}: Processing {ZL.shape[0]} samples, {ZL.shape[1]} latents")
            print(f"    Positive samples: {pos.sum()}, Negative samples: {neg.sum()}")

        freq_pos = (ZL[pos] > 0).mean(axis=0) if pos.sum() > 0 else np.zeros(ZL.shape[1])
        freq_neg = (ZL[neg] > 0).mean(axis=0) if neg.sum() > 0 else np.zeros(ZL.shape[1])
        dfreq = freq_pos - freq_neg

        mag_pos = ZL[pos].mean(axis=0) if pos.sum() > 0 else np.zeros(ZL.shape[1])
        mag_neg = ZL[neg].mean(axis=0) if neg.sum() > 0 else np.zeros(ZL.shape[1])
        dmag = mag_pos - mag_neg

        auroc = np.zeros(ZL.shape[1], dtype=np.float32)
        if has_both_classes:
            for j in range(ZL.shape[1]):
                try:
                    auroc[j] = roc_auc_score(y, ZL[:, j])
                except ValueError:
                    auroc[j] = 0.5
        else:
            # Only one class present, set all AUROCs to 0.5 (random performance)
            auroc.fill(0.5)
            if args.debug:
                print(f"  Layer {L}: Set all AUROC values to 0.5 (single class dataset)")

        s_dfreq = zscore(dfreq)
        s_dmag = zscore(dmag)
        s_auc = zscore(auroc)
        consensus = (s_dfreq + s_dmag + s_auc) / 3.0

        # Per-layer concise summary
        auc06 = int((auroc > 0.6).sum())
        auc07 = int((auroc > 0.7).sum())
        print(f"Layer {L}: N={ZL.shape[0]} D={ZL.shape[1]} pos={int(pos.sum())} neg={int(neg.sum())} auc>0.6={auc06} auc>0.7={auc07}")

        order = np.argsort(-consensus)
        k = min(args.top_k, len(order))
        top_ids = order[:k]
        
        if args.debug:
            print(f"    Top {k} toxic latents for layer {L}:")
            for i, idx in enumerate(top_ids[:10]):
                print(f"      {i+1:2d}. Latent {idx:4d}: consensus={consensus[idx]:.3f}, "
                      f"dfreq={dfreq[idx]:.3f}, dmag={dmag[idx]:.3f}, auroc={auroc[idx]:.3f}")
        else:
            # Non-debug concise top-3
            head = [int(i) for i in top_ids[:3]]
            head_scores = [float(consensus[i]) for i in top_ids[:3]]
            print(f"Layer {L}: top3_indices={head}, top3_consensus={[round(s,3) for s in head_scores]}")

        results["per_layer"][str(L)] = {
            "latent_dim": int(ZL.shape[1]),
            "dfreq": dfreq.tolist(),
            "dmag": dmag.tolist(),
            "auroc": auroc.tolist(),
            "consensus": consensus.tolist(),
            "top_indices": [int(i) for i in top_ids],
            "top_scores": [float(consensus[i]) for i in top_ids],
        }

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    try:
        size = Path(args.output).stat().st_size
        print(f"Saved: {args.output} ({size} bytes)")
    except Exception:
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()