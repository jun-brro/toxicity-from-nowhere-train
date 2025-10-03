# VLM Intrinsic Toxicity Pipeline

This repository implements a comprehensive pipeline for detecting and mitigating intrinsic toxicity in Vision-Language Models (VLMs), as described in `AGENTS.md`. The code supports the LLaVA-Guard v1.2 model with multiple datasets and is organized as a reusable Python package `vlm_intrinsic_tox`.

## Features

- **δ_int Extraction**: Extract cross-modality interaction residuals from VLM decoder layers
- **Top-K SAE Training**: Train overcomplete sparse autoencoders with Top-K activation
- **Toxicity Evaluation**: Comprehensive metrics (dfreq, dmag, AUROC, KL, consensus) for latent analysis
- **Visualization Tools**: t-SNE plots, activation distributions, and feature analysis
- **Concept Steering**: GCAV-based intervention for real-time toxicity mitigation
- **Multi-Dataset Support**: Meme-Safety-Bench, SIUO, and extensible dataset registry

## CLI Tools

- `python -m vlm_intrinsic_tox.cli.extract --config configs/extract.yaml`
- `python -m vlm_intrinsic_tox.cli.train_sae --config configs/sae_train.yaml`
- `python -m vlm_intrinsic_tox.cli.eval --config configs/eval.yaml`
- `python -m vlm_intrinsic_tox.cli.visualize --config configs/eval.yaml --layer 0`

## Workflow

1. **Extract δ_int residuals** for chosen decoder layers using ON/OFF cross-attention
2. **Train overcomplete Top-K sparse autoencoders** on the pooled residuals
3. **Evaluate latent toxicity alignment** with comprehensive metrics
4. **Visualize and analyze** toxic latent features
5. **Apply interventions** using latent gating or concept steering

All configurations are in `configs/` with support for Hydra-style overrides. Artifacts include full reproducibility metadata.

## Documentation

- [Development Guide / 개발 가이드](doc/DEVELOPMENT.md)
- [Usage Guide / 사용 가이드](doc/USAGE.md)

Each guide is bilingual (English and Korean) and explains the repository layout, environment setup, pipeline stages, troubleshooting tips, and artifact expectations in depth.
