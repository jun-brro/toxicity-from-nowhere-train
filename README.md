# VLM Intrinsic Toxicity Pipeline

This repository implements the extraction, sparse autoencoder training, and evaluation pipeline described in `AGENTS.md`. The code supports the LLaVA-Guard v1.2 model and Meme-Safety-Bench dataset by default, and is organized as a reusable Python package `vlm_intrinsic_tox` with three CLI entrypoints:

- `python -m vlm_intrinsic_tox.cli.extract --config configs/extract.yaml`
- `python -m vlm_intrinsic_tox.cli.train_sae --config configs/sae_train.yaml --layer 0 --shards artifacts/extract/layer_00/*.npz --output artifacts/sae/layer_00`
- `python -m vlm_intrinsic_tox.cli.eval --config configs/eval.yaml --extract-dir artifacts/extract --sae-dir artifacts/sae`

The CLIs can be composed to run the full end-to-end workflow:

1. **Extract δ_int residuals** for chosen decoder layers.
2. **Train overcomplete Top-K sparse autoencoders** on the pooled residuals.
3. **Evaluate latent toxicity alignment** and generate JSON/Markdown reports with the highest scoring features per layer.

Refer to the configuration files in `configs/` for tunable parameters such as layers, pooling strategy, latent ratio, and metric weights. All artifacts include reproducibility metadata (Git commit, model name, dataset split, and seed).

## Documentation

- [Development Guide / 개발 가이드](doc/DEVELOPMENT.md)
- [Usage Guide / 사용 가이드](doc/USAGE.md)

Each guide is bilingual (English and Korean) and explains the repository layout, environment setup, pipeline stages, troubleshooting tips, and artifact expectations in depth.
