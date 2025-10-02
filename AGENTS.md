# VLM Intrinsic Toxicity – Repo & Implementation Spec (LLaVA‑Guard + SAE)

> Single‑repo blueprint to extract δ_int activations from LLaVA‑Guard, train an overcomplete Top‑K SAE, and evaluate latent‑level toxic features & intervention effects. Written so any engineer/LLM can implement end‑to‑end without prior context.

---

## 0) Scope & Outcomes

**Goal**: Detect and mitigate *intrinsic toxicity* in a Vision‑Language Model (VLM) by:

1. Extracting **cross‑modality interaction residuals** δ_int at chosen LM decoder layers.
2. Training an **overcomplete Sparse Autoencoder (Top‑K SAE)** on δ_int to disentangle latent features.
3. **Evaluating** toxic‑aligned latents (dfreq/dmag/AUROC/KL/consensus) and **intervening** via latent gating; measuring safety metrics while tracking utility.

**Model (fixed in v1)**: `AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf`  (Vision: SigLIP, LM: Qwen2‑base‑sized decoder, projector in between).

**Datasets (v1)**: `oneonlee/Meme-Safety-Bench` (HF). Additional sets can be plugged in later.

**Decisions fixed**

* Cross‑attention **OFF method = A**: return zero‑tensor from the LM layer’s *self_attn* forward (i.e., vision‑text interaction suppressed at that site) while keeping shapes/dtypes.
* SAE **type = Top‑K** (overcomplete). Latent dim ratio is configurable.
* **All LM decoder layers selectable**; default list given via config/CLI.
* **Pooling** defaults to `text_only_mean` (see §3.4) but configurable.
* Evaluation focuses on **safety benchmarks only** in v1; utility checks are optional hooks.
* Label schema details for intrinsic/extrinsic/benign are **TBD** (placeholder interfaces provided).

**Non‑goals (v1)**: Training/finetuning the base VLM; producing production‑grade visual dashboards.

---

## 1) Theory & Definitions (Minimal)

### 1.1 Intrinsic toxicity residual

For image–text pair (I, T) and LM decoder layer (l):
[
\delta_{int}^{(l)}(I,T) ;=; h^{on}*{l}(I,T); -; h^{off}*{l}(I,T)
]
where:

* **on**: normal forward; cross‑modal interaction present at layer (l).
* **off**: same forward except the layer’s **self_attn output is replaced by zeros** before residual add (see §2.3). All other computations identical.
* We capture **post‑attention residual** (optionally after the layer’s norm if needed; default: *post self_attn add, pre‑MLP*).

### 1.2 Overcomplete Top‑K SAE

Given pooled vectors (x = P(\delta_{int})\in\mathbb{R}^{d}), standardize with global train‑set mean/std to (\tilde{x}), then learn encoder/decoder:

* Encoder: (z = \text{TopK}(W_e\tilde{x} + b_e, k))  (keep top‑k entries, zero others)
* Decoder: (\hat{x} = W_d z + b_d)
* Loss: (\mathcal{L} = |\hat{x}-\tilde{x}|_2^2) + (optional auxiliary terms like activation‑rate targets / L2 reg). Overcompleteness: (\text{latent_dim} > d).

### 1.3 Toxic‑alignment metrics (per latent k)

* **dfreq** = (P(\text{active}|harmful) - P(\text{active}|benign))
* **dmag** = (E[|z_k|,|harmful] - E[|z_k|,|benign])
* **AUROC** comparing (z_k) (or (|z_k|)) for harmful vs benign
* **KL** divergence between harmful vs benign latent distributions (Gaussian or KDE approx)
* **Consensus** = standardized weighted sum of {dfreq, dmag, AUROC, KL}. Weights/signedness configurable.

### 1.4 Intervention (steering)

* **Gating (default)**: For top‑K toxic latents (\mathcal{T}), set (z_k=0) for (k\in\mathcal{T}) before decoding.
* **Subtractive (optional)**: (\hat{h} = h - \sum_{k\in\mathcal{T}} z_k W_{d,k}). (Not required in v1.)

---

## 2) Model Hooking – LLaVA‑Guard/Qwen2 Decoder

### 2.1 Relevant blocks (from provided printout)

```
model.language_model: Qwen2Model
  .layers: ModuleList[0..23] of Qwen2DecoderLayer
    .self_attn: Qwen2Attention
    .mlp: Qwen2MLP
    .input_layernorm: Qwen2RMSNorm
    .post_attention_layernorm: Qwen2RMSNorm
```

The projector maps SigLIP vision tokens into the LM embedding space (dim=896). The LM decoder attends over a sequence combining text tokens (incl. special <image> marker) and (projected) image tokens.

### 2.2 Capture point (default)

* **Tensor**: the *post self_attn residual* hidden state of a decoder layer, i.e., after adding attention output back to the residual stream (optionally after `post_attention_layernorm`, see config). We denote this as (h_l).
* We pool (see §3.4) across **text content tokens** only.

### 2.3 "Cross‑attn OFF" implementation (Method A)

* Patch **`Qwen2DecoderLayer[i].self_attn.forward`** with a context manager that returns a **zeros tensor of identical shape/dtype/device** instead of the computed attention output. Everything else (KV cache, masks, norms, MLP) remains unchanged.
* Keep RNG & caching consistent between ON/OFF passes (no dropout diffs).
* Run **two forwards** per batch/sample: ON then OFF (or vice‑versa) under `torch.no_grad()`.

### 2.4 Layer selection

* Accept arbitrary integer indices (\mathcal{L}\subseteq{0..23}) via CLI/config. Each selected layer produces its own (\delta_{int}^{(l)}) stream.

---

## 3) Data Pipeline

### 3.1 Datasets

* **`oneonlee/Meme-Safety-Bench`** (HF). Split used: `test` (≈50k rows). Columns include: `meme_image` (or `image_path`), `instruction`, `sentiment`, `category`, `task`.
* Future: SIUO, Hateful Memes, benign caption corpora (plug‑in interface provided).

### 3.2 Label schema (v1 placeholder)

* **Binary harmful**: preliminary mapping from `sentiment`/`category` to {harmful, benign}. *Exact mapping is TBD*; the labeler interface expects `label` ∈ {0,1} for harmfulness. A dataset adapter can leave label as `None` (unknown), in which case those samples are ignored for metric computations but still usable for SAE training (unsupervised).

### 3.3 Text/Image preparation

* Text: use dataset `instruction` as the textual input; prepend required system/prompt template for LLaVA‑Guard if any (configurable). Ensure a single `<image>` marker occurs before the text (model‑specific prompt template).
* Image: load and resize per SigLIP preprocessor settings associated with the checkpoint.

### 3.4 Token pooling strategies

Let token indices after tokenization define segments:

* `prefix_span`: system/instruction prompt boilerplate
* `image_marker_idx`: index of `<image>` token (single)
* `content_span`: tokens strictly **after** `<image>` and **not** special EOS/PAD

**Pooling options** (config `extract.pooling`):

1. `text_only_mean` (default): mean over `content_span` text token hidden states.
2. `eos`: take hidden state at EOS.
3. `last_k_mean`: mean of last `k` content tokens (config `k`).
4. (optional) `span_mean`: mean over an explicit `[start,end)` span supplied by the dataset adapter.

### 3.5 Sharding & artifact format

* For each layer (l), write shards of pooled (\delta_{int}^{(l)}) as **`npz` or `parquet`** with columns/keys:

  * `delta`: float16/float32 array `[N_shard, d]`
  * `meta`: `sample_id`, `dataset_name`, `layer`, `pooling`, `prompt_hash`, `image_hash`, `hf_commit`, `model_commit`, `dtype`, `token_span` info
* Global stats pass: compute **mean/std** per layer over the **training subset** only; save `scaler_{layer}.json`.

---

## 4) SAE Training

### 4.1 Input & standardization

* Load δ_int shards for specified layers and split (train). Apply global **(mean, std)** standardization per layer computed on the train split. Persist scaler.

### 4.2 Model (Top‑K SAE)

* Encoder weight (W_e\in\mathbb{R}^{m\times d}), bias (b_e\in\mathbb{R}^{m}), latent size (m = r\cdot d), ratio `r` configurable (e.g., 2–8).
* **Top‑K activation**: per sample, keep largest‑magnitude `k` elements of pre‑activation; set others to zero. `k` controls **target density** (~k/m).
* Decoder (W_d\in\mathbb{R}^{d\times m}), bias (b_d\in\mathbb{R}^{d}).

### 4.3 Objective & schedule

* Loss: MSE reconstruction (|\hat{x}-\tilde{x}|_2^2). Optional L2 on weights; optional penalty to stabilize activation rate.
* Optimizer: AdamW (default). LR schedule: cosine with warmup; early stop on val recon + activation‑rate stability.
* Batch size: large (e.g., 1–4k) since operations are linear.

### 4.4 Outputs

* Save per‑layer SAE: `sae_layer{l}.pt` (weights/bias/state), `train_stats.json` (recon error, density, k histogram).

---

## 5) Evaluation & Analysis

### 5.1 Latent extraction for eval

* For each eval dataset split:

  1. Extract δ_int as in §2–§3 using the **same pooling** and **same layers**.
  2. Standardize using the **train scaler** of that layer.
  3. Encode with SAE to get latent vectors `z`.

### 5.2 Metrics (per latent)

Compute for harmful vs benign partitions:

* **dfreq**: active threshold is `z_i ≠ 0` (Top‑K) or `|z_i|>τ` (if needed).
* **dmag**: mean abs activation difference.
* **AUROC** using `|z_i|` as a score.
* **KL**: e.g., two‑Gaussian approximation with small ε for stability.
* **Consensus**: z‑scored metrics aggregated (weights in config); higher is more “toxic‑aligned”.
* Produce layer‑wise tables: top‑N latents by consensus, and full distribution summaries (mean/σ, percentiles, correlations).

### 5.3 Intervention (gating)

* Select **top‑K latents** by consensus per layer (configurable).
* During a *fresh forward*, apply an **online hook** that zeroes those latent components before SAE decoding back to residual space, or equivalently projects out their decoder columns: (h' = h - W_d[:,\mathcal{T}],z_{\mathcal{T}}). For v1, gating at the latent is sufficient since we only report latent‑space metrics; optional back‑projection to the residual stream is provided for end‑to‑end safety metric checks.

### 5.4 Reports

* Structured JSON + Markdown:

  * Dataset summary, layer list, pooling.
  * Metric histograms/percentiles per layer.
  * Top‑10 latents table per layer (dfreq, dmag, AUROC, KL, consensus).
  * Cross‑layer comparison summary.
  * (Optional) Safety metric deltas pre/post intervention.

---

## 6) Repository Layout

```
vlm_intrinsic_tox/
├─ configs/
│  ├─ extract.yaml
│  ├─ sae_train.yaml
│  └─ eval.yaml
├─ src/vlm_intrinsic_tox/
│  ├─ models/
│  │  ├─ llava_guard.py          # HF load, preprocessors, prompt templates
│  │  └─ hooks.py                # layer capture + cross-attn OFF context manager
│  ├─ data/
│  │  ├─ memesafety.py           # HF adapter (image, text, label)
│  │  ├─ registry.py             # dataset registry & factory
│  │  └─ collate.py              # batch, padding, masks
│  ├─ extract/
│  │  ├─ runner.py               # δ_int extraction pipeline (ON/OFF->pool->save)
│  │  └─ pooling.py              # pooling functions
│  ├─ sae/
│  │  ├─ modules.py              # Top-K SAE implementation
│  │  ├─ train.py                # training loop, scaler, logging
│  │  └─ io.py                   # shard IO (npz/parquet), scaler (json)
│  ├─ eval/
│  │  ├─ metrics.py              # dfreq, dmag, AUROC, KL, consensus
│  │  ├─ analyze.py              # tables, cross-layer summaries
│  │  ├─ intervene.py            # latent gating hooks (optional residual proj)
│  │  └─ report.py               # JSON/Markdown writers
│  ├─ utils/
│  │  ├─ config.py               # YAML/OMEGA loader, validation
│  │  ├─ logging.py              # logger + TB/W&B (optional)
│  │  ├─ seed.py                 # reproducibility
│  │  └─ paths.py                # artifact paths
│  └─ cli/
│     ├─ extract.py              # CLI entry for §2–§3
│     ├─ train_sae.py            # CLI entry for §4
│     └─ eval.py                 # CLI entry for §5
├─ artifacts/                    # generated (ignored by git)
│  ├─ activations/
│  ├─ sae/
│  └─ reports/
└─ README.md
```

---

## 7) Configuration Schemas (authoritative)

### 7.1 `configs/extract.yaml`

```yaml
model:
  hf_id: AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf
  dtype: bfloat16
  device_map: auto
  prompt_template: llavaguard_default   # name resolved by llava_guard.py

extract:
  layers: [0, 2, 4]                     # any subset of 0..23
  capture_point: post_attn_resid        # [post_attn_resid|post_attn_ln|post_mlp]
  cross_attn_off_impl: zeros            # fixed to Method A
  pooling:
    name: text_only_mean                # [text_only_mean|eos|last_k_mean|span_mean]
    last_k: 5                           # used if name==last_k_mean
  batch_size: 8
  num_workers: 4
  amp: true
  save:
    out_dir: artifacts/activations/llavaguard
    shard_size: 4000

# Dataset registry names and split
data:
  datasets:
    - memesafety_test
  split: test
  root: null                            # let HF handle caching unless provided

repro:
  seed: 42
```

### 7.2 `configs/sae_train.yaml`

```yaml
sae:
  type: topk                           # fixed in v1
  latent_ratio: 4.0                    # m = ratio * d
  topk: 64                             # absolute K; alternatively use density
  weight_decay: 0.0
  lr: 3e-4
  epochs: 12
  batch_size: 2048
  val_fraction: 0.05

normalization:
  scheme: global_mean_std              # computed on train shards per layer

layers: [0, 2, 4]

io:
  in_dir: artifacts/activations/llavaguard
  out_dir: artifacts/sae/llavaguard
  scaler_filename: scaler_layer{l}.json
  sae_filename: sae_layer{l}.pt
```

### 7.3 `configs/eval.yaml`

```yaml
eval:
  datasets: [memesafety_test]
  layers: [0, 2, 4]
  metrics: [dfreq, dmag, auroc, kl, consensus]
  consensus:
    weights: {dfreq: 1.0, dmag: 1.0, auroc: 1.0, kl: 0.5}
  threshold:
    active: auto                       # auto means Top-K implies binary active
  topk_latents: 50                     # for reporting & intervention

intervention:
  enabled: true
  method: gate                         # [gate|subtract]

io:
  activ_dir: artifacts/activations/llavaguard
  sae_dir: artifacts/sae/llavaguard
  report_dir: artifacts/reports/llavaguard
```

---

## 8) Module Contracts (function‑level specs)

### 8.1 `models/llava_guard.py`

**Responsibilities**: load HF model/tokenizer/processor; build prompt; expose forward utilities.

**Key APIs**

* `load_model(cfg) -> (model, tokenizer, image_processor)`
* `build_inputs(instruction:str, image:Image) -> ModelInputs`
* `get_text_token_spans(token_ids) -> dict(spans)` → returns `prefix_span`, `image_marker_idx`, `content_span` indices.

### 8.2 `models/hooks.py`

**Responsibilities**: capturing layer tensors; cross‑attn OFF context manager.

**Key APIs**

* `capture_post_attn_resid(layer:Qwen2DecoderLayer, cache:dict)`
* `cross_attn_off(layer:Qwen2DecoderLayer) -> contextmanager`
  returns a context that patches `layer.self_attn.forward` to output zeros of appropriate shape.
* `extract_delta_int(model, inputs, layers, capture_point, pooling) -> Dict[layer, np.ndarray]`
  runs ON and OFF passes, pools per §3.4, computes δ_int, returns pooled arrays.

### 8.3 `data/registry.py`

* `get_dataset(name:str, split:str, root:Optional[str]) -> Iterable[Sample]`

### 8.4 `data/memesafety.py`

* Adapter that yields `Sample{id, image, instruction, label(Optional[int]), meta}`.
* **Label mapping** placeholder: if `sentiment` exists, map {`negative`→1, else 0}; else set `label=None` (TBD mapping can replace here later).

### 8.5 `extract/runner.py`

* Stream samples → tokenize & preprocess → `extract_delta_int` → write shards.
* Maintains metadata and shard integrity checks.

### 8.6 `extract/pooling.py`

* `pool_text_only_mean(hiddens, content_span)`
* `pool_eos(hiddens, eos_idx)`
* `pool_last_k_mean(hiddens, content_span, k)`
* `pool_span_mean(hiddens, span)`

### 8.7 `sae/modules.py`

* `class TopKSAE(d:int, m:int, k:int)` implementing encoder/decoder and forward returning `(z, x_hat)`; also `encode(x)->z`, `decode(z)->x_hat`.

### 8.8 `sae/train.py`

* `train_sae(layer:int, shards:Iterable[np.ndarray], cfg) -> TrainResult`
  Handles standardization (fit scaler), val split, training loop, checkpointing, logs.

### 8.9 `sae/io.py`

* Load/save shards (npz/parquet), scalers (json), models (pt). Verify shapes/dtypes.

### 8.10 `eval/metrics.py`

* `dfreq(z, labels)->np.ndarray[m]`
* `dmag(z, labels)->np.ndarray[m]`
* `auroc(z, labels)->np.ndarray[m]`
* `kl(z, labels)->np.ndarray[m]`
* `consensus(scores:Dict[str,np.ndarray], weights)->np.ndarray[m]`

### 8.11 `eval/analyze.py`

* Aggregate metric distributions, percentiles, correlations; rank latents; produce summaries.

### 8.12 `eval/intervene.py`

* `gate_latents(z, idxs)` → zero selected indices
* (optional) `project_out_residual(h, Wd, z, idxs)` if residual‑space steering is requested.

### 8.13 `eval/report.py`

* Write JSON and Markdown reports; include top‑N tables and cross‑layer summaries.

### 8.14 `cli/*.py`

* `extract.py`: parses `configs/extract.yaml`, runs extraction & sharding.
* `train_sae.py`: per‑layer SAE training with scalers; writes artifacts.
* `eval.py`: loads artifacts, computes metrics, optionally runs intervention, writes reports.

---

## 9) Reproducibility & Quality Gates

* **Seeds**: torch, numpy, random; set deterministic flags where available.
* **Versioning**: record HF model commit, dataset revision, code git SHA in artifacts.
* **Shards**: store row counts, MD5 hashes; verify before training/eval.
* **Numerics**: store δ_int as float16 (space) but scalers and SAE weights in float32.
* **CI smoke tests**:

  1. Tiny subset (N=32) can complete extract→train→eval in minutes.
  2. Assert non‑NaN metrics; percentiles monotone; Top‑K list non‑empty.

---

## 10) Risks & Mitigations

* **h_off leakage** (other paths re‑introduce vision): Keep OFF patch restricted to `self_attn.forward` and validate zero‑output via hooks. Add unit test to assert tensor norms==0.
* **Dataset label noise**: Allow `label=None`; metrics computed on available labels only; keep unsupervised SAE robust to mixed labels.
* **Over‑gating harms utility**: Plot safety vs K curves; set conservative default K.
* **Activation collapse (Top‑K too small/large)**: Monitor recon error & activation counts; expose config for `k` and ratio.

---

## 11) Minimal Run Books

### 11.1 Extract

```
python -m vlm_intrinsic_tox.cli.extract \
  --config configs/extract.yaml \
  extract.layers=[0,2,4] extract.pooling.name=text_only_mean
```

### 11.2 Train SAE

```
python -m vlm_intrinsic_tox.cli.train_sae \
  --config configs/sae_train.yaml \
  layers=[0,2,4] sae.latent_ratio=4.0 sae.topk=64
```

### 11.3 Eval & Report

```
python -m vlm_intrinsic_tox.cli.eval \
  --config configs/eval.yaml \
  eval.layers=[0,2,4] intervention.enabled=true eval.topk_latents=50
```

---

## 12) Open TODOs / Placeholders

* **Label mapping policy (TBD)** for Meme‑Safety‑Bench: define `harmful` vs `benign` logic from `sentiment/category/task`.
* **Prompt template** for LLaVA‑Guard (confirm exact `<image>` token and instruction format).
* **Utility metrics** (optional) if we later track caption/VQA preservation.
* **Add JumpReLU SAE** as an alternative in `sae/modules.py` (v2).

---

## 13) Acceptance Criteria (v1)

1. End‑to‑end run produces δ_int shards, per‑layer SAE, and evaluation reports with top‑K toxic latents and summary tables.
2. Metrics distributions are computed and persisted; cross‑layer comparison is in the report.
3. Intervention (gating) is executable via config flag and reflected in the report (even if only latent‑space metrics change in v1).
4. All artifacts include reproducibility metadata (commits, seeds, configs).

---

**This document is the single source of truth for implementation.** If a choice is missing, prefer defaults stated here and expose the knob via config for ablation.
