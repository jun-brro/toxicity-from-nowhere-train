# Usage Guide / 사용 가이드

> Step-by-step instructions to run the intrinsic toxicity pipeline end-to-end, manage artifacts, and interpret outputs.
> 내재적 독성 파이프라인을 처음부터 끝까지 실행하고 산출물을 관리하며 결과를 해석하는 방법을 단계별로 설명합니다.

---

## 1. Quickstart / 빠른 시작

### 1.1 Environment / 환경 준비
- EN: Install dependencies following the Development Guide, authenticate to Hugging Face, and ensure GPU availability.
- KO: 개발 가이드의 방법으로 의존성을 설치하고 허깅페이스 인증을 완료한 뒤 GPU 사용 가능 여부를 확인합니다.

### 1.2 Config Files / 설정 파일
- EN: Use the default YAML files in `configs/` as templates. Override parameters via CLI flags or by creating copies (e.g., `configs/extract_local.yaml`).
- KO: `configs/` 폴더의 기본 YAML 파일을 템플릿으로 사용하고, CLI 플래그 또는 복사본(`configs/extract_local.yaml` 등)을 만들어 파라미터를 재정의합니다.

---

## 2. Extraction Stage / 추출 단계

1. **Prepare Dataset Cache / 데이터셋 캐시 준비**
   - EN: The default adapter downloads `oneonlee/Meme-Safety-Bench`. Set the `HF_HOME` env var if you need a custom cache path.
   - KO: 기본 어댑터는 `oneonlee/Meme-Safety-Bench`를 다운로드합니다. 캐시 경로를 바꾸려면 `HF_HOME` 환경 변수를 설정하세요.
2. **Run CLI / CLI 실행**
   ```bash
   python -m vlm_intrinsic_tox.cli.extract \
     --config configs/extract.yaml \
     extract.layers="[0,2,4]" extract.pooling.name=text_only_mean \
     extract.limit_examples=128
   ```
   - EN: The command performs ON/OFF forwards, pools δ_int by text tokens, and writes sharded `.npz` artifacts under `artifacts/extract/<run_id>/`.
   - KO: 이 명령은 ON/OFF 순전파를 실행하고 텍스트 토큰 기준으로 δ_int를 풀링한 후 `artifacts/extract/<run_id>/` 아래에 `.npz` 샤드를 저장합니다.
3. **Key Outputs / 주요 산출물**
   - EN: `metadata.json` (config, seeds, git SHA), `layer_<idx>_shard_*.npz`, optional logs.
   - KO: `metadata.json`(설정, 시드, git SHA), `layer_<idx>_shard_*.npz`, 선택적 로그 파일.

---

## 3. SAE Training Stage / SAE 학습 단계

1. **Check Shards / 샤드 확인**
   - EN: Ensure each selected layer has matching shard counts. Use `python -m vlm_intrinsic_tox.sae.io list --root artifacts/extract/<run_id>` (coming soon) or inspect filenames.
   - KO: 선택한 각 레이어에 동일한 샤드 수가 있는지 확인합니다. `python -m vlm_intrinsic_tox.sae.io list --root artifacts/extract/<run_id>`(추가 예정) 또는 파일명을 직접 확인하세요.
2. **Run Training / 학습 실행**
   ```bash
   python -m vlm_intrinsic_tox.cli.train_sae \
     --config configs/sae_train.yaml \
     layers="[0,2,4]" sae.latent_ratio=4.0 sae.topk=64 \
     train.max_steps=5000
   ```
   - EN: Standardizes each layer’s pooled activations, trains `TopKSAE`, and writes checkpoints under `artifacts/sae/<run_id>/layer_<idx>/`.
   - KO: 각 레이어의 풀링된 활성값을 표준화한 뒤 `TopKSAE`를 학습하고 결과를 `artifacts/sae/<run_id>/layer_<idx>/`에 저장합니다.
3. **Artifacts / 산출물**
   - EN: `model.pt` (SAE weights), `scaler.json`, `train_log.jsonl`, reconstruction/activation metrics.
   - KO: `model.pt`(SAE 가중치), `scaler.json`, `train_log.jsonl`, 재구성/활성화 지표.

---

## 4. Evaluation & Intervention / 평가 및 개입

1. **Run Evaluation / 평가 실행**
   ```bash
   python -m vlm_intrinsic_tox.cli.eval \
     --config configs/eval.yaml \
     eval.layers="[0,2,4]" eval.topk_latents=50 \
     intervention.enabled=true
   ```
   - EN: Loads SAE artifacts, computes latent toxicity metrics, optionally zeroes top toxic latents, and saves reports under `artifacts/eval/<run_id>/`.
   - KO: SAE 산출물을 불러와 잠재 독성 지표를 계산하고, 필요 시 상위 독성 잠재벡터를 0으로 만든 뒤 결과를 `artifacts/eval/<run_id>/`에 저장합니다.
2. **Reports / 리포트**
   - EN: `summary.json`, `summary.md`, per-layer CSV tables, intervention diagnostics.
   - KO: `summary.json`, `summary.md`, 레이어별 CSV 테이블, 개입 진단 정보.
3. **Interpreting Metrics / 지표 해석**
   - EN: High `dfreq`/`dmag` indicates harmful preference; AUROC close to 1.0 means strong separation; consensus ranks combine scores.
   - KO: 높은 `dfreq`/`dmag`는 유해한 샘플에 대한 선호를 의미하고, AUROC가 1.0에 가까울수록 분리가 뚜렷함을 의미하며, 종합 순위는 여러 지표를 결합합니다.

---

## 5. Managing Runs / 실행 관리

- EN: Use unique `run_id` prefixes per experiment; artifacts embed metadata so you can trace configs and seeds. Store large `.npz` files on fast SSDs for training throughput.
- KO: 실험마다 고유한 `run_id` 접두사를 사용하고, 산출물에 포함된 메타데이터를 통해 설정과 시드를 추적합니다. 학습 속도를 위해 큰 `.npz` 파일은 빠른 SSD에 보관하세요.

---

## 6. Troubleshooting / 문제 해결

| Symptom (EN) | 증상 (KO) | Cause & Fix (EN) | 원인 및 해결 (KO) |
| --- | --- | --- | --- |
| Extraction tensors all zeros | 추출 텐서가 모두 0 | Hooks misconfigured: verify `cross_attn_off` flag and ensure ON/OFF forwards both run. | 훅 설정 오류: `cross_attn_off` 플래그와 ON/OFF 순전파 실행 여부를 확인합니다. |
| SAE loss diverges | SAE 손실 발산 | Learning rate too high; reduce `optimizer.lr` or increase `train.grad_clip`. | 학습률이 너무 높음: `optimizer.lr`을 낮추거나 `train.grad_clip`을 늘립니다. |
| Missing labels in eval | 평가 라벨 누락 | Dataset adapter may emit `None`; enable `eval.drop_missing_labels=true` or adjust mapping. | 데이터셋 어댑터가 `None`을 반환할 수 있음: `eval.drop_missing_labels=true`를 설정하거나 매핑을 수정합니다. |

---

## 7. Frequently Asked Questions / 자주 묻는 질문

1. **Can I use another VLM? / 다른 VLM을 사용할 수 있나요?**
   - EN: Yes, add a model adapter that exposes the same hook interface and update configs accordingly.
   - KO: 가능합니다. 동일한 훅 인터페이스를 제공하는 모델 어댑터를 추가하고 설정을 수정하세요.
2. **How large should the SAE latent ratio be? / SAE 잠재 차원 비율은 얼마나 커야 하나요?**
   - EN: Start with 4× the input dimension; monitor reconstruction error and activation sparsity.
   - KO: 입력 차원의 4배를 기본값으로 시작하고 재구성 오차와 활성 희소성을 모니터링하세요.
3. **Where do intervention results show up? / 개입 결과는 어디에 표시되나요?**
   - EN: `artifacts/eval/<run_id>/intervention/` contains metrics comparing gated vs baseline activations.
   - KO: `artifacts/eval/<run_id>/intervention/` 폴더에 게이팅과 기준 활성값 비교 지표가 저장됩니다.

---

_Last updated: 2025-10-02_

마지막 업데이트: 2025-10-02
