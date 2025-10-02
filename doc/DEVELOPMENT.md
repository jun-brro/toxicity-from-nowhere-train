# Development Guide / 개발 가이드

> This document explains how the repository is organized, how each subsystem fits together, and what conventions to follow when extending or debugging the intrinsic toxicity pipeline.
> 이 문서는 저장소 구조와 각 하위 시스템의 역할, 그리고 내재적 독성 파이프라인을 확장하거나 디버깅할 때 따라야 할 규칙을 설명합니다.

---

## 1. Repository Map / 저장소 구조

| Path | Description (EN) | 설명 (KO) |
| --- | --- | --- |
| `vlm_intrinsic_tox/` | Core Python package with extraction, SAE training, evaluation, and utilities modules. | 추출, SAE 학습, 평가, 유틸리티 모듈이 포함된 핵심 파이썬 패키지. |
| `vlm_intrinsic_tox/cli/` | CLI entrypoints wrapping each stage of the pipeline. | 파이프라인 단계별 CLI 엔트리포인트. |
| `vlm_intrinsic_tox/extract/` | Hooks, pooling, sharding logic for δ_int feature extraction. | δ_int 특징 추출용 훅, 풀링, 샤딩 로직. |
| `vlm_intrinsic_tox/sae/` | Top‑K sparse autoencoder modules and training helpers. | Top‑K 희소 오토인코더 모듈과 학습 헬퍼. |
| `vlm_intrinsic_tox/eval/` | Latent metrics, intervention, report builders. | 잠재 공간 지표, 개입, 리포트 구성 도구. |
| `vlm_intrinsic_tox/data/` | Dataset adapters, prompt templates, label plumbing. | 데이터셋 어댑터, 프롬프트 템플릿, 라벨 처리. |
| `configs/` | YAML defaults for extraction, SAE training, evaluation. | 추출, SAE 학습, 평가용 기본 YAML 설정. |
| `doc/` | Markdown guides for development and usage. | 개발 및 사용 가이드가 담긴 마크다운 문서. |
| `README.md` | Quick overview and references to detailed docs. | 개요 및 상세 문서 링크. |

---

## 2. Local Environment / 로컬 환경 준비

### 2.1 Tooling
- **Python**: 3.10 이상.
- **Poetry** 또는 `pip`를 사용하여 `pyproject.toml`의 종속성을 설치합니다.
- **GPU**: 추출/학습 단계는 CUDA가 있는 GPU에서 실행하는 것을 권장합니다.

### 2.2 Setup Steps
1. (EN) Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
2. (KO) 가상 환경을 만들고 종속성을 설치합니다:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
3. (EN) Download Hugging Face model weights (`AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf`) and ensure you are authenticated if needed.
4. (KO) 허깅페이스 모델 가중치(`AIML-TUDA/LlavaGuard-v1.2-7B-OV-hf`)를 내려받고 필요 시 인증을 완료합니다.

---

## 3. Core Workflow Concepts / 핵심 워크플로 개념

### 3.1 δ_int Extraction / δ_int 추출
- (EN) `extract/runner.py` orchestrates two forwards per batch (cross-attention on/off) using hooks defined in `extract/hooks.py` and writes pooled activations via `extract/sharding.py`.
- (KO) `extract/runner.py`는 `extract/hooks.py`에 정의된 훅으로 배치마다 교차-어텐션 on/off 두 번의 순전파를 실행하고, `extract/sharding.py`를 사용해 풀링된 활성값을 저장합니다.

### 3.2 Top-K SAE Training / Top-K SAE 학습
- (EN) `sae/train.py` standardizes shard data, instantiates `sae/modules.py::TopKSAE`, runs training, and saves checkpoints & scalers in `sae/io.py`.
- (KO) `sae/train.py`는 샤드 데이터를 표준화하고 `sae/modules.py::TopKSAE`를 생성하여 학습을 진행하며, 결과 체크포인트와 스케일러는 `sae/io.py`로 저장합니다.

### 3.3 Evaluation & Intervention / 평가 및 개입
- (EN) `eval/metrics.py` computes dfreq/dmag/AUROC/KL, `eval/analyze.py` ranks latents, `eval/intervene.py` applies gating, and `eval/report.py` emits Markdown/JSON summaries.
- (KO) `eval/metrics.py`가 dfreq/dmag/AUROC/KL을 계산하고, `eval/analyze.py`가 잠재벡터를 순위화하며, `eval/intervene.py`가 게이팅을 적용하고, `eval/report.py`가 Markdown/JSON 요약을 생성합니다.

---

## 4. Development Conventions / 개발 규칙

### 4.1 Coding Standards / 코딩 표준
- (EN) Prefer functional decomposition per module with explicit dataclasses for configuration (`config.py`). Keep hooks pure and side-effect free.
- (KO) 각 모듈을 기능 단위로 분해하고 설정은 `config.py`의 dataclass로 명시합니다. 훅은 순수 함수 형태로 유지합니다.

### 4.2 Testing & Validation / 테스트 및 검증
- (EN) Run unit smoke tests (e.g., `python -m compileall vlm_intrinsic_tox`) and stage-specific dry-runs on tiny subsets (`configs/*` expose overrides for `limit_examples`).
- (KO) 단위 스모크 테스트(`python -m compileall vlm_intrinsic_tox`)와 단계별 소량 데이터 드라이런을 실행합니다(`configs/*`에 `limit_examples` 오버라이드 존재).

### 4.3 Logging & Reproducibility / 로깅 및 재현성
- (EN) `utils/logging.py` centralizes structured logging; include git SHA, HF revisions, and random seeds in each artifact.
- (KO) `utils/logging.py`에서 구조화된 로깅을 관리하며, 각 산출물에 git SHA, HF 리비전, 랜덤 시드를 포함합니다.

---

## 5. Extending the Pipeline / 파이프라인 확장 방법

1. **Add a new dataset adapter / 새로운 데이터셋 어댑터 추가**
   - EN: Create a class in `vlm_intrinsic_tox/data/datasets.py` implementing `iter_samples`; wire it into CLI config registry.
   - KO: `vlm_intrinsic_tox/data/datasets.py`에 `iter_samples`를 구현하는 클래스를 만들고 CLI 설정 레지스트리에 연결합니다.
2. **Introduce a pooling strategy / 새로운 풀링 전략 추가**
   - EN: Implement in `extract/pooling.py` and register inside pooling factory.
   - KO: `extract/pooling.py`에 구현하고 풀링 팩토리에 등록합니다.
3. **Add metrics / 지표 추가**
   - EN: Extend `eval/metrics.py`; update `eval/report.py` to display new columns.
   - KO: `eval/metrics.py`를 확장하고 `eval/report.py`에 새 컬럼을 반영합니다.

---

## 6. Debugging Tips / 디버깅 팁

- EN: Use verbose logging (`logging_level=DEBUG`) to inspect ON/OFF norms; confirm cross-attention hooks zero tensors by printing `tensor.norm()`. Enable `limit_examples` during iteration.
- KO: 자세한 로깅(`logging_level=DEBUG`)으로 ON/OFF 노름을 확인하고, `tensor.norm()` 출력으로 교차-어텐션 훅이 0 텐서를 반환하는지 검증합니다. 반복 중에는 `limit_examples`를 사용하세요.

---

## 7. Contribution Checklist / 기여 체크리스트

- [ ] EN: Added or updated tests/smoke runs for new features.
- [ ] KO: 새로운 기능에 대한 테스트/스모크 실행을 추가 또는 갱신했는가?
- [ ] EN: Documented new configs, CLI flags, or artifacts.
- [ ] KO: 신규 설정, CLI 플래그, 산출물을 문서화했는가?
- [ ] EN: Ensured artifacts record seeds and revisions.
- [ ] KO: 산출물에 시드와 리비전을 기록했는가?

---

_Last updated: 2025-10-02_

마지막 업데이트: 2025-10-02
