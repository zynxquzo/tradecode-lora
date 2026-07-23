# TradeCode-LoRA

상품설명 텍스트를 입력받아 HS코드(품목분류코드)를 추천하는 경량 파인튜닝 모델
(Gemma2-2B + LoRA) 및 로컬 서빙 프로젝트.

## Status: ✅ 완료

Zero-shot baseline 측정 → LoRA 파인튜닝 → GGUF 변환/Ollama 서빙 → 재평가까지
전체 파이프라인을 구축하고 끝까지 실행했다. 과정에서 trl/unsloth 버전 호환성
이슈를 여러 건 근본 원인까지 추적해 해결했고, completion-only loss masking 도입
전후로 학습 loss가 크게 개선되는 것도 확인했다(perplexity 환산 약 6500 → 약 60).

다만 최종 정량 평가에서 정확도 지표는 baseline 대비 개선되지 않았다 — loss
개선이 실제 HS코드 생성 능력으로는 이어지지 않았다는 뜻이다. 이 간극이 왜
발생했는지에 대한 원인 분석과 다음 개선 방향을 [`docs/04-comparison.md`](docs/04-comparison.md)에
정리했다.

## 결과 요약

| 지표 | Baseline (zero-shot) | Fine-tuned (LoRA) |
|---|---|---|
| Exact Match (6자리) | 0.36% | 0.00% |
| Partial Match (4자리) | 3.57% | 0.00% |
| Partial Match (2자리) | 38.93% | 0.00% |
| Top-3 Recall | 0.71% | 0.00% |
| Parse Failure Rate | (baseline 리포트엔 없음) | 100.00% |

파인튜닝 후 eval loss는 8.88 → 4.09까지 꾸준히 개선됐다. 모델은
`{"hs_code": ..., "confidence_basis": ...}` JSON **스키마의 형태**는 익혔지만,
**6자리 숫자 자체를 생성하는 법은 배우지 못했다** — 880건/210클래스라는 데이터
희소성과, LoRA를 attention projection에만 적용한 설정이 유력한 원인으로 보인다.
자세한 원인 분석과 다음 시도 방향은 [`docs/04-comparison.md`](docs/04-comparison.md)에 정리했다.

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/00-project-plan.md`](docs/00-project-plan.md) | 최초 기획안 |
| [`docs/01-baseline_result.md`](docs/01-baseline_result.md) | zero-shot baseline 평가 결과 |
| [`docs/02-training_log.md`](docs/02-training_log.md) | LoRA 학습 로그 (loss curve) |
| [`docs/03-finetuned_result.md`](docs/03-finetuned_result.md) | 파인튜닝 후 재평가 결과 (280건) |
| [`docs/04-comparison.md`](docs/04-comparison.md) | baseline vs fine-tuned 비교 및 원인 분석 |

## 폴더 구조

```
tradecode-lora/
├── data/
│   ├── raw/                 원본 CSV (git 제외)
│   └── processed/           instruction 포맷 jsonl (train/eval/augmented)
├── src/
│   ├── finetune/
│   │   ├── augment.py        원본 설명문 패러프레이징 증강 (OpenAI API)
│   │   ├── preprocess.py     증강 데이터 -> instruction 포맷 변환 + train/eval split
│   │   ├── train.py          Unsloth LoRA 학습 (Colab/Kaggle GPU 전제)
│   │   └── merge_adapter.py  LoRA adapter -> 16bit 병합 (Unsloth 공식 API)
│   ├── eval/
│   │   └── baseline_eval.py  Ollama 서빙 모델 평가 (zero-shot/fine-tuned 겸용)
│   └── serving/
│       └── build_ollama_model.sh  GGUF 변환 + 양자화 + Ollama 등록
├── docs/                     실험 로그 및 결과 리포트
├── requirements.txt          로컬(평가/증강)용 의존성
└── requirements-colab.txt    학습(GPU)용 의존성
```

## 재현 방법

### 1. 데이터 준비 (로컬)
```
python src/finetune/augment.py --input data/raw/products_real.csv --output data/processed/augmented.jsonl
python src/finetune/preprocess.py --input data/processed/augmented.jsonl --output-dir data/processed
```

### 2. Baseline 평가 (로컬, Ollama 필요)
```
python src/eval/baseline_eval.py --model gemma2:2b --prompt-style zero_shot --output docs/01-baseline_result.md
```

### 3. LoRA 학습 (Colab/Kaggle GPU)
```
pip install -r requirements-colab.txt
python src/finetune/train.py --smoke-test --max-steps 60   # 사전 확인
python src/finetune/train.py                                # 전체 학습
python src/finetune/merge_adapter.py --adapter-dir outputs/adapter --output-dir outputs/merged
```

### 4. GGUF 변환 + Ollama 등록 (로컬)
```
bash src/serving/build_ollama_model.sh outputs/merged
```

### 5. 재평가 (로컬)
```
python src/eval/baseline_eval.py --model tradecode-gemma2 --prompt-style finetuned --output docs/03-finetuned_result.md
```

## 배운 점

- **trl/unsloth 버전 호환성**: `SFTConfig`/`SFTTrainer`의 파라미터명이 trl 버전마다
  바뀌고(`max_seq_length`→`max_length`, `tokenizer`→`processing_class`), unsloth가
  VRAM 절약을 위해 `outputs.logits`를 지연 계산용으로 바꾸는 등 라이브러리 조합
  특유의 호환성 문제가 다수 있었다 (`src/finetune/train.py` 상단 docstring에 기록).
- **loss 개선 ≠ 태스크 성능 개선**: completion-only loss masking으로 eval loss를
  perplexity 기준 100배 이상 낮췄지만, 실제 정확도는 0%에서 벗어나지 못했다.
  손실 함수가 측정하는 것과 실제로 원하는 능력 사이의 간극을 정량적으로 보여주는
  사례.
- **소규모 데이터/모델의 한계**: 880건/210클래스 조합은 2B급 모델이 세부 분류
  코드를 암기하기엔 부족한 규모였을 가능성이 높다.
