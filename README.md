# TradeCode-LoRA

상품설명 텍스트를 입력받아 HS코드(품목분류코드)를 추천하는 경량 파인튜닝 모델
(Gemma2-2B + LoRA) 및 로컬 서빙 프로젝트.

## Status: 🚧 실험 3 진행 중

Zero-shot baseline 측정 → LoRA 파인튜닝 → GGUF 변환/Ollama 서빙 → 재평가로
이어지는 전체 파이프라인을 구축했고, 두 차례 실험을 거치며 실패를 원인까지
추적해 다음 실험을 설계하는 과정을 반복하고 있다.

- **실험 1** (6자리 소호 타깃, 61/62류 한정): loss는 크게 개선됐지만
  (perplexity 약 6500 → 약 60) 실제 정확도는 baseline·fine-tuned 모두
  0%대에 그쳤다. 원인은 880건/210클래스라는 극심한 클래스 희소성.
- **실험 2** (4자리 호 타깃으로 축소, 데이터 1,795건까지 증강, LoRA 용량 확대):
  eval loss는 0.99까지 떨어져 언뜻 개선처럼 보였으나, 재평가하니 오히려
  baseline보다 나빠졌다(Exact 2.61% → 0.40%). 직접 생성해보니 모델이 상품
  설명과 무관하게 숫자 "6"만 반복하는 **완전한 mode collapse** 상태였다.
  원인은 학습 데이터(`products_real.csv`)가 61/62류(의류)로만 구성돼 정답의
  첫 자리가 100% "6"이었던 것 — 여기에 LoRA 용량 확대가 겹치며 "6만 찍으면
  loss가 잘 떨어진다"는 지름길로 완전히 최적화됐다.
- **실험 3** (진행 중): 관세청 공식 HS코드 품목분류표를 참고해 63/64/84/85류
  데이터를 추가하고(첫 자리 '6' 아닌 비중 22%까지 확보), LoRA 용량은 실험 1
  수준으로 되돌렸다. 데이터 준비는 끝났고 Colab 재학습 결과를 기다리는 중.

자세한 경과는 [`docs/04-comparison.md`](docs/04-comparison.md)(실험 1),
[`docs/08-experiment2_training_log.md`](docs/08-experiment2_training_log.md)(실험 2 원인 분석),
[`docs/09-experiment3-plan.md`](docs/09-experiment3-plan.md)(실험 3 계획)에 정리했다.

## 결과 요약

**실험 1** (6자리 타깃, 61/62류만)

| 지표 | Baseline (zero-shot) | Fine-tuned (LoRA) |
|---|---|---|
| Exact Match (6자리) | 0.36% | 0.00% |
| Partial Match (4자리) | 3.57% | 0.00% |
| Partial Match (2자리) | 38.93% | 0.00% |
| Top-3 Recall | 0.71% | 0.00% |
| Parse Failure Rate | (baseline 리포트엔 없음) | 100.00% |

**실험 2** (4자리 타깃, 데이터 증강 + LoRA 용량 확대, 498건 재평가)

| 지표 | Baseline (zero-shot) | Fine-tuned (LoRA) |
|---|---|---|
| Exact Match (4자리) | 2.61% | 0.40% |
| Partial Match (2자리) | 38.55% | 7.63% |
| Top-3 Recall | 6.22% | 0.40% |
| Parse Failure Rate | 0.20% | 10.84% |

파인튜닝 모델이 baseline보다 전 지표에서 나쁘다 — 위에서 설명한 mode collapse
때문이다. 실험 3 결과가 나오면 이 표에 이어서 추가할 예정이다.

## 문서

| 문서 | 내용 |
|---|---|
| [`docs/00-project-plan.md`](docs/00-project-plan.md) | 최초 기획안 |
| [`docs/01-baseline_result.md`](docs/01-baseline_result.md) | 실험 1 zero-shot baseline 평가 결과 |
| [`docs/02-training_log.md`](docs/02-training_log.md) | 실험 1 LoRA 학습 로그 (loss curve) |
| [`docs/03-finetuned_result.md`](docs/03-finetuned_result.md) | 실험 1 파인튜닝 후 재평가 결과 (280건) |
| [`docs/04-comparison.md`](docs/04-comparison.md) | 실험 1 baseline vs fine-tuned 비교 및 원인 분석 |
| [`docs/05-experiment2-plan.md`](docs/05-experiment2-plan.md) | 실험 2 계획 (4자리 타깃 전환 + 데이터 밀도 개선) |
| [`docs/06-experiment2_baseline_result.md`](docs/06-experiment2_baseline_result.md) | 실험 2 zero-shot baseline 재평가 결과 |
| [`docs/07-experiment2_finetuned_result.md`](docs/07-experiment2_finetuned_result.md) | 실험 2 파인튜닝 후 재평가 결과 |
| [`docs/08-experiment2_training_log.md`](docs/08-experiment2_training_log.md) | 실험 2 학습 로그 + mode collapse 원인 분석 |
| [`docs/09-experiment3-plan.md`](docs/09-experiment3-plan.md) | 실험 3 계획 (데이터 다양화 + LoRA 설정 원복) |

## 폴더 구조

```
tradecode-lora/
├── data/
│   ├── raw/                 원본 CSV (git 제외)
│   └── processed/           instruction 포맷 jsonl (train/eval/augmented)
├── src/
│   ├── finetune/
│   │   ├── extract_hs_reference.py  관세청 HS코드 품목분류표에서 특정 류 참조
│   │   │                            데이터 추출 (클래스 다양화용, 실험 3~)
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
# (선택) 새 류를 추가하려면: 관세청 HS코드 품목분류표 엑셀에서 참조 데이터 추출
python src/finetune/extract_hs_reference.py --xlsx "<엑셀 경로>" \
    --chapters 63,64,84,85 --per-chapter 55 --output data/raw/products_new_chapters.csv
# -> 추출 결과를 data/raw/products_real.csv에 직접 병합해서 사용

python src/finetune/augment.py --input data/raw/products_real.csv --output data/processed/augmented.jsonl
python src/finetune/preprocess.py --input data/processed/augmented.jsonl --output-dir data/processed --code-length 4
```

### 2. Baseline 평가 (로컬, Ollama 필요)
```
python src/eval/baseline_eval.py --model gemma2:2b --prompt-style zero_shot --output docs/01-baseline_result.md
```

### 3. LoRA 학습 (Colab/Kaggle GPU)
```
pip install -r requirements-colab.txt
python src/finetune/train.py --smoke-test --max-steps 60   # 사전 확인
python src/finetune/train.py                                # 전체 학습 (기본값 = lora_r 16 / attention-only)
python src/finetune/merge_adapter_plain.py --adapter-dir outputs/adapter --output-dir outputs/merged
```
(순정 transformers + peft(QLoRA) 기반. unsloth를 쓰던 `merge_adapter.py`는
더 이상 쓰지 않는다 — 이유는 `docs/10-experiment3_investigation.md` 참고.)

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
- **eval loss 개선이 mode collapse를 가릴 수 있다**: 실험 2에서 eval loss가
  큰 폭으로 떨어져 개선처럼 보였지만, 실제로는 정답 클래스가 전부 "6"으로
  시작하는 데이터 편중을 모델이 "무조건 6이라고 답하기"로 외운 결과였다. loss
  곡선만 보고 성공을 판단하면 안 되고, 학습 직후 실제 생성 샘플을 몇 개라도
  직접 눈으로 확인하는 절차가 반드시 필요하다는 걸 확인했다.
- **클래스 분포는 라벨 자릿수보다 먼저 봐야 한다**: 목표 자릿수(6자리→4자리)를
  낮춰 클래스 희소성을 줄여도, 데이터 자체가 특정 카테고리(류)에 편중돼 있으면
  분류기가 아니라 편중 암기기를 학습하게 된다. 데이터셋을 구성할 때 클래스별
  샘플 수뿐 아니라 "정답 값의 자릿수별 분포"도 함께 점검해야 한다.
