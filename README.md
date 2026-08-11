# TradeCode-LoRA

상품설명 텍스트를 입력받아 HS코드(품목분류코드)를 추천하는 경량 파인튜닝 모델
(Gemma2-2B + LoRA) 및 로컬 서빙 프로젝트.

## Status: ✅ 실험 5 완료 — 학습 카탈로그 범위 내에서만 안전하게 사용 가능함을 확인

Zero-shot baseline 측정 → LoRA 파인튜닝 → GGUF 변환/Ollama 서빙 → 재평가로
이어지는 전체 파이프라인을 구축했고, 세 차례 실험을 거치며 실패를 원인까지
추적해 다음 실험을 설계하는 과정을 반복한 끝에 실용적인 수준의 정확도에
도달했다.

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
- **실험 3** (완료): 관세청 공식 HS코드 품목분류표를 참고해 63/64/84/85류
  데이터를 추가하고(첫 자리 '6' 아닌 비중 22%까지 확보), completion을
  `{"hs_code": "NNNN"}` 형태로 단순화했다. 여기까지는 eval loss가 0.18까지
  정상적으로 떨어졌지만, 병합된 모델이 여전히 무의미한 텍스트만 생성하는
  새로운 문제에 부딪혔다 — 긴 조사 끝에 **unsloth가 학습에 쓰는 Gemma-2 계산
  방식이 순정 transformers/llama.cpp 구현과 근본적으로 달라서, unsloth 밖에서는
  병합 방법을 몇 가지로 바꿔봐도 항상 실패한다**는 걸 확인했다. 학습을 순정
  transformers + peft(QLoRA)로 바꿔 이 불일치 자체를 없앤 뒤 재학습 →
  **eval.jsonl 405건 기준 Exact Match 97.53%** 달성.
- **실험 4** (완료): 실험 3의 97.53%를 검증하는 과정에서 **train/eval 데이터
  누수**를 발견했다. `preprocess.py`가 원본 상품설명 1건과 그 패러프레이징
  3건(같은 상품을 문체만 바꿔 GPT-4o-mini로 증강한 것)을 클래스 단위로만
  나누고 개별 레코드 단위로 train/eval에 무작위 배정했던 탓에, eval 문장의
  87%(405건 중 354건)가 train에 있는 어떤 문장과 같은 클래스이면서 단어 절반
  이상 겹치는 근접 중복이었다 — 사실상 "같은 상품을 다른 말투로 다시 물어봤을
  때 기억해내는 능력"을 정확도로 측정하고 있었던 것. 같은 원본에서 나온
  레코드를 그룹으로 묶어 통째로 train 또는 eval 한쪽에만 배정하도록
  (group split) `preprocess.py`를 고치고, eval로 뽑힌 그룹은 원본만 남기고
  패러프레이징은 버려 근접 중복도 없앴다 → train 1,564건/eval 107건으로
  재구성해 재학습 → **eval.jsonl 107건 기준 Exact Match 93.46%**. 97.53%
  대비 -4.07%p로, 우려했던 것만큼 크게 떨어지지 않아 모델이 단순 암기를 넘어
  어느 정도 일반화 가능한 패턴을 배웠다는 근거가 됐다.
- **실험 5** (완료): 실험 4까지의 정확도는 모두 "학습 때 본 것과 같은 클래스"
  안에서 잰 것이라, "학습 때 아예 안 본 새 카테고리"에도 일반화하는지는 별개
  질문이었다. 85류(전기기기, 220건)를 학습 데이터에서 통째로 제외하고
  61/62/63/64/84류만으로 재학습 → 나머지 류에 대한 정확도는 94.79%로 유지됐지만
  (일반화 자체는 잘 됨), **85류에 대해서는 Exact/Partial(2자리)/Top-3 전부
  0.00%**였다. 같은 85류 220건을 파인튜닝 안 한 순정 `gemma2:2b`(zero-shot)에게
  물어보면 Partial Match(2자리)가 33.18%로, 사전학습 지식으로 최소한 "류"는
  어느 정도 맞히는 것과 대조적이었다 — **파인튜닝이 모델이 원래 갖고 있던
  지식을 지운 게 아니라, 학습한 카탈로그 범위 밖의 답을 아예 후보에서
  배제해버린 것**. 게다가 Parse Failure Rate는 0%(항상 형식은 멀쩡한 답을
  냄)라, "모르겠다"는 신호 없이 학습 범위 안의 엉뚱한 코드를 확신 있게
  내놓는다. 결론: 이 모델은 **학습 카탈로그 범위 안에서만** 안전하게 쓸 수
  있고, 범위 밖 상품이 들어올 수 있는 환경에서는 OOD 탐지나 사람 검수 없이
  단독 배포하면 위험하다.

자세한 경과는 [`docs/04-comparison.md`](docs/04-comparison.md)(실험 1),
[`docs/08-experiment2_training_log.md`](docs/08-experiment2_training_log.md)(실험 2 원인 분석),
[`docs/10-experiment3_investigation.md`](docs/10-experiment3_investigation.md)(실험 3 전체 조사 기록),
[`docs/13-experiment4_leakfixed_result.md`](docs/13-experiment4_leakfixed_result.md)(실험 4 데이터 누수
제거 및 재평가 결과), [`docs/16-experiment5_ch85heldout_result.md`](docs/16-experiment5_ch85heldout_result.md)
+ [`docs/17-experiment5_ch85_zeroshot_baseline.md`](docs/17-experiment5_ch85_zeroshot_baseline.md)
(실험 5 leave-one-chapter-out 결과)에 정리했다.

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
때문이다.

**실험 3** (4자리 타깃, 데이터 다양화 + completion 단순화 + unsloth 제거, 405건)

| 지표 | Fine-tuned (LoRA) |
|---|---|
| Exact Match (4자리) | **97.53%** |
| Partial Match (2자리) | 100.00% |
| Top-3 Recall | 97.53% |
| Parse Failure Rate | 0.00% |

원본 리포트: [`docs/11-experiment3_finetuned_result.md`](docs/11-experiment3_finetuned_result.md).

**실험 4** (4자리 타깃, train/eval 데이터 누수 제거, 107건 재평가)

| 지표 | Fine-tuned (LoRA) |
|---|---|
| Exact Match (4자리) | **93.46%** |
| Partial Match (2자리) | 99.07% |
| Top-3 Recall | 93.46% |
| Parse Failure Rate | 0.93% |

원본 리포트: [`docs/13-experiment4_leakfixed_result.md`](docs/13-experiment4_leakfixed_result.md).

**실험 5** (leave-one-chapter-out: 85류 제외 학습, code-length 4)

| 지표 | 학습 범위 내 (61/62/63/64/84류, 96건) | 85류 held-out (220건, fine-tuned) | 85류 (220건, zero-shot gemma2:2b) |
|---|---|---|---|
| Exact Match (4자리) | 94.79% | 0.00% | 2.73% |
| Partial Match (2자리) | 100.00% | 0.00% | 33.18% |
| Top-3 Recall | 94.79% | 0.00% | 5.91% |
| Parse Failure Rate | 0.00% | 0.00% | 4.09% |

학습 범위 내 정확도는 실험 4와 비슷하게 유지되지만(85류를 빼도 나머지 류
성능엔 영향 없음), 학습 때 아예 안 본 85류에 대해서는 fine-tuned 모델이
zero-shot보다도 못하다 — fine-tuning이 카탈로그 밖 지식을 아예 못 쓰게
좁혀버린다는 뜻.

원본 리포트: [`docs/15-experiment5_indist_result.md`](docs/15-experiment5_indist_result.md)
(학습 범위 내), [`docs/16-experiment5_ch85heldout_result.md`](docs/16-experiment5_ch85heldout_result.md)
(85류 held-out, fine-tuned), [`docs/17-experiment5_ch85_zeroshot_baseline.md`](docs/17-experiment5_ch85_zeroshot_baseline.md)
(85류, zero-shot).

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
| [`docs/10-experiment3_investigation.md`](docs/10-experiment3_investigation.md) | 실험 3 전체 조사 기록 (unsloth 근본 원인 규명 포함) |
| [`docs/11-experiment3_finetuned_result.md`](docs/11-experiment3_finetuned_result.md) | 실험 3 최종 파인튜닝 재평가 결과 (405건, Exact 97.53%) |
| [`docs/12-experiment4_leakfixed_training_log.md`](docs/12-experiment4_leakfixed_training_log.md) | 실험 4 학습 로그 (누수 제거 데이터로 재학습) |
| [`docs/13-experiment4_leakfixed_result.md`](docs/13-experiment4_leakfixed_result.md) | 실험 4 파인튜닝 재평가 결과 (107건, Exact 93.46%) |
| [`docs/14-experiment5_ch85heldout_training_log.md`](docs/14-experiment5_ch85heldout_training_log.md) | 실험 5 학습 로그 (85류 제외 학습) |
| [`docs/15-experiment5_indist_result.md`](docs/15-experiment5_indist_result.md) | 실험 5 학습 범위 내(61/62/63/64/84류) 재평가 결과 (96건, Exact 94.79%) |
| [`docs/16-experiment5_ch85heldout_result.md`](docs/16-experiment5_ch85heldout_result.md) | 실험 5 85류 held-out 평가 결과 (220건, fine-tuned, 전부 0%) |
| [`docs/17-experiment5_ch85_zeroshot_baseline.md`](docs/17-experiment5_ch85_zeroshot_baseline.md) | 실험 5 85류 zero-shot 베이스라인 (220건, gemma2:2b, Partial 33.18%) |

## 폴더 구조

```
tradecode-lora/
├── data/
│   ├── raw/                 원본 CSV (git 제외)
│   ├── processed/           augmented.jsonl 등 중간 산출물 (git 제외)
│   └── processed_simple/    실제 학습에 쓰는 train/eval jsonl, --simple-target
│                            + group split 적용 (git 제외, 실험 3~)
├── src/
│   ├── finetune/
│   │   ├── extract_hs_reference.py  관세청 HS코드 품목분류표에서 특정 류 참조
│   │   │                            데이터 추출 (클래스 다양화용, 실험 3~)
│   │   ├── augment.py        원본 설명문 패러프레이징 증강 (OpenAI API)
│   │   ├── preprocess.py     증강 데이터 -> instruction 포맷 변환 + train/eval split
│   │   ├── train.py          순정 transformers + peft(QLoRA) LoRA 학습 (Colab/Kaggle GPU 전제)
│   │   ├── merge_adapter_plain.py  LoRA adapter -> fp16 병합 (순정 peft, 현재 사용)
│   │   └── merge_adapter.py  LoRA adapter -> 16bit 병합 (Unsloth 공식 API, 더 이상 안 씀 - docs/10 참고)
│   ├── eval/
│   │   └── baseline_eval.py  Ollama 서빙 모델 평가 (zero-shot/fine-tuned 겸용)
│   └── serving/
│       └── build_ollama_model.sh  GGUF 변환 + 양자화 + Ollama 등록
│                                   (OUT_DIR/MODEL_NAME 환경변수로 출력 경로/모델명
│                                   오버라이드 가능 - 기존 모델과 비교하며 재학습할 때 사용)
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
python src/finetune/preprocess.py --input data/processed/augmented.jsonl --output-dir data/processed_simple --code-length 4 --simple-target
```
`preprocess.py`의 `stratified_split`은 같은 원본 상품에서 나온 레코드(원본 +
패러프레이징들)를 `_group_id`로 묶어 그룹째로 train 또는 eval 한쪽에만
배정한다(실험 4에서 추가된 group-aware split — 자세한 배경은
[`docs/13-experiment4_leakfixed_result.md`](docs/13-experiment4_leakfixed_result.md)
참고). `--simple-target`은 completion을 `{"hs_code": "NNNN"}`로 단순화하는
옵션으로, 실험 3부터 계속 이 설정을 쓴다.

### 2. Baseline 평가 (로컬, Ollama 필요)
```
python src/eval/baseline_eval.py --model gemma2:2b --prompt-style zero_shot --output docs/01-baseline_result.md
```

### 3. LoRA 학습 (Colab/Kaggle GPU)
```
pip install -r requirements-colab.txt
# data/processed_simple/{train,eval}.jsonl을 Kaggle Dataset으로 업로드 후 불러오기
python src/finetune/train.py --smoke-test --max-steps 60   # 사전 확인
python src/finetune/train.py \
    --lora-r 32 --lora-alpha 64 \
    --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
python src/finetune/merge_adapter_plain.py --adapter-dir outputs/adapter --output-dir outputs/merged_plain
```
(순정 transformers + peft(QLoRA) 기반. unsloth를 쓰던 `merge_adapter.py`는
더 이상 쓰지 않는다 — 이유는 `docs/10-experiment3_investigation.md` 참고.
`notebooks/experiment3_retrain.ipynb`에 Kaggle에서 그대로 실행 가능한 노트북
버전이 있다.)

### 4. GGUF 변환 + Ollama 등록 (로컬)
```
bash src/serving/build_ollama_model.sh outputs/merged_plain
```

### 5. 재평가 (로컬)
```
python src/eval/baseline_eval.py --eval-file data/processed_simple/eval.jsonl \
    --model tradecode-gemma2 --prompt-style finetuned --code-length 4 \
    --output docs/13-experiment4_leakfixed_result.md
```
(`--code-length`는 `preprocess.py`로 데이터를 만들 때 쓴 자릿수와 반드시 맞춰야
한다 — 실험 3부터 4자리.)

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
- **학습 프레임워크와 서빙 프레임워크는 같은 계산 경로를 써야 한다**: 실험 3에서
  eval loss가 0.18까지 정상적으로 떨어진 unsloth 학습 결과물이, 병합 방법을
  6가지 넘게 바꿔가며 시도해도 unsloth 밖(순정 transformers, GGUF/llama.cpp)에서는
  전부 실패했다. 원인은 저장/병합 버그가 아니라 unsloth가 Gemma-2를 계산하는
  방식 자체가 순정 구현과 근본적으로 달랐던 것 — 학습에 쓴 프레임워크의 최적화가
  실제 서빙 환경과 다른 계산을 한다면, 아무리 가중치를 정확히 옮겨도 재현되지
  않는다. 학습·서빙이 같은 계산 경로를 쓰도록 처음부터 설계하는 게 훨씬 안전하다
  (`docs/10-experiment3_investigation.md` 참고).
- **loss가 낮아도 실제 생성 테스트 전에는 아무것도 확정 짓지 말 것**: 이번
  프로젝트에서 "loss는 정상인데 생성은 무의미함" 패턴이 실험 1, 실험 3(1·2·3차
  재학습 전부)에서 반복됐다. 원인은 매번 달랐지만(스키마만 학습, 병합 버그로
  오진단했던 프레임워크 불일치, 마지막엔 Ollama의 채팅 템플릿 자동 적용까지)
  공통점은 하나 — teacher-forcing 기반 loss만으로는 절대 알 수 없고, 실제
  자유 생성(`generate()`)을 몇 건이라도 직접 봐야 잡히는 문제였다는 것.
- **추론 서버(Ollama 등)가 프롬프트에 무언가를 몰래 덧붙일 수 있다**: Ollama의
  `/api/generate`는 `raw: true`를 안 주면 모델의 채팅 템플릿을 자동으로 씌운다.
  파인튜닝 모델을 순수 Alpaca 스타일 프롬프트(채팅 템플릿 아님)로 학습했다면,
  이 자동 래핑 때문에 학습 때와 다른 입력이 들어가 겉보기엔 "병합이 잘못됐나?"
  싶은 증상이 재현된다. 재평가 스크립트(`baseline_eval.py`)가 이 버그를 그대로
  갖고 있었다면 정량 평가 결과 전체가 조용히 오염됐을 것이다.
- **데이터 증강은 클래스 단위가 아니라 원본(소스) 단위로 split해야 한다**:
  실험 3의 `preprocess.py`는 "같은 클래스 내에서 원본을 증강본보다 eval에
  우선 배정"하는 방식으로 데이터 누수를 어느 정도 막으려 했지만, 정작 같은
  원본에서 나온 패러프레이징들끼리는 여전히 train/eval에 무작위로 흩어졌다.
  그 결과 eval 문장의 87%가 train에 있는 근접 중복이었다 — "클래스가 겹치지
  않게"만으로는 부족하고, "패러프레이징의 원본 소스가 겹치지 않게"까지
  split 단위를 좁혀야 한다는 걸 확인했다. 고친 뒤에도(누수 제거 후에도)
  93%대 정확도가 유지된 게 오히려 이 모델이 완전한 암기는 아니었다는 걸
  보여준 셈이다.
- **"일반화"는 한 단어가 아니라 최소 두 축으로 나눠서 물어야 한다**: 실험
  4까지의 정확도는 전부 "학습 때 본 것과 같은 클래스" 안에서 "표현만 다른
  입력"에 대한 일반화였다. 이것과 "학습 때 아예 안 본 새 클래스/카테고리"에
  대한 일반화는 완전히 다른 질문이고, 전자가 잘 된다고 후자도 잘 되는 게
  아니다(실험 5: 전자 94.79%, 후자 0.00%). 평가 설계를 할 때 "무엇에 대해
  일반화를 주장하는지"를 먼저 명시해야, 숫자 하나로 성능을 과장하거나
  과소평가하지 않는다.
- **파인튜닝은 지식을 안 넣어줄 뿐 아니라, 있던 지식도 못 쓰게 좁힐 수 있다**:
  85류를 뺀 파인튜닝 모델은 85류에서 Partial Match(2자리)조차 0%였는데,
  파인튜닝 안 한 순정 베이스 모델(zero-shot)은 같은 85류에서 33.18%를
  기록했다. 베이스 모델이 사전학습 때 이미 갖고 있던 지식을, 좁은 카탈로그로
  파인튜닝하면서 "학습한 라벨 공간 밖은 아예 후보에서 배제"하는 방향으로
  덮어써버린 것 — 게다가 Parse Failure Rate는 0%(항상 형식은 멀쩡함)라
  "모르겠다"는 신호도 없이 확신 있게 틀린 답을 낸다. 좁은 도메인으로
  파인튜닝한 모델을 배포할 땐 "학습 카탈로그 밖의 입력이 들어올 수 있는가"를
  반드시 확인하고, 그렇다면 OOD 탐지나 사람 검수 단계를 반드시 같이 설계해야
  한다.
