# 실험 2 계획: 4자리(호) 타겟 전환 + 데이터 밀도 개선

실험 1(`docs/02-training_log.md`, `docs/03-finetuned_result.md`, `docs/04-comparison.md`)은
loss는 크게 개선됐지만(perplexity 약 6500→약 60) 실제 6자리 HS코드 생성 정확도는
0%였다. 이 문서는 그 실패를 어떻게 재설계해서 다시 시도할지 정리한 **실행 전
계획서**다 — 아직 실행하지 않았고, 학습/평가 결과가 나오면 `06-experiment2_training_log.md`
이후 번호로 결과 문서를 이어 작성한다.

## 1. 문제 재정의: 왜 6자리가 안 됐는가

`docs/04-comparison.md`는 "880건/210클래스 = 클래스당 평균 4건은 2B 모델이 6자리를
암기하기엔 근본적으로 부족한 규모"라고 짚었다. raw 데이터(`data/raw/products_real.csv`,
290행)를 다시 집계해보면 문제가 더 뿌리 깊다:

| 기준 | 클래스 수 | 클래스당 평균 원본 건수 | 단일 샘플 클래스 |
|---|---|---|---|
| 6자리(소호) | 210 | 1.4건 | 144개 |
| 4자리(호) | 34 | 8.5건 | 0개(최소 2건) |

`augment.py`는 같은 원본 문장을 문체만 바꿔 증강하므로(패러프레이징), 클래스당 실제
**정보량**은 증강 비율을 아무리 올려도 거의 늘지 않는다 — 즉 실험 1의 근본 문제는
"데이터 총량 부족"이 아니라 "6자리 기준 클래스 희소성"이었다.

반면 4자리(호) 기준으로는 같은 raw 데이터만으로도 밀도가 6배 이상 좋다. 이번 실험은
이 사실을 이용해 **모델이 생성하는 주 타겟을 6자리 → 4자리(호)로 낮춘다.**

- 트레이드오프: 프로젝트 원래 목표(`docs/00-project-plan.md`)인 "6자리 HS코드 Top-3
  추천"보다 세분화 수준이 낮아진다. 이번 실험에서는 "4자리 호 단위 분류가 2B급 모델
  파인튜닝으로 가능한가"를 먼저 검증하고, 성공하면 6자리 도전은 실험 3 후보로 남긴다.

## 2. 데이터 설계

### 2.1 얇은 클래스 보강 (원본 소량 추가 수집)

4자리 헤딩 34개 중 샘플 5건 미만인 11개:

| heading | 현재 건수 |
|---|---|
| 6213, 6216, 6217 | 2건 |
| 6113, 6215 | 3건 |
| 6101, 6105, 6106, 6109, 6114, 6205 | 4건 |

이 11개 헤딩을 헤딩당 최소 8건까지 채우는 것을 목표로, `00-project-plan.md`에 이미
정리된 소스 후보(관세청 품목분류 사전회시 사례, 수출입무역통계 품목명-HS코드 매핑)에서
소량 추가 수집한다. 목표 raw 총량: 약 380~450행(현재 290행).

### 2.2 패러프레이징 증강 비율 확대

```
python src/finetune/augment.py --input data/raw/products_real.csv \
    --output data/processed/augmented.jsonl --n-per-item 6
```

기존 `--n-per-item 3`(원본 1건당 3건 증강, 총 4배)에서 `6`(총 7배)으로 확대한다.

### 2.3 4자리 기준 stratified split

```
python src/finetune/preprocess.py --input data/processed/augmented.jsonl \
    --output-dir data/processed --code-length 4
```

`preprocess.py --code-length 4`(신규 플래그)로 4자리 헤딩 기준 stratified train/eval
split을 수행한다. instruction 문구도 자동으로 "HS코드를 4자리까지 추천"으로 바뀐다.

### 2.4 예상 결과

raw 380~450행 × 7배 증강 ≈ 2,660~3,150건 → 80/20 split 후 **train 약 2,100~2,500건,
eval 약 530~630건** (실험 1의 880/280 대비 약 2.5~3배).

## 3. 파인튜닝 설계

`docs/04-comparison.md`의 "향후 개선 방향" 1, 2번을 반영:

- `lora_r: 16 → 32`
- `--target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
  (신규 플래그, attention뿐 아니라 MLP까지 LoRA 적용 — 숫자 암기 같은 사실적 지식
  주입에는 MLP 용량이 중요하다는 보고를 반영)
- `lora_alpha: 32 → 64` (rank를 2배로 올린 만큼 alpha/r 비율 유지)

### 3.1 epoch 축소 (Kaggle 세션 예산에 맞춤)

데이터가 약 2.5~3배 늘면 epoch당 step 수도 비례해서 늘어난다(유효 배치 16 기준,
현재 880건→55 step/epoch, 신규 2,100~2,500건→131~156 step/epoch). 실험 1은
10 epoch·500 step을 단일 Kaggle 세션 안에서 무리 없이 끝냈으므로, 이번에도 총
step 수를 비슷한 범위로 유지하는 것을 목표로 **epoch을 10 → 4**로 줄인다
(4 epoch ≈ 525~625 step, 실험 1과 같은 자릿수).

다만 실험 1 로그(`02-training_log.md`)에는 wall-clock 시간이 기록되어 있지 않아
"9~12시간 세션에 500 step이 실제로 몇 시간 걸렸는지"는 추정치다. 이번 `train.py`에
`elapsed_min` 컬럼을 로그에 추가했으니(신규), 본 학습 전 스모크 테스트로 실측한 뒤
epoch 수를 조정한다:

```
python src/finetune/train.py --smoke-test --max-steps 60 \
    --lora-r 32 --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora-alpha 64
# docs/02-training_log.md의 elapsed_min으로 step당 소요 시간 확인 후
# (9~12시간 - 여유분) / step당 시간 으로 최대 안전 step 수 재계산, 필요시 --epochs 조정
python src/finetune/train.py \
    --lora-r 32 --target-modules q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj \
    --lora-alpha 64 --epochs 4
```

- `EarlyStoppingCallback patience: 1 → 2` (epoch 수가 줄어든 만큼, eval loss가 1번만
  정체돼도 바로 멈추면 학습이 너무 짧게 끝날 위험이 있어 여유를 둔다 — `train.py`에서
  직접 코드 수정 필요, 현재는 CLI 플래그가 아님)

## 4. 평가 설계

```
python src/eval/baseline_eval.py --model tradecode-gemma2-v2 --prompt-style finetuned \
    --code-length 4 --output docs/07-experiment2_finetuned_result.md
```

`baseline_eval.py --code-length 4`(신규 플래그)로 재평가하면:

| 지표 | 실험 1(6자리 기준) | 실험 2(4자리 기준) |
|---|---|---|
| Exact Match | 6자리 완전일치 | 4자리 완전일치 |
| Partial Match | 4자리, 2자리 | 2자리만 |
| Top-3 Recall | 6자리 기준 | 4자리 기준 |
| Parse Failure Rate | 동일 정의 | 동일 정의 |

zero-shot baseline도 같은 eval.jsonl(4자리 기준)로 다시 측정해야 공정한 비교가
된다:

```
python src/eval/baseline_eval.py --eval-file data/processed/eval.jsonl \
    --model gemma2:2b --prompt-style zero_shot --code-length 4 \
    --output docs/06-experiment2_baseline_result.md
```

또한 `04-comparison.md`가 지적했듯 eval loss 개선이 실제 생성 능력 개선을
보장하지 않으므로, 학습 중간에도(예: `--limit 20`으로) 실제 생성 샘플을 눈으로
확인해 "숫자를 생성하는지" 자체를 정성적으로 모니터링한다.

## 5. 이번 문서 이후 예정 (미실행)

- `06-experiment2_baseline_result.md`: zero-shot baseline을 4자리 기준으로 재측정
- `07-experiment2_finetuned_result.md`: 재평가 결과
- `08-experiment2_comparison.md`: 실험 1 대비 비교 및 원인 분석

## 6. 리스크

- 원본 추가 수집이 계획만큼 안 되면(예: 사전회시 사례에서 얇은 헤딩 보강이 어려우면)
  2.1의 목표 raw 총량에 못 미칠 수 있다 — 이 경우 증강 비율(`--n-per-item`)을 더
  올려 총량은 맞추되, 클래스당 실제 정보량 부족 문제는 일부 남을 수 있음을 감안한다.
- epoch 4가 스모크 테스트 실측 결과 세션 예산에 비해 너무 적거나(조기 수렴 전
  세션 종료) 너무 많을 수 있다 — 3절의 실측 후 조정 절차를 반드시 거친다.
- 4자리로 낮춰도 정확도가 낮으면, 문제는 "클래스 희소성"이 아니라 "LoRA
  설정/completion 파싱/모델 용량" 등 다른 요인일 가능성을 열어둬야 한다.
