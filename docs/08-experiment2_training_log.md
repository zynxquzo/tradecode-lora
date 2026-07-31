# 파인튜닝 학습 로그 (실험 2)

Kaggle Notebook(GPU T4x2)에서 `src/finetune/train.py`로 학습했다. 실험 1과 달리
4자리 heading을 타깃으로 하고, LoRA 설정을 더 공격적으로 키웠다(아래 참고).
학습 도중 실시간으로 쌓인 원본 로그는 Kaggle에서 다운로드한 파일
(`Downloads/02-training_log (1).md`)에서 그대로 옮겨왔다.

## 학습 설정

- base_model: unsloth/gemma-2-2b-bnb-4bit
- lora_r: 32, lora_alpha: 64 (실험 1: 16/32)
- target_modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  (실험 1: q/k/v/o만, MLP 레이어 미포함)
- learning_rate: 2e-4, epochs: 4 (실험 1: 10)
- batch_size: 4, grad_accumulation: 4 (유효 배치 16)
- train_records: 1795, val_records: 199 (실험 1: 792/88)

## Loss

| step/epoch | 구분 | loss | elapsed_min |
|---|---|---|---|
| step 10 | train | 16.8068 | 2.5 |
| step 20 | train | 8.0072 | 3.2 |
| step 30 | train | 5.3167 | 3.8 |
| step 40 | train | 4.3705 | 4.5 |
| step 50 | train | 3.6923 | 5.2 |
| step 60 | train | 3.2153 | 5.8 |
| step 70 | train | 2.9952 | 6.4 |
| step 80 | train | 2.6738 | 7.1 |
| step 90 | train | 2.5279 | 7.7 |
| step 100 | train | 2.3046 | 8.4 |
| step 110 | train | 2.2860 | 9.0 |
| step 113 | eval | 2.1516 | 9.8 |
| step 120 | train | 2.1141 | 10.3 |
| step 130 | train | 2.0041 | 10.9 |
| step 140 | train | 1.8848 | 11.6 |
| step 150 | train | 1.7736 | 12.2 |
| step 160 | train | 1.6986 | 12.9 |
| step 170 | train | 1.7066 | 13.5 |
| step 180 | train | 1.5658 | 14.2 |
| step 190 | train | 1.5274 | 14.9 |
| step 200 | train | 1.4883 | 15.5 |
| step 210 | train | 1.4057 | 16.2 |
| step 220 | train | 1.4312 | 16.8 |
| step 226 | eval | 1.3795 | 17.5 |
| step 230 | train | 1.3706 | 17.8 |
| step 240 | train | 1.3520 | 18.4 |
| step 250 | train | 1.2737 | 19.1 |
| step 260 | train | 1.3272 | 19.8 |
| step 270 | train | 1.2132 | 20.4 |
| step 280 | train | 1.2288 | 21.1 |
| step 290 | train | 1.1681 | 21.7 |
| step 300 | train | 1.1715 | 22.4 |
| step 310 | train | 1.1049 | 23.0 |
| step 320 | train | 1.1022 | 23.6 |
| step 330 | train | 1.1461 | 24.3 |
| step 339 | eval | 1.1406 | 25.2 |
| step 340 | train | 1.1408 | 25.2 |
| step 350 | train | 1.0444 | 25.9 |
| step 360 | train | 1.1107 | 26.5 |
| step 370 | train | 1.0769 | 27.2 |
| step 380 | train | 1.0812 | 27.9 |
| step 390 | train | 1.0558 | 28.6 |
| step 400 | train | 1.0242 | 29.2 |
| step 410 | train | 0.9494 | 29.9 |
| step 420 | train | 1.0189 | 30.5 |
| step 430 | train | 0.9372 | 31.2 |
| step 440 | train | 0.9555 | 31.8 |
| step 450 | train | 0.9695 | 32.5 |
| step 452 | eval | 0.9916 | 32.9 |

## 재평가 결과와 붕괴(mode collapse) 정황

- 최종 eval loss가 **0.99**까지 떨어졌다 (실험 1은 4.09). perplexity로 환산하면 약
  2.7 수준으로, 다양한 자연어(JSON 구조 + 설명 문구 + 숫자)를 생성해야 하는
  태스크치고는 비정상적으로 낮다 — "잘 배웠다"보다는 지름길로 loss를 낮췄다는
  신호에 가깝다.
- 실제로 GGUF 변환/Ollama 등록 후 재평가(`docs/07-experiment2_finetuned_result.md`)
  해보면 baseline보다도 전 지표가 나쁘고(Exact 0.40% vs baseline 2.61%), 원본
  병합 모델(GGUF 변환 전, transformers로 직접 로드)로 같은 프롬프트를 넣어봐도
  `"6 6 6 6 ... 6"`처럼 숫자 "6" 하나만 반복해서 뱉는 완전한 mode collapse가
  확인됐다. GGUF 변환/양자화 문제가 아니라 학습 자체의 문제라는 뜻이다.
- 원인 가설: 학습 데이터(`data/processed/train.jsonl`, 1994건)의 `hs_code` 첫
  자리가 **100% "6"**이다 (전 품목이 의류 관련 61/62류라서 원래 그런 특성이고,
  4자리로 줄이면서 고유 클래스도 34개뿐으로 줄었다). 실험 1도 데이터 특성 자체는
  같았지만 붕괴까지 가지 않았던 것과 비교하면, 이번에 새로 추가한 요인들
  - LoRA 용량 확대(r 16→32, alpha 32→64)
  - MLP 레이어(gate/up/down_proj)까지 학습 대상에 포함
  - 클래스 수 감소(4자리 타깃 → 34개)

  가 겹치면서 모델이 "그냥 6을 반복하면 loss가 잘 떨어진다"는 얕은 지름길에
  완전히 최적화되어 버린 것으로 보인다. eval loss만 보면 실험 1보다 훨씬
  개선된 것처럼 보이지만, 실제로는 더 나쁜 결과로 이어진 사례 — loss 개선이
  태스크 성능 개선을 보장하지 않는다는 `PERSONAL_NOTES.md`의 교훈이 더 극단적인
  형태로 재현된 것이다.
