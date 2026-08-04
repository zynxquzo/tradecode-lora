# 실험 3 진행 기록: 데이터 다양화, 타깃 단순화, unsloth 병합 버그 발견

`docs/09-experiment3-plan.md`에서 세운 계획(데이터 다양화 + LoRA 용량 원복)을
실행하고, 그 과정에서 예상 못한 문제를 추적한 기록. 아직 최종 결론(실제 정량
평가)에는 도달하지 못했고, 다음 세션에서 이어서 진행해야 한다.

## 1. 데이터 다양화 (완료)

관세청이 공개한 HS부호 단위별 품목명 엑셀(`관세청_HS부호 단위별 품목명_20260101.xlsx`,
HS2/HS4/HS6 시트)을 참고 자료로 사용해 `src/finetune/extract_hs_reference.py`를
새로 작성했다. 이 스크립트는 특정 류(chapter)의 HS4(호)+HS6(소호) 품목명을
`products_real.csv`와 동일한 스키마(description/hs_code/confidence_basis)로 추출한다.

- 63류(기타 섬유제품) 14건, 64류(신발) 18건, 84류(기계) 55건, 85류(전기기기) 55건,
  총 142건을 새로 추출해 기존 `products_real.csv`(356건, 61/62류만)에 병합 →
  498건, 첫 자리 '6' 아닌 비중 22.1%로 확보.
- `augment.py`로 패러프레이징 증강(498건 → 1,992건) → `preprocess.py --code-length 4`로
  train/eval 재생성(train 1,587 / eval 405). 4자리 헤딩 클래스 115개, 최다 클래스도
  6.0%로 편중 해소 확인.

## 2. 1차 재학습: LoRA 용량 원복만으로는 재발 (완료, 실패)

`docs/09-experiment3-plan.md`대로 LoRA를 attention-only(r=16, alpha=32)로 원복해
재학습. 10 epoch(900 step) 끝까지 정상적으로 돌았고 조기 종료도 없었다. 최종
eval_loss **4.31** — 실험 1의 최종 eval_loss(4.09)와 거의 동일한 수준.

병합 모델을 transformers로 직접 로드해 생성해보니, 실험 2 때(숫자 "6" 반복)와는
다른 실패 양상이 나왔다 — 학습에 직접 쓰인 train 샘플을 넣어도 완전히 빈 문자열
또는 공백만 생성. `docs/03-finetuned_result.md`(실험 1)를 다시 보니 그때도
Parse Failure Rate 100%(예측 전부 `(none)`)였다는 걸 확인 — **즉 이번 실험 3의
"빈 출력"은 새 버그가 아니라 실험 1부터 있었던, 한 번도 해결 안 된 근본 문제가
재현된 것**이었다. 데이터 편중은 고쳤지만 LoRA 용량까지 같이 원복해버려서,
실험 1의 한계(모델이 스키마 형태는 익혀도 내용을 못 배움)를 다시 노출시킨 셈이다.

## 3. 2차 재학습: LoRA 용량 재확대(r=32+MLP) — 완전한 붕괴는 아니지만 여전히 실패

데이터는 이미 균형 잡혔으니 "지름길"(무조건 6 찍기) 자체가 없어졌다는 가정 하에,
LoRA를 다시 실험 2 수준(r=32, alpha=64, MLP 포함)으로 키워 재학습. 스모크
테스트(60 step)만으로 eval_loss가 4.63까지 떨어져 attention-only의 900 step
결과(4.31)를 근접하게 따라잡음 — MLP 레이어가 실제로 학습 신호를 훨씬 잘
흡수한다는 정황.

하지만 병합 모델 생성 테스트에서 여전히 문제 발견 — 이번엔 `)`만 반복하는
구두점 루프(repetition_penalty를 걸어도 `)` 위주 노이즈만 나옴). loss는 낮은데
생성은 무의미하다는 실험 1 패턴이 반복.

## 4. completion 단순화 실험 — 결정적 개선, 그러나 새로운 문제 노출

`confidence_basis`(류>호>소호 한글 설명, 괄호·기호가 매우 많은 문자열)가 학습
신호를 희석시켰다는 가설을 세우고, `preprocess.py`에 `--simple-target` 옵션을
추가했다(`docs/`가 아니라 코드에 반영, 커밋 `a3df237`). completion을
`{"hs_code": "6402"}`처럼 숫자만 담도록 단순화하고 instruction 문구에서도
"근거를 설명하세요" 부분을 제거.

같은 LoRA 설정(r=32+MLP)으로 재학습한 결과, eval_loss가 **0.89(epoch1) → 0.18(epoch10)**
까지 떨어짐 — 클래스 115개짜리 태스크에서 빈도수만으로 찍는 것으로는 나올 수 없는
수준(추정 baseline loss 4~5)이라, 실제로 입력을 보고 분류를 학습했다는 강한 정황.

**그런데 병합 모델로 생성 테스트를 하면 여전히 무의미한 출력이 나온다** — 이번엔
`purpoſe`, `vectorielle`, `myſelf`, `auroit`, `Portail` 같은 특정 희귀
단어(장음 s가 섞인 고어체, 프랑스어 단어)들이 입력과 거의 무관하게 반복 등장.
다만 완전히 무관하지는 않고, 의류/신발류(61~64류) 입력엔 `purpoſe`/`drawal`/`Portail`
계열이, 기계/전기류(84/85류) 입력엔 `vectorielle` 계열이 나오는 등 카테고리
수준의 구분은 남아있었다.

## 5. 근본 원인 확정: unsloth 병합(`save_pretrained_merged`) 자체의 버그

다음 단계로 여러 가설을 하나씩 제거했다:

1. **Gemma-2 logit softcapping 손상 의심** → `outputs/merged/config.json` 확인
   결과 `attn_logit_softcapping: 50.0`, `final_logit_softcapping: 30.0`으로 정상값.
   기각.
2. **prompt/completion 경계 토큰화 불일치 의심** → 프롬프트에
   `{"hs_code": "` 를 미리 열어서 힌트를 줘도 똑같이 공백 다음에 같은 희귀
   단어가 나옴. 기각.
3. **베이스 모델 자체(순정, 병합 전) 검증** → LoRA도 병합도 전혀 거치지 않은
   `unsloth/gemma-2-2b-bnb-4bit`를 순정 transformers로 로드해 동일 프롬프트로
   생성 → **정상적으로 숫자열(`42029000000000000000`) 생성**. 이걸로 확정:
   **문제는 학습 데이터도, LoRA 설정도 아니라 `merge_adapter.py`가 쓰는
   unsloth의 `FastLanguageModel.save_pretrained_merged(save_method="merged_16bit")`
   가 병합 과정에서 가중치를 손상시키고 있다.**

`requirements-colab.txt`는 unsloth 버전을 의도적으로 고정하지 않았는데
("최신 버전이 trl 0.24.0과 알아서 호환될 것"이라는 가정, 파일 상단 주석 참고),
이번에 설치된 버전이 "Unsloth 2026.8.1"이었다 — 이 최신 버전에 Gemma-2 병합
관련 회귀가 있을 가능성이 유력하다.

## 6. 우회책: 순정 peft 병합 스크립트 추가 (검증 대기 중)

`merge_adapter.py`의 기존 docstring에는 "순정 peft.PeftModel.merge_and_unload()는
이미 시도했다가 실패했다"는 기록이 있었지만, 그건 학습에 실제로 쓰인 4bit 베이스
(`unsloth/gemma-2-2b-bnb-4bit`)가 아니라 별도의 풀 정밀도 베이스
(`unsloth/gemma-2-2b`)에 병합해서 아키텍처 불일치 가능성이 있었던 시도였다.

이번엔 학습에 실제로 쓰인 것과 동일한 4bit 베이스에, 순정 peft로 병합하는
`src/finetune/merge_adapter_plain.py`를 새로 작성했다(커밋 `67bb172`). 다음
세션에서 이 스크립트로 병합한 모델이 정상 생성을 하는지 확인하는 것부터
시작해야 한다.

## 다음 단계

1. Kaggle 세션이 끊겨 `outputs/adapter`가 유실됨 — 재학습부터 다시 시작해야 함
   (데이터: `data/processed_simple/{train,eval}.jsonl`, 이미 로컬에 있고 Kaggle에도
   `simple-data`로 업로드돼 있음. LoRA 설정: r=32, alpha=64, MLP 포함, 그대로).
2. 병합은 `merge_adapter.py`(unsloth) 말고 **`merge_adapter_plain.py`(순정 peft)**로
   바로 진행.
3. train/eval 샘플 생성 테스트로 정상 출력 나오는지 확인.
4. 정상이면 GGUF 변환(`src/serving/build_ollama_model.sh`) → Ollama 등록 →
   `eval.jsonl` 전체(405건) 정량 재평가 → `docs/11-experiment3_*_result.md`로 기록.
5. 만약 `merge_adapter_plain.py`도 같은 증상이면, unsloth 버전을 낮춰서
   (`requirements-colab.txt`에 특정 버전 고정) `merge_adapter.py`(unsloth 경로)를
   다시 시도해볼 것 — 어느 버전부터 회귀가 생겼는지는 아직 특정 못 함.
