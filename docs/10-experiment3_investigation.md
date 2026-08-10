# 실험 3 진행 기록: 데이터 다양화, 타깃 단순화, unsloth 제거, 그리고 97.53% 달성

`docs/09-experiment3-plan.md`에서 세운 계획(데이터 다양화 + LoRA 용량 원복)을
실행하고, 그 과정에서 예상 못한 문제를 추적한 기록. **5·6절의 "unsloth 병합
자체의 버그"라는 결론은 재학습 후 재검증 과정에서 틀린 것으로 확인됐다 — 실제
원인은 7절 참고.** 최종적으로 `eval.jsonl` 405건 기준 **Exact Match 97.53%**를
달성했다(13절).

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

## 5. (틀린 결론이었음, 7절에서 정정) 당시 추정: unsloth 병합(`save_pretrained_merged`) 자체의 버그

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

## 6. (당시 계획) 우회책: 순정 peft 병합 스크립트 추가 (검증 대기 중)

`merge_adapter.py`의 기존 docstring에는 "순정 peft.PeftModel.merge_and_unload()는
이미 시도했다가 실패했다"는 기록이 있었지만, 그건 학습에 실제로 쓰인 4bit 베이스
(`unsloth/gemma-2-2b-bnb-4bit`)가 아니라 별도의 풀 정밀도 베이스
(`unsloth/gemma-2-2b`)에 병합해서 아키텍처 불일치 가능성이 있었던 시도였다.

이번엔 학습에 실제로 쓰인 것과 동일한 4bit 베이스에, 순정 peft로 병합하는
`src/finetune/merge_adapter_plain.py`를 새로 작성했다(커밋 `67bb172`). 다음
세션에서 이 스크립트로 병합한 모델이 정상 생성을 하는지 확인하는 것부터
시작해야 한다.

## 7. 재학습 및 재검증: "병합 버그"가 아니라 순정 transformers의 Gemma-2 회귀였다

Kaggle 세션이 끊겨 `outputs/adapter`가 유실돼 재학습부터 다시 시작했다(데이터는
`data/processed_simple/{train,eval}.jsonl` 그대로, LoRA는 r=32/alpha=64/MLP 포함
동일 설정). 새 클론에는 `pip install -r requirements-colab.txt`가 unsloth
**2026.8.2**를 설치했다(직전 세션의 2026.8.1보다도 최신).

### 7.1 병합은 성공, 그런데 순정 transformers로 다시 실패

`merge_adapter_plain.py`(6절에서 작성한 순정 peft 병합 스크립트)를 그대로 쓰려니
새로운 문제가 나왔다 — 학습 베이스(`unsloth/gemma-2-2b-bnb-4bit`, 4bit)에 그대로
병합하면 peft가 결과를 다시 4bit로 재양자화하는데(`Linear4bit.merge()`), 이
상태를 최신 transformers(5.5.0)의 `save_pretrained()` 내부 `revert_weight_conversion()`
로직이 처리하지 못해 `NotImplementedError`로 죽었다. 해결책: 학습 베이스가 아니라
같은 모델의 **fp16(비양자화) 버전**(`unsloth/gemma-2-2b`)에 병합하도록
`merge_adapter_plain.py`를 수정(`--full-precision-base` 옵션 추가, 커밋 `8a09ea8`).
LoRA 가중치는 dtype에 무관하게 모듈 이름/shape만 맞으면 병합되므로 문제없이 동작.

병합 자체는 성공했지만, 병합된 모델을 순정 transformers로 생성 테스트하니 **5절과
완전히 동일한 증상**(`purpoſe`/`vectorielle` 등 특정 희귀 단어 반복, 카테고리별로
다른 단어군)이 재현됐다.

### 7.2 결정적 반증: 병합을 아예 안 해도 똑같이 실패한다

병합 방식이 원인이라는 가설을 검증하기 위해, **병합을 하지 않고** fp16 베이스에
`PeftModel.from_pretrained()`로 adapter만 얹어서(merge_and_unload 호출 없이)
생성 테스트를 했다 — 결과는 병합했을 때와 **바이트 단위로 동일**했다. 이것으로
"병합 로직이 가중치를 손상시킨다"는 5절의 가설은 완전히 기각된다. 병합을 거치지
않은 순수 adapter+base 조합조차 같은 증상을 보이므로, 문제는 병합이 아니라 다른
곳에 있다.

추가로 두 가설을 더 검증했다:
- **teacher-forcing 검증**: 실제 정답 completion을 입력에 그대로 넣고 한 번의
  forward pass로 각 위치의 다음 토큰 예측을 확인 → 토큰 단위 일치율 **0%**,
  여전히 같은 희귀 단어 생성. eval_loss 0.18이 실제로 이 위치들에서 나온
  값이라면 나올 수 없는 결과.
- **attention 구현 차이 의심**(Gemma-2의 `attn_logit_softcapping`이 `sdpa` 등에서
  조용히 무시될 수 있음) → `attn_implementation="eager"`로 명시해 재시도해도
  동일하게 실패. 기각.

### 7.3 근본 원인 확정: unsloth 학습 결과물은 unsloth로만 정확히 재현된다

`adapter_config.json`(r=32, alpha=64, MLP 포함 target_modules 확인, `"unsloth_fixed":
true` 필드 존재)과 `docs/02-training_log.md`(step 630→900 구간에서 eval_loss
0.23→0.20→0.19→0.18로 매끄럽게 수렴, 우연이 아닌 진짜 수렴 곡선)를 재확인해
학습 설정 자체는 의도한 그대로였음을 확인했다.

마지막으로, **unsloth 자체의 로드/추론 경로**(`FastLanguageModel.from_pretrained("outputs/adapter")`
+ `FastLanguageModel.for_inference(model)`)로 같은 adapter를 불러와 같은 프롬프트로
생성해보니 — **정상 동작**했다. `6402→6404`, `6105→6103`처럼 카테고리가 맞는
근사값이 나왔고 `6110→6110`은 정확히 일치했다. 형식도 전부 올바른 JSON.

**결론**: 학습은 처음부터 제대로 됐다(eval_loss 0.18은 진짜였다). 실패의 원인은
"unsloth 병합 버그"가 아니라, **unsloth가 학습에 쓰는 Gemma-2 내부 구현(임베딩
스케일링/logit softcapping 등 처리 방식)이 순정 transformers(이번 세션 기준
5.5.0)의 Gemma-2 구현과 수치적으로 다르다**는 것이다. unsloth로 저장한 가중치를
순정 transformers/peft로 그대로 불러와 `model.generate()`를 돌리면, 병합 여부나
병합 방법과 무관하게 항상 이 불일치가 재현된다. `merge_adapter.py`(unsloth 병합)와
`merge_adapter_plain.py`(순정 peft 병합) 둘 다 5절 시점엔 "실패"로 보였지만,
실제로는 둘 다 가중치 자체는 올바르게 저장했을 가능성이 높고, 그걸 검증하는 데
썼던 **순정 transformers 생성 테스트 방법 자체가 신뢰할 수 없었다**.

## 8. (틀린 가설이었음) "llama.cpp는 이 불일치에서 자유로울 것"

이 프로젝트의 실제 서빙 경로는 순정 transformers가 아니라 GGUF 변환 →
llama.cpp → Ollama이므로, llama.cpp가 Gemma-2를 독립적으로 구현해 이 불일치의
영향을 안 받을 것이라는 가설을 세웠었다. **9절에서 검증한 결과 이 가설은
틀렸다.**

## 9. 6가지 조합 전수 테스트 — unsloth 밖에서는 무엇을 해도 실패한다

재학습된 adapter를 놓고, 저장/병합/서빙 방식을 바꿔가며 가능한 조합을 전부
테스트했다:

| # | 병합 방법 | 검증 방법 | 결과 |
|---|---|---|---|
| 1 | 없음 (adapter만 부착) | 순정 transformers | 실패 |
| 2 | 순정 peft 병합 (별도 fp16 베이스) | 순정 transformers | 실패 |
| 3 | 순정 peft 병합 (학습 때 그 4bit 베이스를 직접 역양자화 후 병합) | 순정 transformers | 실패 |
| 4 | 순정 peft 병합 (#3과 동일) | GGUF 변환 → Ollama | 실패 |
| 5 | unsloth 공식 병합(`save_pretrained_merged`) | GGUF 변환 → Ollama | 실패 |
| 6 | 병합 없음, LoRA를 `convert_lora_to_gguf.py`로 GGUF 어댑터화 후 llama.cpp가 추론 시점에 직접 적용 | Ollama (`ADAPTER` 지시자) | 실패 |
| 7 | 없음 (adapter만 부착) | **unsloth 네이티브**(`FastLanguageModel.for_inference`) | **성공** |

#3(순정 peft 역양자화 병합) 과정에서 부수적으로 발견/수정한 버그들(모두
`merge_adapter_plain.py`에 반영, 커밋 `900e439`~`95ce2cc`):
- `state_dict()`를 거치면 `bnb.nn.Params4bit`의 `quant_state`가 유실되고 4bit로
  패킹된 원본이 그대로 나온다(shape가 찌그러짐) → 살아있는 모듈 객체
  (`module.weight`)에서 직접 읽어야 함.
- Gemma-2는 `lm_head`가 `embed_tokens`와 가중치를 공유하는데, `named_parameters()`
  가 중복 텐서 중 하나를 건너뛰어 키가 하나 빠짐 → 수동으로 채워 넣음.
- `config.quantization_config = None`으로만 지워도 속성 자체는 남아
  `config.json`에 `"quantization_config": null`로 직렬화되고, 나중에 그 모델을
  다시 불러올 때 transformers가 "사전 양자화된 모델"로 오인해 `AttributeError`로
  죽음 → `del config.quantization_config`로 속성 자체를 제거해야 함.

이 버그들을 다 고쳐서 **가중치 자체는 정확하게 역양자화·병합·저장됐다고 볼 수
있는 상태**로 만들었는데도(#3), 여전히 실패했고, unsloth 공식 병합(#5)도
GGUF까지 가면 실패했고, 심지어 **병합을 아예 거치지 않고 LoRA 델타를
llama.cpp가 직접 계산**하게 한 #6도 실패했다. 마지막 #6은 결정적이다 — 이 경우
llama.cpp가 베이스 forward와 LoRA 델타 적용을 전부 자기 방식대로 계산하므로,
저장/병합/역양자화 로직의 버그가 전혀 개입할 여지가 없다. 그런데도 실패했다는
것은, **문제가 가중치나 저장 방식이 아니라 "이 LoRA 가중치는 unsloth가 학습 중
사용한 Gemma-2 forward 계산(예: 속도를 위해 근사되거나 생략된 logit
softcapping/임베딩 스케일링 등)에 맞춰 최적화됐고, 순정 구현(transformers든
llama.cpp든)의 계산 결과와는 안 맞는다"**는 뜻이다. 7절의 결론이 최종 확정됐다.

## 10. 결정: unsloth 없이 순정 transformers/peft로 재학습

병합·서빙 쪽에서 고칠 수 있는 문제가 아니므로, **학습 자체를 unsloth 없이 순정
transformers + peft(QLoRA) + trl로 바꿔서 학습과 추론이 처음부터 같은 계산
경로를 쓰게 만들기로 결정**했다(커밋으로 반영 예정: `src/finetune/train.py`
전면 재작성, `requirements-colab.txt`에서 unsloth 제거, 기본 베이스 모델을
unsloth의 사전 양자화 체크포인트(`unsloth/gemma-2-2b-bnb-4bit`)에서 순정
`google/gemma-2-2b`로 변경 — `BitsAndBytesConfig`로 학습 시점에 4bit 동적
양자화).

트레이드오프: unsloth의 Triton 커널 최적화가 없어 학습 속도가 느려진다(정확한
배수는 미측정, GPU 기준으로도 체감 가능한 수준일 것으로 예상). 대신 병합 후
정상 동작이 보장된다는 이득이 훨씬 크다 — 지금까지 7번의 조합 테스트로 이미
그만큼의 시간을 썼다.

`merge_adapter.py`(unsloth 경로)는 더 이상 학습 파이프라인에서 쓰지 않지만
과거 조사 기록으로서 파일은 남겨둔다. 새 파이프라인은 `merge_adapter_plain.py`
하나로 통일된다 — QLoRA로 동적 양자화한 표준 베이스에 병합하는 것은 peft의
공식 지원 워크플로우라, 지금까지 겪은 문제들(예: 사전 양자화 체크포인트 특유의
`Params4bit` 처리) 없이 정상 동작할 것으로 기대된다.

## 11. 재학습 성공 — 순정 경로에서도 정상 생성 확인

unsloth 제거 후 재학습하면서 두 가지 인프라 문제를 추가로 겪었다 (둘 다 Kaggle의
GPU 2장(T4 x2) 환경 특유의 문제, `train.py`에 반영):
- `google/gemma-2-2b`는 라이선스 동의가 필요한 gated 저장소라 인증 없이 401로
  막힘 → unsloth가 올려둔 ungated 미러 `unsloth/gemma-2-2b`(fp16, 가중치
  호스팅만 unsloth 계정일 뿐 unsloth의 학습 코드 패치와는 무관함)로 기본값 변경.
- `device_map="auto"`가 모델을 GPU 2장에 나눠 올리자 Trainer가 자동으로
  `nn.DataParallel`을 씌우며 `CUBLAS_STATUS_EXECUTION_FAILED`로 충돌 →
  `CUDA_VISIBLE_DEVICES=0`을 CUDA 초기화 전에 강제해 GPU 1장만 보이게 고정.

스모크 테스트(60 step)에서 loss 2.22 → 0.06, `mean_token_accuracy` 0.43 → 0.97+
로 정상 수렴 확인 후 본 학습 진행, `merge_adapter_plain.py`로 병합, 순정
transformers `AutoModelForCausalLM.from_pretrained()` + `generate()`로 생성
테스트 — **10건 중 9건 정확히 일치, 1건은 근접 오류(`6101`→`6110`, 같은 61류
내 혼동)**, 형식도 전부 올바른 JSON. unsloth를 제거하고 학습·추론이 같은 계산
경로(순정 transformers)를 쓰게 만든 것으로 실험 3 전체를 관통한 문제가
해결됐다.

## 12. GGUF/Ollama 배포 직전에 발견한 마지막 버그: 프롬프트 템플릿 이중 적용

`outputs/merged`를 GGUF로 변환해 Ollama에 등록하고 `ollama run`으로 생성 테스트를
해보니 `{"` 만 반복하는 새로운 증상이 나왔다 — 11절까지의 성공(순정 transformers
`generate()`)과 모순되는 것처럼 보였지만, 원인은 가중치가 아니라 **Ollama가
`/api/generate` 호출 시 프롬프트에 Gemma의 채팅 템플릿(`<start_of_turn>...`)을
자동으로 씌운다는 것**이었다. 이 모델은 채팅 템플릿이 아니라 `train.py`의 순수
Alpaca 스타일 프롬프트(`### Instruction:\n...\n### Response:\n`)로 학습됐으므로,
템플릿이 덧씌워지면 학습 때와 다른 입력이 들어가 무너진다.

`src/eval/baseline_eval.py`의 `call_ollama()`도 `raw` 옵션을 안 넘겨서 같은
문제를 그대로 갖고 있었다 — **재평가 스크립트 자체가 고쳐지지 않았다면 정량
평가 결과 전체가 이 버그로 오염됐을 것**이다. `predict()`에서 `prompt_style ==
"finetuned"`일 때만 `raw=True`를 넘기도록 고쳤다(zero_shot은 베이스 모델
평가용이라 채팅 템플릿이 적용되는 게 정상 사용법이라 그대로 둠). 고친 뒤
curl 수준 테스트로 정상 출력 확인.

## 13. 최종 결과: `eval.jsonl` 전체(405건) 정량 평가

`docs/11-experiment3_finetuned_result.md`에 원본 리포트 기록. 요약:

| 지표 | 점수 |
|---|---|
| Exact Match (4자리) | **97.53%** |
| Partial Match (2자리) | 100.00% |
| Parse Failure Rate | 0.00% |

실험 1(baseline/fine-tuned 둘 다 0%대)과 실험 2(mode collapse)를 거쳐, 실험
3에서 데이터 다양화(1절) → completion 단순화(4절) → unsloth 제거(10절) →
프롬프트 템플릿 이중 적용 수정(12절)까지 거친 끝에 나온 결과. **실험 3, 그리고
이 프로젝트 전체의 최종 목표(HS코드 분류 LoRA 파인튜닝)가 실질적으로 달성됐다.**

## 다음 단계

실험 3의 핵심 목표는 달성됐다. 남은 건 다듬는 작업 수준:

1. `docs/12-*` 또는 README에 실험 3 최종 결과를 요약 반영 (baseline vs
   fine-tuned 비교 표 등).
2. 필요하면 misclassified 10건(2.47%)을 살펴봐 패턴이 있는지 확인 — 지금
   수준(97.53%)에서는 우선순위 낮음.
3. `merge_adapter.py`(unsloth 경로)는 이제 활성 파이프라인에서 완전히 안 쓰이니,
   과거 조사 기록 가치만 남기고 README/문서에서 "쓰지 않음"으로 명확히 표시된
   상태를 유지.
5. 혹시 이번에도 실패하면(가능성은 낮지만), 그때는 정말로 데이터/학습
   하이퍼파라미터 쪽 문제일 가능성이 높다 — 프레임워크 조합은 이걸로 완전히
   소거됐기 때문이다.
