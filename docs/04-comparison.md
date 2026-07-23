# Baseline vs Fine-tuned 비교

| 지표 | Baseline (zero-shot) | Fine-tuned (LoRA) | 개선폭 |
|---|---|---|---|
| Exact Match (6자리) | 0.36% | 0.00% | -0.36%p |
| Partial Match (4자리) | 3.57% | 0.00% | -3.57%p |
| Partial Match (2자리) | 38.93% | 0.00% | -38.93%p |
| Top-3 Recall | 0.71% | 0.00% | -0.71%p |
| Parse Failure Rate | (baseline 리포트엔 없음, 샘플상 낮음) | 100.00% | 악화 |

(fine-tuned: `docs/03-finetuned_result.md`, 280건 전체, `tradecode-gemma2` Q4_K_M, 2026-07-23)

## 무슨 일이 있었나

880건/LoRA r=16/3 epoch 기본 설정으로 처음 학습했을 때 eval loss가 `ln(vocab_size)`보다도
높은 값(24대)에서 시작해 3 epoch 내내 8~9대에 머물렀다. 원인은 프롬프트 전체
(Instruction+Input+Response)에 대해 loss를 계산해서, 매번 거의 동일한 Instruction/Input
부분(예측하기 쉬움)이 실제로 배워야 할 JSON 응답의 학습 신호를 희석시킨 것이었다.
`SFTConfig(completion_only_loss=True)`로 completion(JSON 응답)에만 loss를 매기고
epoch을 10으로 늘려 재학습하자 eval loss가 8.88 → 4.09까지 꾸준히 개선됐다
(perplexity 환산 시 약 6500 → 약 60, 파인튜닝이 실제로 뭔가를 학습했다는 뚜렷한 신호).

하지만 실제 생성 결과를 보면, 모델은 `{"hs_code": ..., "confidence_basis": ...}` 스키마의
**형태**(특히 "code"라는 단어, 한국어 confidence_basis에 쓰이는 "편물/직물" 계열 어휘)는
어느 정도 익혔지만, **6자리 숫자 자체를 생성하는 법은 거의 배우지 못했다**. 예시:

```
입력: Men's or boys' overcoats, car-coats, capes, cloaks
출력: "Code of the code in Code\ncode"
```

이 때문에 `extract_hs_codes`가 숫자를 하나도 못 뽑아내 Parse Failure Rate가 **100%**로,
모든 분류 지표가 0%로 나왔다 — baseline(무작위에 가깝지만 그래도 숫자는 냈던 상태)보다
표면적으로는 더 나빠 보이는 결과다.

## 해석

- **loss가 크게 개선된 것과 실제 태스크 성능이 개선된 것은 다른 문제였다.** loss는
  JSON 스키마의 토큰 시퀀스(키 이름, 중괄호, 흔한 단어)를 얼마나 잘 예측하는지에
  많이 좌우되는데, 이건 상대적으로 배우기 쉽다. 반면 "이 상품설명 → 이 6자리 숫자"라는
  매핑은 사실상 210개 클래스에 대한 암기에 가까운 태스크라 정보량이 훨씬 크고,
  같은 loss 개선폭이라도 이 부분까지 학습됐다는 보장이 없다.
- **880건/210클래스 = 클래스당 평균 4건**은 이 정도 규모의 세분화 분류를 2B급 모델이
  숫자 단위까지 암기하기엔 근본적으로 부족한 데이터 규모였을 가능성이 높다. baseline
  해석(`docs/01-baseline_result.md`)에서 이미 "사람도 관세율표 문구만 보고 6자리를
  암기해서 맞추기 어려운 태스크"라고 짚었던 우려가, 파인튜닝을 거치고도 그대로
  드러난 셈이다.
- **LoRA를 attention projection(q/k/v/o_proj)에만 적용**했다는 점도 한계일 수 있다.
  숫자 암기처럼 사실적 지식을 새로 주입하는 작업은 MLP(gate/up/down_proj) 쪽 용량이
  더 중요하다는 보고가 많다 — 다음 시도에서는 target_modules를 넓히거나 LoRA rank를
  올리는 게 우선순위가 될 것이다.
- **Top-3가 아니라 단일 예측으로 수렴한 것은 설계상 의도된 결과다** — 학습 데이터가
  레코드당 정답 1개뿐이라 fine-tuned 모델은 애초에 top-3 array가 아닌 단일 JSON
  객체를 생성하도록 학습했다(`--prompt-style finetuned`). 다만 이번 결과에서는 그
  단일 예측조차 숫자를 담지 못했다.
- **Parse Failure Rate가 baseline보다 악화된 것**은 "포맷은 배웠지만 내용을 못
  배운" 상태를 정확히 보여주는 지표라, 오히려 실패 양상을 명확히 설명하는 근거로
  쓸 수 있다.

## 다음에 시도해볼 것 (시간이 더 있다면)

1. LoRA target_modules에 `gate_proj, up_proj, down_proj` 추가
2. LoRA rank를 16 → 32 이상으로 확대
3. 데이터 증강 비율을 늘리거나(현재 augment.py), 클래스당 최소 샘플 수 보장
4. epoch을 더 늘리되(현재 10에서 eval loss가 아직 완만하게 개선 중이었음) 과적합
   여부를 eval loss뿐 아니라 실제 숫자 생성 성공률로도 모니터링

## 포트폴리오 서사로서의 정리

이번 프로젝트는 "파인튜닝으로 정확도를 올렸다"는 결과 대신, **왜 초기 시도가 실패했고
어떻게 원인을 좁혀갔는지**(entropy_from_logits 호환성, SFTConfig 클래스 identity
불일치, dataset.map() 멀티프로세싱 pickling, 그리고 결정적으로 completion-only loss
masking 도입까지) 자체가 더 뚜렷한 학습 곡선을 보여준다. loss 개선(perplexity
~6500→~60)과 실제 태스크 성능 미개선(정확도 0%) 사이의 간극은, "loss가 떨어진다고
과제를 잘 푸는 건 아니다"라는 실무적으로 중요한 교훈을 정량적으로 보여주는 사례로
정리할 수 있다.
