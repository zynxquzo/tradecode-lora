# Baseline vs Fine-tuned 비교 (플레이스홀더)

`docs/03-finetuned_result.md`가 실제 값으로 채워진 뒤 이 표와 해석을 갱신한다.

| 지표 | Baseline (zero-shot) | Fine-tuned (LoRA) | 개선폭 |
|---|---|---|---|
| Exact Match (6자리) | 0.36% | ? | ? |
| Partial Match (4자리) | 3.57% | ? | ? |
| Partial Match (2자리) | 38.93% | ? | ? |
| Top-3 Recall | 0.71% | ? | ? |
| Parse Failure Rate | (baseline 리포트에 없음, 재계산 필요) | ? | ? |

## 해석 (재평가 후 작성)

- 6자리/4자리 exact match 개선폭이 핵심 서사: zero-shot은 사실상 무작위 수준(0.36%)
  이었는데, 880건 LoRA 파인튜닝만으로 얼마나 올라가는지가 "직접 파인튜닝 역량"을
  증명하는 포인트.
- Top-3 Recall이 Exact Match와 거의 같은 값으로 나올 가능성이 높다 — 학습 데이터가
  레코드당 정답 1개뿐이라 fine-tuned 모델은 top-3 array가 아니라 단일 JSON 객체를
  생성하도록 학습했기 때문 (`--prompt-style finetuned`, `extract_hs_codes`가 단일
  객체에서 코드 1개만 추출). 이 자체가 "왜 Top-3가 아니라 Top-1로 수렴했는가"라는
  질문에 대한 근거 있는 설명이 된다.
- Parse Failure Rate가 baseline 대비 크게 줄었다면(포맷 준수 개선), 그 자체로도
  파인튜닝의 성과로 강조할 수 있다 — baseline 해석(`docs/01-baseline_result.md`)에서
  지적한 "출력 포맷 준수 자체가 불안정" 문제의 해소 여부.
- 880건 규모 학습 데이터와 210개 세분화 클래스라는 조합을 고려하면, 4자리(호) 수준
  개선이 6자리(완전일치) 개선보다 더 안정적으로 나타날 가능성이 높다 — 결과가 그렇게
  나오면 "세부 소호까지는 데이터가 더 필요하다"는 남은 한계로 기록.
- 남은 한계점 (실행 후 실측치로 구체화): 클래스 불균형, eval 샘플 수(280건)의 신뢰
  구간, 소규모 모델(2B)의 한계 등.
