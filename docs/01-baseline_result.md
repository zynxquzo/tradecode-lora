# 베이스라인 평가 결과

- 모델: `gemma2:2b` (Ollama 로컬 서빙, zero-shot)
- 평가 샘플 수: 280건 (실데이터, 원본 상품설명 기준)
- 생성 시각: 2026-07-20T23:18:16+09:00

## 지표

| 지표 | 점수 |
|---|---|
| Exact Match (6자리 완전일치) | 0.36% |
| Partial Match (4자리 일치) | 3.57% |
| Partial Match (2자리 일치) | 38.93% |
| Top-3 Recall | 0.71% |

## 해석

- **2자리(류/Chapter) 수준**에서는 약 39%를 맞추지만, **4자리·6자리로 갈수록 거의 0%**에 수렴합니다. 즉 gemma2:2b는 "이 상품이 의류(제61류·제62류 등)에 속한다" 정도의 대분류 감은 어느 정도 잡지만, 소재·형태별 세부 소호(6자리)까지는 zero-shot 상태로는 전혀 구분하지 못하고 있습니다.
- **Top-3 Recall(0.71%)도 Exact Match(0.36%)와 거의 차이가 없습니다.** 후보를 3개까지 넓혀줘도 정답을 못 맞춘다는 뜻으로, 모델이 정답 근처에도 가지 못하고 사실상 무작위에 가깝게 답하고 있다는 신호입니다.
- 샘플 예측 결과를 보면 **출력 포맷 준수 자체도 불안정**합니다. 존재하지 않는 4자리 코드("6206")만 내놓거나, 8자리 확장코드("62091000")를 섞어 쓰거나, 두 코드를 이어 붙인 값("620510620590")을 내놓거나, 아예 코드 추출에 실패("(none)")하는 경우가 다수입니다. 이는 순수한 분류 성능 문제 외에 프롬프트 지시(6자리 JSON 응답) 자체를 안정적으로 따르지 못하는 문제도 겹쳐 있음을 시사합니다.
- 이번 평가는 **210개의 세분화된 HS코드 클래스 + 2B급 소형 모델**이라는, 난이도가 높은 조합에서 나온 전형적인 저성능 베이스라인입니다. 사람도 관세율표 문구만 보고 6자리를 암기해서 맞추기 어려운 수준의 태스크이므로, 이 결과 자체는 예상 범위 내에 있습니다.
- 결론적으로 이 베이스라인은 **LoRA 파인튜닝의 필요성을 명확히 뒷받침**합니다. 파인튜닝 이후에는 최소한 4자리(호) 수준의 Partial Match와 Top-3 Recall이 크게 개선되는지를 핵심 성공 기준으로 삼는 것이 합리적입니다.

## 샘플 예측 결과

| 입력(상품설명) | 정답 | 예측 | Exact | Top-3 |
|---|---|---|---|---|
| Men's or boys' overcoats, car-coats, capes, cloaks | 620120 | (없음) | ❌ | ❌ |
| Other made up clothing accessories; parts of garme | 621710 | 620910, 630410 | ❌ | ❌ |
| Garments, made up of fabrics of heading 56.02, 56. | 621010 | 610610, 620410, 620810 | ❌ | ❌ |
| Women's or girls' overcoats, car-coats, capes, clo | 620290 | 621590, 621700, 620810 | ❌ | ❌ |
| Women's or girls' suits, ensembles, jackets, blaze | 620439 | 620590, 620910, 620510 | ❌ | ❌ |
| Men's or boys' shirts, knitted or crocheted - Of a | 610520 | 620110, 620110, 620110 | ❌ | ❌ |
| Women's or girls' slips, petticoats, briefs, panti | 610822 | (없음) | ❌ | ❌ |
| Men's or boys' suits, ensembles, jackets, blazers, | 620322 | (없음) | ❌ | ❌ |
| Men's or boys' shirts - Of artificial fibres | 620530 | 620510, 620590, 6206 | ❌ | ❌ |
| Men's or boys' singlets and other vests, underpant | 620719 | 620591, 620599, 620590 | ❌ | ❌ |
| Garments, made up of fabrics of heading 56.02, 56. | 621020 | 590310, 590310, 560210 | ❌ | ❌ |
| Track suits, ski suits and swimwear; other garment | 621143 | 610990, 620490, 620910 | ❌ | ❌ |
| Men's or boys' overcoats, car-coats, capes, cloaks | 620140 | 610890, 611190, 611990 | ❌ | ❌ |
| Brassieres, girdles, corsets, braces, suspenders,  | 621210 | 620510, 620590, 620510620590 | ❌ | ❌ |
| Ties, bow ties and cravats - Of silk or silk waste | 621510 | 620510, 620590, 62059010 | ❌ | ❌ |
| Women's or girls' suits, ensembles, jackets, blaze | 610449 | 620910, 620911, 620912 | ❌ | ❌ |
| Women's or girls' suits, ensembles, jackets, blaze | 620463 | (없음) | ❌ | ❌ |
| Women's or girls' blouses, shirts and shirt-blouse | 620640 | 62091000, 62091100, 62091900 | ❌ | ❌ |
| Women's or girls' slips, petticoats, briefs, panti | 610892 | 620490, 620510, 620610 | ❌ | ❌ |
| Men's or boys' singlets and other vests, underpant | 620729 | 620510, 620510, 620510 | ❌ | ❌ |
