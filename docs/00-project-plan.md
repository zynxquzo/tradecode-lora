# TradeCode-LoRA 상세 기획안

## 1. 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 프로젝트명 | TradeCode-LoRA |
| 목표 | 상품설명 → HS코드 Top-3 추천 파인튜닝 모델 개발 및 로컬 서빙 |
| 베이스 모델 | google/gemma-2-2b (HF) → 학습 후 GGUF 변환 → Ollama 서빙 |
| 차별화 포인트 | RAG/에이전트 위주였던 기존 포트폴리오 대비 "직접 파인튜닝" 역량 증명 |

## 2. 문제 정의

- **Input**: 상품 영문/국문 설명 (예: "Men's cotton knitted T-shirts, short sleeve")
- **Output**: HS코드 Top-3 후보 + 확신도, 근거 키워드
- **실무 임팩트**: 품목분류 실수는 관세 추징·통관 지연으로 이어짐 → 1차 스크리닝 자동화 가치

## 3. 데이터 설계

**범위 축소**: 프로젝트 규모 고려해 **섬유·의류 품목(HS 제61~62류)** 한 챕터로 한정

### 소스 후보
- 관세청 품목분류 사전회시 사례 (관세법령정보포털, tariff.go.kr)
- 수출입무역통계 품목명-HS코드 매핑 (무역통계진흥원, 관세청 공개 API)
- 수집 방법: 공개 API/다운로드 우선, 없으면 웹 페이지 구조화 크롤링 (저작권/이용약관 확인 필수)

### 목표 규모
- 400~600쌍 (train 80% / eval 20%)

### 스키마
```json
{
  "instruction": "다음 상품설명에 해당하는 HS코드를 6자리까지 추천하고 근거를 설명하세요.",
  "input": "Men's cotton knitted T-shirts, short sleeve, crew neck",
  "output": {
    "hs_code": "610910",
    "confidence_basis": "메리야스 편물, 면 소재, 티셔츠류 → 61류(편물제 의류) > 6109(티셔츠·러닝셔츠) > 610910(면제)"
  }
}
```

### 전처리
- 중복 제거
- 클래스 불균형 체크 (특정 HS코드에 샘플 쏠림 방지)
- train/eval 분리 시 6자리 코드 기준 stratified split

## 4. 파인튜닝 설계

- **방법**: Unsloth 기반 LoRA (Colab 무료 티어 T4 GPU로 충분)
- **하이퍼파라미터 초안**
  - LoRA rank: 16, alpha: 32
  - target_modules: q_proj, k_proj, v_proj, o_proj
  - learning_rate: 2e-4, epochs: 3, batch_size: 4 (gradient accumulation 4)
- **Baseline 비교**: 파인튜닝 전 gemma2:2b에 동일 프롬프트로 zero-shot / few-shot(3-shot) 추론 → 파인튜닝 후와 비교할 기준선 확보
- **변환**: PEFT merge → GGUF 변환(llama.cpp) → Ollama Modelfile 작성

## 5. 평가 설계 (핵심 지표)

| 지표 | 정의 |
|---|---|
| Exact Match (6자리) | 완전일치 정확도 |
| Partial Match (4자리) | 호 단위 일치율 |
| Partial Match (2자리) | 류 단위 일치율 |
| Top-3 Recall | 정답이 Top-3 후보 안에 포함된 비율 |

Baseline vs Fine-tuned 결과를 표/차트로 정리 → "6자리 완전일치 X%→Y%, 2자리 기준 A%→B%" 서사 확보

## 6. 서빙 아키텍처

```
사용자 입력(상품설명)
    ↓
FastAPI (/predict)
    ↓
Ollama REST API (localhost:11434) — fine-tuned gemma2:2b
    ↓
후처리: Top-3 파싱, confidence 낮으면 "수동확인 필요" 플래그
    ↓
응답 (JSON) → (선택) 간단 프론트/Streamlit 데모
```

## 7. 폴더 구조

```
tradecode-lora/
├── data/
│   ├── raw/            # 원본 (git 제외)
│   └── processed/       # instruction 포맷 jsonl
├── src/
│   ├── finetune/         # LoRA 학습 스크립트
│   ├── eval/             # baseline vs fine-tuned 평가 스크립트
│   └── serving/          # FastAPI + Ollama 연동
├── notebooks/            # 탐색/실험용
├── docs/                 # 실험일지, 결과 리포트
└── README.md
```

## 8. 리스크 및 대응

- **데이터 부족/품질**: 공개 사전회시 데이터가 예상보다 적으면 범위를 61류(편물) 하나로 더 좁히거나, 수출입통계 매핑 데이터로 보강
- **GGUF 변환 이슈**: llama.cpp 버전 호환성 문제 대비, Ollama 공식 gemma2 Modelfile 예시 먼저 확인
- **평가 신뢰도**: eval셋이 작으므로(80~120개) 절대 수치보다 baseline 대비 상대적 개선폭을 중심 서사로 사용
