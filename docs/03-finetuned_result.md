# Fine-tuned Evaluation Result (플레이스홀더)

아직 파인튜닝/GGUF 변환/Ollama 등록이 완료되지 않아 실제 결과가 없다. 아래 명령으로
생성된다.

## 1차 확인 (100건, 빠른 검증용)

```
python src/eval/baseline_eval.py \
  --model tradecode-gemma2 \
  --prompt-style finetuned \
  --limit 100 \
  --output docs/03-finetuned_result_partial.md
```

## 전체 재평가 (280건, baseline과 동일 조건)

```
python src/eval/baseline_eval.py \
  --model tradecode-gemma2 \
  --prompt-style finetuned \
  --output docs/03-finetuned_result.md
```

실행 후 이 파일은 `save_markdown_report()`가 생성한 실제 결과로 교체된다
(Exact Match / Partial(4자리) / Partial(2자리) / Top-3 Recall / Parse Failure Rate +
샘플 예측 20건).
