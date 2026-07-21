# 파인튜닝 학습 로그

이 파일은 `src/finetune/train.py`가 실행될 때 자동으로 덮어써지고, step/epoch별
loss가 실시간으로 append된다 (`TrainingLogWriter` 참고). 아래는 로컬에 GPU가 없어
아직 실행 전 상태의 플레이스홀더다.

## 실행 방법 (Colab, T4 GPU)

```
!pip install unsloth transformers trl peft accelerate bitsandbytes

# 1) 스모크 테스트 (50~60 step, loss가 정상적으로 떨어지는지만 확인)
!python src/finetune/train.py --smoke-test --max-steps 60

# 2) 스모크 테스트 통과 후 전체 학습 (eval loss 기준 early stopping 포함)
!python src/finetune/train.py
```

## 기록할 항목 (실행 후 이 섹션을 실제 로그로 교체)

- 학습 설정 (base model, LoRA r/alpha, lr, epoch, batch size)
- step별 train loss, epoch별 eval loss
- early stopping이 실제로 발동했는지 (몇 epoch에서 멈췄는지)
- 총 학습 소요 시간
- 최종 adapter 저장 경로 (`outputs/adapter`)
