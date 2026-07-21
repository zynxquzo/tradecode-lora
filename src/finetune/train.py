"""
data/processed/train.jsonl(880건)로 google/gemma-2-2b를 Unsloth 기반 LoRA로
파인튜닝하는 스크립트. Colab(T4 GPU) 실행을 전제로 작성했다 — 이 저장소를 관리하는
로컬 머신에는 CUDA GPU가 없어 이 스크립트는 로컬에서 실행/검증되지 않았다.

Colab에서 실행하는 방법:
  1. 런타임을 GPU(T4)로 설정
  2. !pip install -r requirements-colab.txt   # trl==0.24.0으로 버전 고정, 아래 코드가 이 버전 API 기준
  3. 이 저장소를 클론하거나 data/processed/train.jsonl, eval.jsonl을 업로드
  4. python src/finetune/train.py --smoke-test        # 먼저 50~100 step만 돌려서 loss 확인
  5. python src/finetune/train.py                     # 스모크 테스트 통과 후 전체 학습

trl API 노트 (0.24.0 기준, 아래 코드가 이미 반영함):
  - SFTConfig는 max_seq_length가 아니라 max_length를 쓴다.
  - SFTConfig.bf16을 명시하지 않으면 fp16 미설정 시 자동으로 bf16=True가 되는데, T4는
    bf16을 지원하지 않는 GPU(Ampere 이전 세대)라 fp16=True, bf16=False를 명시해야 한다.
  - SFTTrainer.__init__에는 tokenizer 파라미터가 없고 processing_class만 받는다.
  - trl이 이후 버전에서 이 파라미터명을 또 바꾸면 이 스크립트도 같이 고쳐야 한다 —
    requirements-colab.txt에서 trl 버전을 고정해 두었으니, 원인 불명의 TypeError가
    나면 먼저 `pip show trl`로 실제 설치된 버전이 0.24.0인지부터 확인할 것.

학습 데이터 포맷은 baseline_eval.py의 zero-shot 프롬프트와 다르다 — baseline은
"Top-3 JSON 배열"을 요구하지만, 학습 데이터(data/processed/train.jsonl)에는 레코드당
정답 1개(hs_code, confidence_basis)만 있다. 그래서 이 스크립트는 정답 스키마
({"hs_code": ..., "confidence_basis": ...}) 그대로를 생성하도록 타겟을 구성한다
(baseline에서 "출력 포맷 준수 불안정"이 문제였으므로, 학습 시 스키마와 추론 시 스키마를
일치시키는 것이 최우선). 이 프롬프트 형식은 src/eval/baseline_eval.py의
PROMPT_TEMPLATES["finetuned"]와 반드시 짝을 맞춰야 한다 — 하나를 바꾸면 다른 하나도
같이 바꿀 것.

train.jsonl 자체는 eval.jsonl(최종 baseline vs fine-tuned 비교용 홀드아웃, 절대 학습에
사용하지 않음)과 별개로, 내부적으로 다시 90/10 분리해 eval loss를 모니터링하고
early stopping에 사용한다.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# src/eval/baseline_eval.py의 finetuned 프롬프트 스타일과 반드시 동일해야 함
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}"""


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_training_text(rec: dict, eos_token: str) -> str:
    response_json = json.dumps(rec["output"], ensure_ascii=False)
    text = PROMPT_TEMPLATE.format(
        instruction=rec["instruction"], input=rec["input"], response=response_json
    )
    return text + eos_token


def internal_train_val_split(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """train.jsonl을 다시 나눠 학습 중 eval loss 모니터링용 val 셋을 만든다.
    eval.jsonl(최종 비교용 홀드아웃)에는 절대 손대지 않는다."""
    import random

    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


class TrainingLogWriter:
    """step별 loss / epoch별 eval_loss를 docs/02-training_log.md에 실시간으로 append."""

    def __init__(self, output_path: Path, run_config: dict):
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["# 파인튜닝 학습 로그", "", "## 학습 설정", ""]
        for k, v in run_config.items():
            header.append(f"- {k}: {v}")
        header += ["", "## Loss", "", "| step/epoch | 구분 | loss |", "|---|---|---|"]
        self.output_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    def append_row(self, label: str, kind: str, loss: float) -> None:
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(f"| {label} | {kind} | {loss:.4f} |\n")


def build_log_callback(log_writer: TrainingLogWriter):
    from transformers import TrainerCallback

    class LogCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            if "loss" in logs:
                log_writer.append_row(f"step {state.global_step}", "train", logs["loss"])
            if "eval_loss" in logs:
                log_writer.append_row(f"step {state.global_step}", "eval", logs["eval_loss"])

    return LogCallback()


def run(args: argparse.Namespace) -> None:
    # Unsloth/trl/transformers는 GPU 환경(Colab 등)에서만 설치되어 있다고 가정하고
    # 함수 내부에서 import한다 (로컬 CPU 환경에서 이 파일을 import만 해도 에러 나지 않도록).
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    logger.info("베이스 모델 로드: %s", args.base_model)
    # 이 max_seq_length는 unsloth FastLanguageModel 고유 파라미터(모델/토크나이저 로드 시
    # 시퀀스 길이 최적화용)이며, 아래 SFTConfig(max_length=...)와 이름이 다르지만
    # 같은 값을 의미한다 - 헷갈리지 않도록 둘 다 args.max_seq_length에서 채운다.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,  # 자동 감지 (T4 -> float16)
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    records = load_jsonl(args.train_file)
    logger.info("train.jsonl 레코드 수: %d", len(records))
    train_records, val_records = internal_train_val_split(records, args.val_ratio, args.seed)
    logger.info("내부 분리: train=%d건, val(early stopping 모니터링용)=%d건", len(train_records), len(val_records))

    eos_token = tokenizer.eos_token
    train_ds = Dataset.from_list(
        [{"text": to_training_text(r, eos_token)} for r in train_records]
    )
    val_ds = Dataset.from_list([{"text": to_training_text(r, eos_token)} for r in val_records])

    run_config = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": "q_proj,k_proj,v_proj,o_proj",
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
        "smoke_test": args.smoke_test,
        "max_steps": args.max_steps if args.smoke_test else "N/A (full run)",
        "train_records": len(train_records),
        "val_records": len(val_records),
    }
    log_writer = TrainingLogWriter(args.training_log, run_config)

    # trl==0.24.0 기준 SFTConfig/SFTTrainer API (requirements-colab.txt에서 trl 버전을
    # 고정하고 있으니 여기서는 버전 분기 없이 해당 버전의 실제 파라미터명을 그대로 쓴다):
    #   - max_seq_length -> max_length로 이름 변경됨 (SFTConfig 자체 필드).
    #     참고: 바로 위 FastLanguageModel.from_pretrained(max_seq_length=...)는 unsloth
    #     고유 파라미터라 이름이 같아도 별개이며 그대로 유지한다.
    #   - SFTConfig.bf16은 fp16을 명시하지 않으면 기본값 None -> bf16=True로 자동
    #     전환된다. T4는 Ampere 이전 세대라 bf16을 지원하지 않으므로 fp16=True,
    #     bf16=False를 반드시 명시해야 한다 (A100/L4 등 bf16 지원 GPU로 옮기면 반대로
    #     바꿀 것).
    #   - SFTTrainer.__init__에는 tokenizer 파라미터가 더 이상 존재하지 않고
    #     processing_class만 받는다.
    training_args = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.smoke_test else -1,
        learning_rate=args.learning_rate,
        logging_steps=1 if args.smoke_test else 10,
        eval_strategy="steps" if args.smoke_test else "epoch",
        eval_steps=10 if args.smoke_test else None,
        save_strategy="steps" if args.smoke_test else "epoch",
        save_steps=10 if args.smoke_test else None,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        dataset_text_field="text",
        max_length=args.max_seq_length,
        fp16=True,
        bf16=False,
        report_to="none",
        # dataset_num_proc 기본값(None)이면 trl이 datasets.map()을 멀티프로세스로 돌리는데,
        # unsloth로 패치된 모델/설정 객체를 워커 프로세스로 넘기려다 dill이
        # pickle하지 못해 TypeError('ConfigModuleInstance')가 난다. 880건 규모라
        # 단일 프로세스로도 충분히 빠르므로 1로 고정해 멀티프로세싱 자체를 피한다.
        dataset_num_proc=1,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        callbacks=[
            build_log_callback(log_writer),
            # 880건 규모라 3 epoch도 과적합 위험 -> eval_loss가 1회라도 개선 안 되면 중단
            EarlyStoppingCallback(early_stopping_patience=1),
        ],
    )

    logger.info("학습 시작 (smoke_test=%s)", args.smoke_test)
    trainer.train()

    adapter_dir = args.output_dir / "adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    logger.info("LoRA adapter 저장 완료: %s", adapter_dir)
    logger.info("학습 로그: %s", args.training_log)
    logger.info(
        "다음 단계: python src/finetune/merge_adapter.py --adapter-dir %s --base-model %s",
        adapter_dir,
        args.base_model,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-2-2b-bnb-4bit",
        help="Unsloth 4bit 사전 양자화 모델명 (google/gemma-2-2b도 가능하나 4bit 버전이 Colab T4에서 더 빠름)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--training-log", type=Path, default=Path("docs/02-training_log.md"))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accumulation", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--val-ratio", type=float, default=0.1, help="train.jsonl 내부 early-stopping용 val 비율")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="본 학습 전 --max-steps 만큼만 빠르게 돌려 loss 하강을 확인하는 모드",
    )
    parser.add_argument("--max-steps", type=int, default=60, help="--smoke-test에서 사용할 최대 step 수")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
