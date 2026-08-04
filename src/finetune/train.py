"""
data/processed/train.jsonl(880건)로 gemma-2-2b를 QLoRA(순정 transformers +
peft + trl)로 파인튜닝하는 스크립트. Colab/Kaggle(T4 GPU) 실행을 전제로 작성했다 —
이 저장소를 관리하는 로컬 머신에는 CUDA GPU가 없어 이 스크립트는 로컬에서
실행/검증되지 않았다.

왜 unsloth를 안 쓰는가 (docs/10-experiment3_investigation.md 7~9절 참고):
  실험 3에서 unsloth로 학습한 adapter가 eval_loss 0.18까지 정상적으로 수렴했고
  unsloth 자체의 로드/추론 경로에서는 정상 생성됐지만, 병합 방식(unsloth 공식 병합,
  순정 peft 병합, 병합 없이 adapter만 부착, LoRA를 GGUF로 변환해 llama.cpp가 직접
  적용)을 6가지 넘게 바꿔가며 테스트해도 unsloth 밖에서 실행하면 전부 특정 희귀
  토큰만 반복하는 동일한 증상이 재현됐다. 즉 저장/병합 방식의 문제가 아니라 unsloth가
  Gemma-2를 계산하는 방식(예: 학습 속도를 위해 logit softcapping을 근사/생략하는 등)이
  순정 구현(transformers, llama.cpp 둘 다)과 근본적으로 다르고, 그 위에서 학습된 LoRA
  가중치는 순정 구현으로는 재현이 안 된다는 결론에 도달했다. 이 프로젝트의 실제 서빙
  경로(GGUF 변환 -> llama.cpp -> Ollama)는 순정 구현이므로, 애초에 학습을 순정
  transformers/peft로 해서 학습·추론이 같은 계산 경로를 쓰게 만드는 것으로 문제
  자체를 없앤다. unsloth의 Triton 최적화가 없어 학습은 더 느려지지만(그래도 GPU
  기준 수십 분 내), 병합 후 정상 동작이 보장된다는 이득이 훨씬 크다.

Colab/Kaggle에서 실행하는 방법:
  1. 런타임을 GPU(T4)로 설정
  2. !pip install -r requirements-colab.txt   # trl==0.24.0으로 버전 고정, 아래 코드가 이 버전 API 기준
  3. 이 저장소를 클론하거나 data/processed/train.jsonl, eval.jsonl을 업로드
  4. python src/finetune/train.py --smoke-test        # 먼저 50~100 step만 돌려서 loss 확인
  5. python src/finetune/train.py                     # 스모크 테스트 통과 후 전체 학습
  6. python src/finetune/merge_adapter_plain.py --adapter-dir outputs/adapter --output-dir outputs/merged

trl API 노트 (0.24.0 기준, 아래 코드가 이미 반영함):
  - SFTConfig는 max_seq_length가 아니라 max_length를 쓴다.
  - SFTConfig.bf16을 명시하지 않으면 fp16 미설정 시 자동으로 bf16=True가 되는데, T4는
    bf16을 지원하지 않는 GPU(Ampere 이전 세대)라 fp16=True, bf16=False를 명시해야 한다.
  - SFTTrainer.__init__에는 tokenizer 파라미터가 없고 processing_class만 받는다.
  - trl이 이후 버전에서 이 파라미터명을 또 바꾸면 이 스크립트도 같이 고쳐야 한다 —
    requirements-colab.txt에서 trl 버전을 고정해 두었으니, 원인 불명의 TypeError가
    나면 먼저 `pip show trl`로 실제 설치된 버전이 0.24.0인지부터 확인할 것.
  - (오진단 기록) 한때 데이터셋을 직접 토큰화해 input_ids만 넘기고 trl의 라벨 생성을
    건너뛰게 한 적이 있었는데, loss가 24대에서 시작해 3 epoch 내내 8~9대에 머무는
    걸 보고 그 우회가 원인이라 의심해 text 기반 파이프라인으로 되돌렸었다. 하지만
    되돌린 뒤에도 동일한 loss 곡선이 그대로 재현됐다 - 그 우회는 원인이 아니었다.
    진짜 원인은 프롬프트 전체(Instruction+Input+Response)에 대해 loss를 계산해서,
    매번 거의 동일한 Instruction/Input 부분(예측하기 쉬움)이 실제로 배워야 할 JSON
    응답 부분의 학습 신호를 희석시킨 것이었다. prompt/completion 컬럼을 분리하고
    SFTConfig(completion_only_loss=True)로 completion(응답)에만 loss를 매기도록
    고쳤다 - to_prompt_completion() 참고.

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
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# src/eval/baseline_eval.py의 FINETUNED_PROMPT_TEMPLATE과 반드시 동일해야 함 (그쪽은
# response 없이 "### Response:\n"까지만 있는 추론용 프롬프트 - 이 PROMPT_TEMPLATE의
# {response} 앞부분과 정확히 일치해야 학습/추론 프롬프트가 어긋나지 않는다)
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_prompt_completion(rec: dict) -> tuple[str, str]:
    """trl의 completion-only loss masking(SFTConfig(completion_only_loss=True))은
    prompt/completion 컬럼이 분리된 데이터셋에서만 적용된다. Instruction/Input
    부분(매 예제마다 거의 동일해 예측하기 쉬움)에는 loss를 매기지 않고, 실제로 배워야
    할 JSON 응답 부분에만 loss를 집중시키기 위해 분리한다. EOS는 trl이 completion에
    자동으로 붙여주므로 여기서 넣지 않는다."""
    prompt = PROMPT_TEMPLATE.format(instruction=rec["instruction"], input=rec["input"])
    completion = json.dumps(rec["output"], ensure_ascii=False)
    return prompt, completion


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
    """step별 loss / epoch별 eval_loss를 docs/02-training_log.md에 실시간으로 append.
    Kaggle 세션 시간 예산을 실측으로 검증할 수 있도록 각 행에 학습 시작 이후 경과 분도
    같이 남긴다(elapsed_min)."""

    def __init__(self, output_path: Path, run_config: dict):
        self.output_path = output_path
        self.start_time = time.monotonic()
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        header = ["# 파인튜닝 학습 로그", "", "## 학습 설정", ""]
        for k, v in run_config.items():
            header.append(f"- {k}: {v}")
        header += [
            "",
            "## Loss",
            "",
            "| step/epoch | 구분 | loss | elapsed_min |",
            "|---|---|---|---|",
        ]
        self.output_path.write_text("\n".join(header) + "\n", encoding="utf-8")

    def append_row(self, label: str, kind: str, loss: float) -> None:
        elapsed_min = (time.monotonic() - self.start_time) / 60
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(f"| {label} | {kind} | {loss:.4f} | {elapsed_min:.1f} |\n")


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
    # transformers/peft/trl/bitsandbytes는 GPU 환경(Colab/Kaggle 등)에서만 설치되어
    # 있다고 가정하고 함수 내부에서 import한다 (로컬 CPU 환경에서 이 파일을 import만
    # 해도 에러 나지 않도록).
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, EarlyStoppingCallback, set_seed
    from trl import SFTConfig, SFTTrainer

    set_seed(args.seed)

    logger.info("베이스 모델 로드: %s", args.base_model)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        dtype=torch.float16,
        device_map="auto",
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    records = load_jsonl(args.train_file)
    logger.info("train.jsonl 레코드 수: %d", len(records))
    train_records, val_records = internal_train_val_split(records, args.val_ratio, args.seed)
    logger.info("내부 분리: train=%d건, val(early stopping 모니터링용)=%d건", len(train_records), len(val_records))

    # prompt/completion으로 분리해서 넘기고, 포맷/EOS추가/토큰화/라벨 생성(completion에만
    # loss를 매기는 마스킹 포함)은 trl의 검증된 내부 파이프라인에 맡긴다.
    def to_records(recs: list[dict]) -> list[dict]:
        return [
            {"prompt": prompt, "completion": completion}
            for prompt, completion in (to_prompt_completion(r) for r in recs)
        ]

    train_ds = Dataset.from_list(to_records(train_records))
    val_ds = Dataset.from_list(to_records(val_records))

    run_config = {
        "base_model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": args.target_modules,
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
    #   - max_seq_length가 아니라 max_length를 쓴다.
    #   - bf16을 명시하지 않으면 fp16 미설정 시 자동으로 bf16=True가 되는데, T4는
    #     bf16을 지원하지 않는 GPU(Ampere 이전 세대)라 fp16=True, bf16=False를 명시해야
    #     한다 (A100/L4 등 bf16 지원 GPU로 옮기면 반대로 바꿀 것).
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
        # prompt/completion 데이터셋에서만 적용됨(dataset_text_field는 language-modeling
        # 형식용이라 여기선 안 씀). Instruction/Input 부분(매번 거의 동일해 예측하기
        # 쉬움)은 loss에서 제외하고 실제로 배워야 할 completion(JSON 응답)에만 loss를
        # 집중시킨다 - 이게 없으면 전체 시퀀스 평균 loss에 쉬운 boilerplate가 희석돼
        # 학습 신호가 약해진다.
        completion_only_loss=True,
        max_length=args.max_seq_length,
        fp16=True,
        bf16=False,
        report_to="none",
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
    logger.info("다음 단계: python src/finetune/merge_adapter_plain.py --adapter-dir %s", adapter_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-2-2b",
        help="HF 모델명 (기본값은 unsloth가 올려둔 ungated fp16 미러 - QLoRA로 학습 시점에 "
        "4bit 동적 양자화한다. google/gemma-2-2b 원본은 라이선스 동의/인증이 필요한 "
        "gated 저장소라 피했다. 이 미러는 가중치 호스팅만 unsloth 계정일 뿐 unsloth의 "
        "학습 코드 패치와는 무관하다 - unsloth의 사전 양자화 체크포인트(-bnb-4bit)와 "
        "달리 순정 transformers로 그대로 로드된다. docs/10-experiment3_investigation.md 참고)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--training-log", type=Path, default=Path("docs/02-training_log.md"))
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="LoRA를 적용할 모듈(콤마 구분). MLP까지 넓히려면 예: "
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="EarlyStoppingCallback(patience=1)이 eval_loss 정체 시 조기 종료하므로 넉넉히 잡음 "
        "(completion-only loss 도입 전 3 epoch로는 loss가 8대에서 못 벗어났음)",
    )
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
