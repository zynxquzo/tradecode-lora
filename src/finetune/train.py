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
  - dataset.map()은 num_proc이 정수면(1이라도) multiprocessing.Pool을 거치는데, 그
    과정에서 unsloth가 건드려놓은 torch._dynamo.config(pickle 불가능한
    ConfigModuleInstance)까지 클로저에 딸려가 TypeError로 죽는다.
    SFTConfig(dataset_num_proc=None)을 명시해 Pool 자체를 안 타는 진짜 단일
    프로세스 경로로 강제한다 (num_proc=1도 여전히 Pool(1)을 쓰므로 안 됨 - None만
    Pool을 건너뜀).
  - (오진단 기록) 한때 데이터셋을 직접 토큰화해 input_ids만 넘기고 trl의 라벨 생성을
    건너뛰게 한 적이 있었는데, loss가 24대에서 시작해 3 epoch 내내 8~9대에 머무는
    걸 보고 그 우회가 원인이라 의심해 text 기반 파이프라인으로 되돌렸었다. 하지만
    되돌린 뒤에도 동일한 loss 곡선이 그대로 재현됐다 - 그 우회는 원인이 아니었다.
    진짜 원인은 프롬프트 전체(Instruction+Input+Response)에 대해 loss를 계산해서,
    매번 거의 동일한 Instruction/Input 부분(예측하기 쉬움)이 실제로 배워야 할 JSON
    응답 부분의 학습 신호를 희석시킨 것이었다. prompt/completion 컬럼을 분리하고
    SFTConfig(completion_only_loss=True)로 completion(응답)에만 loss를 매기도록
    고쳤다 - to_prompt_completion() 참고.
  - trl 0.24.0의 SFTTrainer.compute_loss는 use_liger_kernel이 아니면 무조건
    entropy_from_logits(outputs.logits)와 토큰 정확도 계산으로 outputs.logits를
    두 번 더 쓴다. unsloth는 VRAM 절약을 위해 outputs.logits를 지연 계산용
    콜러블로 반환하는데("Unsloth: Will smartly offload gradients to save VRAM!"
    로그가 그 신호), UNSLOTH_RETURN_LOGITS=1(모듈 최상단)을 걸어도 eval 스텝을
    한 번 통과하며 torch.compile이 그 분기를 다시 캐싱해버려 재발했다
    (`TypeError: 'function' object is not subscriptable`). 두 로깅 모두 실제
    loss 계산과 무관하므로(loss는 hidden_states에서 fused 계산됨),
    build_trainer_class()의 UnslothCompatSFTTrainer가 compute_loss를 상위
    transformers.Trainer 버전으로 완전히 우회해 outputs.logits를 아예 안 건드리게
    한다.
  - unsloth가 컴파일 캐시를 만들며 trl.trainer.sft_config 모듈을 자체적으로 다시
    exec하면, sys.modules에 등록된 SFTConfig가 실제로 인스턴스를 만든 클래스와
    다른 사본이 될 수 있다. 체크포인트 저장 시 pickle이 "Can't pickle <class
    'trl.trainer.sft_config.SFTConfig'>: it's not the same object as ..."로
    죽는 원인이 이것 - SFTConfig 생성 직후 sys.modules 등록을 실제 클래스로 맞춰
    고치고, 혹시 남는 경우를 대비해 UnslothCompatSFTTrainer._save에도 방어적으로
    PicklingError를 흡수하는 안전장치를 둔다(모델/adapter 저장은 이미 끝난 뒤
    training_args.bin 저장만 실패하는 것이므로 무시해도 안전함).

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
import os
import sys
from pathlib import Path

# unsloth가 VRAM 절약을 위해 outputs.logits를 지연 계산용 콜러블로 반환하는 것을
# 막고 실제 텐서를 돌려받기 위한 설정 - trl 0.24.0의 entropy_from_logits 호환성
# 문제(모듈 상단 trl API 노트 참고) 때문에 필요. unsloth를 import하기 전에
# 설정해야 하므로 모듈 최상단에 둔다.
os.environ.setdefault("UNSLOTH_RETURN_LOGITS", "1")

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


def build_trainer_class():
    """trl 0.24.0의 SFTTrainer.compute_loss는 loss 계산과 무관하게 outputs.logits로
    entropy/token-accuracy를 추가로 로깅하는데(use_liger_kernel이 아니면 무조건 실행),
    unsloth는 VRAM 절약을 위해 outputs.logits를 지연 계산용 콜러블로 반환한다.
    UNSLOTH_RETURN_LOGITS=1로 실제 텐서를 강제해도, eval 스텝을 한 번 통과하면서
    torch.compile이 그 분기를 다르게(logits 미반환) 캐싱해버려 재발한다(재현: 첫
    eval_steps 직후 학습 스텝에서 크래시). entropy/accuracy 로깅은 진단용일 뿐 loss
    계산에는 쓰이지 않으므로(unsloth가 hidden_states에서 직접 fused loss를 계산),
    compute_loss를 상위 transformers.Trainer 버전으로 완전히 우회해 outputs.logits
    자체를 건드리지 않게 한다 - unsloth의 지연 logits/재컴파일 문제와 무관해진다."""
    import pickle

    from transformers import Trainer
    from trl import SFTTrainer

    class UnslothCompatSFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return Trainer.compute_loss(
                self, model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )

        def _save(self, output_dir=None, state_dict=None):
            # 방어적 안전장치: run()에서 sys.modules의 SFTConfig 등록을 맞춰주지만,
            # unsloth가 재컴파일 시점에 또 다른 사본을 만들면 여기서도 같은
            # PicklingError가 재발할 수 있다. 이 시점에는 모델/토크나이저 저장은 이미
            # 끝난 뒤 마지막 줄(torch.save(self.args, ...))만 실패하는 것이므로,
            # 실제 어댑터 가중치 손실 없이 training_args.bin 저장만 건너뛰고 경고로
            # 남긴다.
            try:
                super()._save(output_dir=output_dir, state_dict=state_dict)
            except pickle.PicklingError as e:
                logger.warning("training_args.bin 저장 실패(무시, 가중치는 이미 저장됨): %s", e)

    return UnslothCompatSFTTrainer


def run(args: argparse.Namespace) -> None:
    # Unsloth/trl/transformers는 GPU 환경(Colab 등)에서만 설치되어 있다고 가정하고
    # 함수 내부에서 import한다 (로컬 CPU 환경에서 이 파일을 import만 해도 에러 나지 않도록).
    from datasets import Dataset
    from transformers import EarlyStoppingCallback
    from trl import SFTConfig
    from unsloth import FastLanguageModel

    TrainerClass = build_trainer_class()

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
        # prompt/completion 데이터셋에서만 적용됨(dataset_text_field는 language-modeling
        # 형식용이라 여기선 안 씀). Instruction/Input 부분(매번 거의 동일해 예측하기
        # 쉬움)은 loss에서 제외하고 실제로 배워야 할 completion(JSON 응답)에만 loss를
        # 집중시킨다 - 이게 없으면 전체 시퀀스 평균 loss에 쉬운 boilerplate가 희석돼
        # 학습 신호가 약해진다.
        completion_only_loss=True,
        # num_proc이 정수면(1이라도) datasets.map()이 multiprocessing.Pool을 거치며
        # unsloth가 건드린 torch._dynamo.config(ConfigModuleInstance)가 pickle에
        # 실패한다 - None만 진짜 단일 프로세스 경로(Pool 자체를 안 씀)를 탄다.
        # Unsloth가 SFTConfig 기본값을 몰래 다른 값으로 주입하는 경우가 있어
        # 명시적으로 None을 강제한다.
        dataset_num_proc=None,
        max_length=args.max_seq_length,
        fp16=True,
        bf16=False,
        report_to="none",
    )
    # unsloth가 컴파일 캐시를 만들며 trl.trainer.sft_config 모듈을 자체적으로 다시
    # exec해서, sys.modules에 등록된 SFTConfig가 우리가 실제로 인스턴스를 만든 클래스
    # 객체와 다른 사본이 되는 경우가 있다. 체크포인트 저장 시 pickle이
    # "Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>: it's not the same
    # object as ..."로 죽는 원인이 이것이다 (pickle은 클래스를
    # sys.modules[모듈].이름으로 다시 찾아 identity를 대조한다). 등록을 우리가 실제로
    # 쓰는 클래스로 맞춰서 근본 원인을 고친다.
    sft_config_module = sys.modules.get(type(training_args).__module__)
    if sft_config_module is not None:
        setattr(sft_config_module, type(training_args).__qualname__, type(training_args))

    trainer = TrainerClass(
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
    # merge_adapter.py는 이제 unsloth의 FastLanguageModel.from_pretrained(adapter_dir)로
    # base_model_name_or_path(adapter_config.json에 기록됨)까지 자동으로 찾아 불러오므로
    # --base-model 인자 자체가 없다.
    logger.info("다음 단계: python src/finetune/merge_adapter.py --adapter-dir %s", adapter_dir)


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
