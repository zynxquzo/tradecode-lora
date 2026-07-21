"""
train.py가 저장한 LoRA adapter를 풀 정밀도 베이스 모델과 병합해 하나의 HF 모델
디렉토리로 저장한다. GGUF 변환(llama.cpp)은 병합된 전체 가중치 형태를 입력으로
받기 때문에 이 단계가 필요하다.

기본 베이스 모델은 unsloth/gemma-2-2b(비양자화 bf16 미러, config.json에
quantization_config 없음)다. google/gemma-2-2b와 동일 가중치이지만 google 쪽은
HF에서 라이선스 동의 + 인증이 필요한 게이트 저장소라 Colab에서 바로 받으면
401 GatedRepoError가 난다. google/gemma-2-2b를 굳이 쓰려면 huggingface_hub.login()
등으로 먼저 인증하고 https://huggingface.co/google/gemma-2-2b 에서 라이선스에
동의해야 한다.

fp16 병합이라 2B 모델 기준 메모리 요구량이 크지 않으므로(~5GB) GPU 없이 Colab
CPU 런타임이나 로컬에서도 돌릴 수 있다 (단, unsloth 4bit 모델로 학습했다면 병합은
GPU가 있는 편이 안전 — bnb 4bit -> fp16 dequant 과정이 CPU에서 매우 느릴 수 있음).

사용 예:
  python src/finetune/merge_adapter.py \
      --adapter-dir outputs/adapter \
      --base-model unsloth/gemma-2-2b \
      --output-dir outputs/merged
"""

import argparse
import logging
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run(args: argparse.Namespace) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if "4bit" in args.base_model.lower() or "bnb" in args.base_model.lower():
        raise ValueError(
            f"--base-model={args.base_model}는 4bit 양자화 체크포인트로 보입니다. "
            "LoRA는 반드시 풀 정밀도 베이스(예: google/gemma-2-2b)에 병합해야 합니다 - "
            "4bit 베이스에 병합하면 정밀도 손실 경고가 뜨고, save_pretrained 시 "
            "transformers가 4bit 전용 가중치 레이아웃을 되돌리지 못해 "
            "NotImplementedError로 실패합니다. train.py는 학습 효율을 위해 4bit "
            "베이스(unsloth/gemma-2-2b-bnb-4bit)를 썼지만, 병합 단계는 그와 별개로 "
            "항상 풀 정밀도 베이스를 써야 합니다."
        )

    logger.info("베이스 모델(fp16) 로드: %s", args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto" if args.device == "auto" else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    logger.info("LoRA adapter 로드 및 병합: %s", args.adapter_dir)
    merged = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    merged = merged.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))
    logger.info("병합 완료, 저장 경로: %s", args.output_dir)
    logger.info(
        "다음 단계: bash src/serving/build_ollama_model.sh %s", args.output_dir
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/adapter"))
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-2-2b",
        help="풀 정밀도(비양자화) 베이스 모델. google/gemma-2-2b는 HF 게이트 저장소라 "
        "인증 없이는 401이 난다 - 기본값(unsloth 미러)은 게이트 없이 동일 가중치를 받는다.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/merged"))
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu"],
        help="'cpu'는 GPU 없는 환경에서 안전하게 병합할 때 사용 (느림)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
