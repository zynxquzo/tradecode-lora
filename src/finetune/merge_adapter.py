"""
train.py가 저장한 LoRA adapter를 베이스 모델(google/gemma-2-2b)과 병합해 하나의
HF 모델 디렉토리로 저장한다. GGUF 변환(llama.cpp)은 병합된 전체 가중치 형태를
입력으로 받기 때문에 이 단계가 필요하다.

fp16 병합이라 2B 모델 기준 메모리 요구량이 크지 않으므로(~5GB) GPU 없이 Colab
CPU 런타임이나 로컬에서도 돌릴 수 있다 (단, unsloth 4bit 모델로 학습했다면 병합은
GPU가 있는 편이 안전 — bnb 4bit -> fp16 dequant 과정이 CPU에서 매우 느릴 수 있음).

사용 예:
  python src/finetune/merge_adapter.py \
      --adapter-dir outputs/adapter \
      --base-model google/gemma-2-2b \
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
    parser.add_argument("--base-model", type=str, default="google/gemma-2-2b")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/merged"))
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu"],
        help="'cpu'는 GPU 없는 환경에서 안전하게 병합할 때 사용 (느림)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
