"""
train.py가 저장한 LoRA adapter를 Unsloth 공식 병합 API(save_pretrained_merged)로
16bit 풀 정밀도 모델과 병합해 하나의 HF 모델 디렉토리로 저장한다. GGUF 변환
(llama.cpp)은 병합된 전체 가중치 형태를 입력으로 받기 때문에 이 단계가 필요하다.

Colab(GPU) 실행 전제 - unsloth가 필요하다 (requirements-colab.txt).

왜 순정 peft.PeftModel.merge_and_unload()를 안 쓰는가:
  처음엔 순정 transformers.AutoModelForCausalLM + peft.PeftModel.merge_and_unload()로
  별도의 풀 정밀도 베이스(unsloth/gemma-2-2b)에 병합했는데, GGUF 변환 후(양자화
  여부와 무관하게 f16 상태에서도) 출력이 완전히 깨진 텍스트만 나오는 문제가 있었다.
  train.py는 Unsloth가 자체 최적화한 4bit 베이스(unsloth/gemma-2-2b-bnb-4bit)로
  학습했는데, 그렇게 학습한 adapter를 순정 HF/PEFT로 별도 로드한 (아키텍처가 완전히
  똑같다고 보장되지 않는) 베이스에 병합하는 조합은 Unsloth가 공식 지원하지 않는다.
  Unsloth는 이 경우 FastLanguageModel.from_pretrained(model_name=adapter_dir)로
  adapter까지 포함해 다시 불러온 뒤 model.save_pretrained_merged(...,
  save_method="merged_16bit")로 병합하는 것을 공식 API로 제공하므로 이걸 쓴다.

사용 예:
  python src/finetune/merge_adapter.py \
      --adapter-dir outputs/adapter \
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
    from unsloth import FastLanguageModel

    logger.info("adapter + 베이스 모델 로드 (unsloth): %s", args.adapter_dir)
    # adapter_dir의 adapter_config.json에 base_model_name_or_path가 기록돼 있어서
    # unsloth가 베이스 모델까지 알아서 찾아 불러온다 (train.py가 학습에 쓴
    # unsloth/gemma-2-2b-bnb-4bit). load_in_4bit=False로 줘서 병합은 16bit로 한다.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.adapter_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("16bit 병합 저장 시작: %s", args.output_dir)
    model.save_pretrained_merged(str(args.output_dir), tokenizer, save_method="merged_16bit")

    logger.info("병합 완료, 저장 경로: %s", args.output_dir)
    logger.info(
        "다음 단계: bash src/serving/build_ollama_model.sh %s", args.output_dir
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/merged"))
    parser.add_argument("--max-seq-length", type=int, default=1024)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
