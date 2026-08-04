"""
merge_adapter.py의 대안 경로 — unsloth의 save_pretrained_merged() 대신 순정
transformers + peft.PeftModel.merge_and_unload()로 병합한다.

왜 필요한가: unsloth(2026.8.1 기준, requirements-colab.txt에 버전 고정 안 돼 있어
설치 시점 최신이 깔림)의 save_pretrained_merged()로 병합한 모델이, 학습 loss는
정상(eval_loss 0.18까지 하강)인데도 어떤 입력을 줘도 특정 희귀 토큰
(purpoſe/vectorielle/myſelf 등, 장음 s가 섞인 고어체·프랑스어 단어)만 반복하는
현상이 실험 3에서 반복 확인됐다. 반면 학습/병합을 전혀 거치지 않은 순정 베이스
모델(unsloth/gemma-2-2b-bnb-4bit)은 순정 transformers로 로드했을 때 같은 프롬프트에
정상적으로 숫자를 생성했다 — 즉 문제는 학습이 아니라 unsloth의 병합 단계 자체에
있다고 결론 내리고, 이 스크립트로 병합 단계만 순정 라이브러리 조합으로 교체해본다.

merge_adapter.py의 docstring에 "순정 peft.PeftModel.merge_and_unload()를 이미
시도했다가 실패했다"는 기록이 있지만, 그때는 학습에 쓴 4bit 베이스
(unsloth/gemma-2-2b-bnb-4bit)가 아니라 별도의 풀 정밀도 베이스(unsloth/gemma-2-2b)에
병합해 아키텍처 불일치 가능성이 있었다. 이 스크립트는 학습에 실제로 쓰인 것과 동일한
4bit 베이스에 peft로 병합해 그 변수를 제거한다.

사용 예:
  python src/finetune/merge_adapter_plain.py \
      --adapter-dir outputs/adapter --output-dir outputs/merged_plain
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
    import json

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = args.adapter_dir / "adapter_config.json"
    trained_base_name = json.loads(adapter_config_path.read_text())["base_model_name_or_path"]

    # 학습에 쓴 베이스는 4bit(bnb) 사전 양자화 체크포인트라, peft가 거기에 병합하면
    # 결과를 다시 4bit로 재양자화한다(bnb.py의 Linear4bit.merge()). 이 재양자화된
    # 상태를 최신 transformers의 safetensors 저장 로직(core_model_loading.py의
    # revert_weight_conversion)이 아직 지원하지 않아 save_pretrained()가
    # NotImplementedError로 죽는다. LoRA 가중치는 dtype에 무관하게 모듈 이름/shape만
    # 맞으면 병합되므로, 같은 모델의 fp16(비양자화) 버전에 병합해 이 문제를 피한다.
    base_model_name = args.full_precision_base or trained_base_name.removesuffix("-bnb-4bit")
    logger.info("학습 베이스(4bit): %s / 병합에 쓸 fp16 베이스: %s", trained_base_name, base_model_name)

    logger.info("베이스 모델 로드 (순정 transformers, fp16)")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    logger.info("adapter 로드 및 병합: %s", args.adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("병합 모델 저장: %s", args.output_dir)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))

    logger.info("병합 완료, 저장 경로: %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/merged_plain"))
    parser.add_argument(
        "--full-precision-base",
        type=str,
        default=None,
        help="병합에 쓸 fp16 베이스 모델명. 생략하면 adapter_config.json의 "
        "base_model_name_or_path에서 '-bnb-4bit' 접미사만 제거해 자동 유추한다 "
        "(예: unsloth/gemma-2-2b-bnb-4bit -> unsloth/gemma-2-2b).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
