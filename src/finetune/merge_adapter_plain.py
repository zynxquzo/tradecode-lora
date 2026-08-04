"""
merge_adapter.py의 대안 경로 — unsloth의 save_pretrained_merged() 대신 순정
transformers + peft.PeftModel.merge_and_unload()로 병합한다.

배경(실험 3에서 확인된 사실들, docs/10-experiment3_investigation.md 7~9절 참고):
- 처음엔 unsloth의 save_pretrained_merged()가 가중치를 손상시킨다고 의심했으나,
  병합을 아예 거치지 않고 adapter만 얹어도 같은 증상(특정 희귀 토큰만 반복 생성)이
  재현되어 그 가설은 기각됐다.
- unsloth 자체의 로드 경로(FastLanguageModel.from_pretrained + for_inference)로는
  정상 생성됐는데, 이 테스트는 베이스로 학습에 실제 쓰인 4bit 체크포인트
  (unsloth/gemma-2-2b-bnb-4bit)를 그대로 썼다.
- 반면 실패한 모든 순정 transformers/GGUF 테스트는 별도의 fp16 리포
  (unsloth/gemma-2-2b, "-bnb-4bit" 접미사만 뗀 것)를 베이스로 썼다 — 이 두 체크포인트가
  진짜로 동일한 가중치인지 검증한 적이 없었다. GGUF 변환본에서도 동일 증상이 재현되어
  "llama.cpp는 이 문제에서 자유로울 것"이라는 가설도 기각됐으므로, 남은 유력한 원인은
  fp16 대체 베이스가 학습에 쓴 4bit 베이스와 미묘하게 다른 체크포인트라는 것이다.

그래서 이 스크립트는 별도 fp16 리포로 바꾸지 않고, 학습에 실제로 쓰인 4bit 베이스
그대로에 병합한 뒤, peft가 재양자화한 4bit 가중치를 bitsandbytes로 직접
역양자화(dequantize)해서 순수 fp16 state_dict를 만들어 저장한다. (peft의
Linear4bit.merge()는 병합 결과를 다시 4bit로 양자화해서 보관하는데, 이 상태를 그대로
save_pretrained()하면 최신 transformers의 revert_weight_conversion()이 처리하지
못해 NotImplementedError가 난다 — 그래서 저장 전에 수동으로 역양자화가 필요하다.)

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

    import bitsandbytes.functional as bnb_functional
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = args.adapter_dir / "adapter_config.json"
    base_model_name = json.loads(adapter_config_path.read_text())["base_model_name_or_path"]
    logger.info("베이스 모델(학습에 실제로 쓰인 4bit 체크포인트 그대로): %s", base_model_name)

    logger.info("베이스 모델 로드 (순정 transformers, 4bit)")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    logger.info("adapter 로드 및 병합: %s", args.adapter_dir)
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model = model.merge_and_unload()

    logger.info("병합 결과 역양자화 (4bit -> fp16)")
    merged_state_dict = model.state_dict()
    fp16_state_dict = {}
    dequantized_count = 0
    for name, param in merged_state_dict.items():
        quant_state = getattr(param, "quant_state", None)
        if quant_state is not None:
            fp16_state_dict[name] = bnb_functional.dequantize_4bit(param.data, quant_state).to(torch.float16)
            dequantized_count += 1
        else:
            fp16_state_dict[name] = param.to(torch.float16)
    logger.info("역양자화한 파라미터 수: %d / 전체 %d", dequantized_count, len(fp16_state_dict))

    logger.info("역양자화된 가중치로 순수 fp16 모델 재구성")
    config = AutoConfig.from_pretrained(base_model_name)
    # 베이스 config에는 quantization_config가 남아있어 그대로 두면 다시 4bit로 로드를
    # 시도하니, 순수 fp16 모델임을 명시하기 위해 제거한다.
    config.quantization_config = None
    fp16_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    fp16_model.load_state_dict(fp16_state_dict, strict=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("병합 모델 저장: %s", args.output_dir)
    fp16_model.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))

    # AutoTokenizer.save_pretrained()가 fast tokenizer 파일만 저장하고 sentencepiece
    # 원본(tokenizer.model)은 빠뜨리는 경우가 있다 - convert_hf_to_gguf.py의 gemma
    # 컨버터가 이 파일을 직접 요구하므로, adapter_dir에 저장돼 있던 것을 그대로 복사한다.
    adapter_tokenizer_model = args.adapter_dir / "tokenizer.model"
    output_tokenizer_model = args.output_dir / "tokenizer.model"
    if adapter_tokenizer_model.exists() and not output_tokenizer_model.exists():
        output_tokenizer_model.write_bytes(adapter_tokenizer_model.read_bytes())
        logger.info("tokenizer.model 보완 완료: %s", output_tokenizer_model)

    logger.info("병합 완료, 저장 경로: %s", args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-dir", type=Path, default=Path("outputs/adapter"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/merged_plain"))
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
