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

그래서 이 스크립트는 별도 fp16 리포로 바꾸지 않고, 학습에 실제로 쓰인 4bit 체크포인트
그대로를 읽되(그 파일이 4bit로 저장돼 있어 읽는 것 자체는 bnb 없이는 불가능하다),
**LoRA를 얹기 전에 먼저 역양자화**한다. peft의 Linear4bit.merge()는 병합 결과를 다시
4bit로 양자화해서 보관하기 때문에(그 상태로 save_pretrained()하면 최신 transformers의
revert_weight_conversion()이 처리하지 못해 NotImplementedError가 난다), 병합을 4bit
공간에서 하지 않도록 순서를 바꿔 이 문제를 원천적으로 피한다: 베이스를 fp16으로
역양자화 -> 순정 nn.Linear로 재구성 -> 그 위에 LoRA를 얹고 병합(평범한 fp16 덧셈,
재양자화 없음) -> 저장.

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
    from bitsandbytes.nn import Linear4bit
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    adapter_config_path = args.adapter_dir / "adapter_config.json"
    base_model_name = json.loads(adapter_config_path.read_text())["base_model_name_or_path"]
    logger.info("베이스 모델(학습에 실제로 쓰인 4bit 체크포인트 그대로): %s", base_model_name)

    logger.info("베이스 모델 로드 (4bit, 파일이 이 형식으로만 저장돼 있어 읽기 위해 필요)")
    quantized_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    logger.info("베이스를 fp16으로 역양자화 (LoRA 병합을 4bit 공간에서 하지 않기 위해)")
    # model.state_dict()를 거치면 bnb.nn.Params4bit의 quant_state 속성이 유실되고
    # 4bit로 패킹된 원본 바이트가 그대로 나온다(shape가 [N, 1]처럼 찌그러져 있음) -
    # quant_state는 살아있는 모듈 객체(module.weight)에서 직접 읽어야 한다.
    fp16_state_dict = {}
    dequantized_count = 0
    for name, module in quantized_model.named_modules():
        if isinstance(module, Linear4bit):
            weight_key = f"{name}.weight" if name else "weight"
            fp16_state_dict[weight_key] = bnb_functional.dequantize_4bit(
                module.weight.data, module.weight.quant_state
            ).to(torch.float16)
            dequantized_count += 1
            if module.bias is not None:
                fp16_state_dict[f"{name}.bias"] = module.bias.data.to(torch.float16)
    for name, param in quantized_model.named_parameters():
        if name not in fp16_state_dict:
            fp16_state_dict[name] = param.data.to(torch.float16)

    # Gemma-2는 lm_head가 embed_tokens와 가중치를 공유(tie_word_embeddings=True)한다 -
    # named_parameters()는 기본적으로 동일 텐서를 가리키는 중복 키를 하나만 남기고
    # 건너뛰므로, 두 키 중 하나가 fp16_state_dict에서 빠져 있을 수 있다. 로드할
    # 모델도 같은 config(tie_word_embeddings=True)라 두 키 모두 필요하니 채워 넣는다.
    if "lm_head.weight" not in fp16_state_dict and "model.embed_tokens.weight" in fp16_state_dict:
        fp16_state_dict["lm_head.weight"] = fp16_state_dict["model.embed_tokens.weight"]
    elif "model.embed_tokens.weight" not in fp16_state_dict and "lm_head.weight" in fp16_state_dict:
        fp16_state_dict["model.embed_tokens.weight"] = fp16_state_dict["lm_head.weight"]

    logger.info(
        "역양자화한 Linear4bit 모듈 수: %d / 전체 유지 파라미터: %d",
        dequantized_count, len(fp16_state_dict),
    )

    config = AutoConfig.from_pretrained(base_model_name)
    # 베이스 config에는 quantization_config가 남아있어 그대로 두면(None으로만 덮어써도
    # 속성 자체는 남아 config.json에 "quantization_config": null로 직렬화된다) 나중에
    # 이 모델을 다시 불러올 때 transformers가 "사전 양자화된 모델"로 오인해
    # AutoHfQuantizer.supports_quant_method(None)에서 AttributeError로 죽는다.
    # 속성 자체를 지워서 config.json에 아예 안 남도록 한다.
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    del quantized_model
    torch.cuda.empty_cache()

    base_model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)
    base_model.load_state_dict(fp16_state_dict, strict=True)
    base_model = base_model.to("cuda" if torch.cuda.is_available() else "cpu")
    del fp16_state_dict

    logger.info("adapter 로드 및 병합 (순정 fp16 nn.Linear 위에서, 재양자화 없이)")
    model = PeftModel.from_pretrained(base_model, str(args.adapter_dir))
    model = model.merge_and_unload()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("병합 모델 저장: %s", args.output_dir)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
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
