"""
Ollama로 서빙 중인 모델에 HS코드 Top-3 추천을 요청하고, Exact/Partial/Top-3 Recall
지표를 계산해 마크다운 리포트로 저장하는 스크립트. --model과 --prompt-style만 바꾸면
baseline(zero-shot gemma2:2b)과 fine-tuned 모델 모두 동일 스크립트로 평가할 수 있다.

프롬프트 스타일 두 가지:
  - zero_shot  : baseline용. Top-3 JSON 배열을 요구하는 프롬프트 (파인튜닝 안 한 모델용).
  - finetuned  : src/finetune/train.py가 학습에 사용한 것과 동일한 Alpaca 스타일
                 프롬프트. 파인튜닝된 모델은 이 스타일로 물어야 학습 때 익힌 출력
                 포맷(JSON 객체 1개)을 그대로 낸다. train.py의 PROMPT_TEMPLATE을 바꾸면
                 여기 PROMPT_TEMPLATES["finetuned"]도 같이 바꿀 것.

사전 조건: `ollama serve` 실행 중이고 대상 모델이 `ollama pull`/`ollama create`로
준비된 상태.

사용 예 (baseline):
  python src/eval/baseline_eval.py --eval-file data/processed/eval.jsonl \
      --model gemma2:2b --prompt-style zero_shot --output docs/01-baseline_result.md

사용 예 (fine-tuned, 먼저 100건만 빠르게 확인):
  python src/eval/baseline_eval.py --model tradecode-gemma2 --prompt-style finetuned \
      --limit 100 --output docs/03-finetuned_result_partial.md
"""

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

ZERO_SHOT_PROMPT_TEMPLATE = """당신은 무역 품목분류 전문가입니다. 아래 상품설명을 보고 \
가장 적절한 HS코드(6자리)를 Top-3로 추천하세요.

반드시 아래 JSON 형식으로만 답하세요. 다른 설명은 추가하지 마세요:
[
  {{"hs_code": "6자리코드", "confidence_basis": "분류 근거"}},
  {{"hs_code": "6자리코드", "confidence_basis": "분류 근거"}},
  {{"hs_code": "6자리코드", "confidence_basis": "분류 근거"}}
]

상품설명: {description}
"""

# src/finetune/train.py의 학습 프롬프트(PROMPT_TEMPLATE)와 반드시 동일한 형식이어야
# 파인튜닝된 모델이 학습 때 익힌 출력을 낸다. instruction 문구는 data/processed의
# 스키마(preprocess.py의 INSTRUCTION_TEXT)와 일치시켰다.
FINETUNED_PROMPT_TEMPLATE = """### Instruction:
다음 상품설명에 해당하는 HS코드를 6자리까지 추천하고 근거를 설명하세요.

### Input:
{description}

### Response:
"""

PROMPT_TEMPLATES = {
    "zero_shot": ZERO_SHOT_PROMPT_TEMPLATE,
    "finetuned": FINETUNED_PROMPT_TEMPLATE,
}


def load_eval_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def call_ollama(model: str, prompt: str, timeout: int = 60) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def extract_hs_codes(raw_response: str) -> list[str]:
    """모델 응답에서 hs_code 값들을 최대한 관대하게 추출.
    Top-3 JSON 배열([{...}, ...])과 fine-tuned 모델의 단일 JSON 객체({...}) 둘 다 지원."""
    codes: list[str] = []
    try:
        match = re.search(r"\[.*\]", raw_response, re.DOTALL)
        if match:
            parsed = json.loads(match.group(0))
            for item in parsed:
                code = "".join(ch for ch in str(item.get("hs_code", "")) if ch.isdigit())
                if code:
                    codes.append(code)
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    if not codes:
        # 배열 파싱 실패 시 단일 JSON 객체({"hs_code": ...})로 재시도
        try:
            match = re.search(r"\{.*\}", raw_response, re.DOTALL)
            if match:
                item = json.loads(match.group(0))
                code = "".join(ch for ch in str(item.get("hs_code", "")) if ch.isdigit())
                if code:
                    codes.append(code)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    if not codes:
        # JSON 파싱 실패 시 텍스트에서 6자리 숫자 패턴을 직접 추출
        codes = re.findall(r"\b\d{6}\b", raw_response)

    return codes[:3]


def predict(model: str, description: str, prompt_style: str) -> list[str]:
    template = PROMPT_TEMPLATES[prompt_style]
    prompt = template.format(description=description)
    raw_response = call_ollama(model, prompt)
    return extract_hs_codes(raw_response)


def is_match(pred_code: str, true_code: str, digits: int) -> bool:
    if len(pred_code) < digits or len(true_code) < digits:
        return False
    return pred_code[:digits] == true_code[:digits]


def evaluate(records: list[dict], model: str, prompt_style: str) -> dict:
    n = len(records)
    exact_match = 0  # 6자리 완전일치 (top-1)
    partial_4 = 0  # 4자리 일치 (top-1)
    partial_2 = 0  # 2자리 일치 (top-1)
    top3_recall = 0  # true code가 top-3 예측 중 하나와 6자리 완전일치
    parse_failures = 0  # 예측 코드를 하나도 추출하지 못한 케이스

    per_record_results = []

    for i, rec in enumerate(records, start=1):
        description = rec["input"]
        true_code = rec["output"]["hs_code"]

        try:
            preds = predict(model, description, prompt_style)
        except requests.RequestException as e:
            logger.error("Ollama 호출 실패 (레코드 %d/%d): %s", i, n, e)
            preds = []

        top1 = preds[0] if preds else ""

        rec_exact = is_match(top1, true_code, 6)
        rec_partial4 = is_match(top1, true_code, 4)
        rec_partial2 = is_match(top1, true_code, 2)
        rec_top3 = any(is_match(p, true_code, 6) for p in preds)
        rec_parse_failure = len(preds) == 0

        exact_match += rec_exact
        partial_4 += rec_partial4
        partial_2 += rec_partial2
        top3_recall += rec_top3
        parse_failures += rec_parse_failure

        per_record_results.append(
            {
                "input": description,
                "true_code": true_code,
                "preds": preds,
                "exact_match": rec_exact,
                "top3_recall": rec_top3,
            }
        )
        logger.info(
            "[%d/%d] true=%s preds=%s exact=%s top3=%s",
            i,
            n,
            true_code,
            preds,
            rec_exact,
            rec_top3,
        )

    return {
        "model": model,
        "prompt_style": prompt_style,
        "n_samples": n,
        "exact_match": exact_match / n if n else 0.0,
        "partial_match_4digit": partial_4 / n if n else 0.0,
        "partial_match_2digit": partial_2 / n if n else 0.0,
        "top3_recall": top3_recall / n if n else 0.0,
        "parse_failure_rate": parse_failures / n if n else 0.0,
        "per_record_results": per_record_results,
    }


def save_markdown_report(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines = [
        "# Evaluation Result",
        "",
        f"- Model: `{metrics['model']}`",
        f"- Prompt style: `{metrics['prompt_style']}`",
        f"- Samples: {metrics['n_samples']}",
        f"- Generated at: {timestamp}",
        "",
        "## Metrics",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Exact Match (6-digit) | {metrics['exact_match']:.2%} |",
        f"| Partial Match (4-digit) | {metrics['partial_match_4digit']:.2%} |",
        f"| Partial Match (2-digit) | {metrics['partial_match_2digit']:.2%} |",
        f"| Top-3 Recall | {metrics['top3_recall']:.2%} |",
        f"| Parse Failure Rate | {metrics['parse_failure_rate']:.2%} |",
        "",
        "## Sample Predictions",
        "",
        "| Input | True | Predicted | Exact | Top-3 |",
        "|---|---|---|---|---|",
    ]
    for r in metrics["per_record_results"][:20]:
        input_snippet = r["input"][:50].replace("|", "\\|")
        preds_str = ", ".join(r["preds"]) if r["preds"] else "(none)"
        lines.append(
            f"| {input_snippet} | {r['true_code']} | {preds_str} "
            f"| {'✅' if r['exact_match'] else '❌'} | {'✅' if r['top3_recall'] else '❌'} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("결과 저장 완료: %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-file", type=Path, default=Path("data/processed/eval.jsonl"), help="eval jsonl 경로"
    )
    parser.add_argument("--model", type=str, default="gemma2:2b", help="Ollama 모델명")
    parser.add_argument(
        "--prompt-style",
        type=str,
        default="zero_shot",
        choices=list(PROMPT_TEMPLATES.keys()),
        help="zero_shot: baseline용 Top-3 프롬프트 / finetuned: train.py와 동일한 학습 프롬프트",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="앞에서 N건만 빠르게 평가 (재평가가 오래 걸릴 때 1차 확인용, 예: 100)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("docs/baseline_result.md"), help="결과 저장 경로"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_records = load_eval_records(args.eval_file)
    if args.limit:
        eval_records = eval_records[: args.limit]
    logger.info(
        "eval 레코드 수: %d, 모델: %s, prompt_style: %s", len(eval_records), args.model, args.prompt_style
    )
    result_metrics = evaluate(eval_records, args.model, args.prompt_style)
    save_markdown_report(result_metrics, args.output)
