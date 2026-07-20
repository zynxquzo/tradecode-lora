"""
파인튜닝 전 gemma2:2b(Ollama 로컬 서빙 가정)에 zero-shot 프롬프트로 HS코드 Top-3
추천을 요청하고, Exact/Partial/Top-3 Recall 지표를 계산해 docs/baseline_result.md로
저장하는 스크립트.

사전 조건: `ollama serve` 실행 중이고 `ollama pull gemma2:2b` 완료된 상태.

사용 예:
  python src/eval/baseline_eval.py --eval-file data/processed/eval.jsonl \
      --model gemma2:2b --output docs/baseline_result.md
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
    """모델 응답에서 hs_code 값들을 최대한 관대하게 추출."""
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
        # JSON 파싱 실패 시 텍스트에서 6자리 숫자 패턴을 직접 추출
        codes = re.findall(r"\b\d{6}\b", raw_response)

    return codes[:3]


def predict(model: str, description: str) -> list[str]:
    prompt = ZERO_SHOT_PROMPT_TEMPLATE.format(description=description)
    raw_response = call_ollama(model, prompt)
    return extract_hs_codes(raw_response)


def is_match(pred_code: str, true_code: str, digits: int) -> bool:
    if len(pred_code) < digits or len(true_code) < digits:
        return False
    return pred_code[:digits] == true_code[:digits]


def evaluate(records: list[dict], model: str) -> dict:
    n = len(records)
    exact_match = 0  # 6자리 완전일치 (top-1)
    partial_4 = 0  # 4자리 일치 (top-1)
    partial_2 = 0  # 2자리 일치 (top-1)
    top3_recall = 0  # true code가 top-3 예측 중 하나와 6자리 완전일치

    per_record_results = []

    for i, rec in enumerate(records, start=1):
        description = rec["input"]
        true_code = rec["output"]["hs_code"]

        try:
            preds = predict(model, description)
        except requests.RequestException as e:
            logger.error("Ollama 호출 실패 (레코드 %d/%d): %s", i, n, e)
            preds = []

        top1 = preds[0] if preds else ""

        rec_exact = is_match(top1, true_code, 6)
        rec_partial4 = is_match(top1, true_code, 4)
        rec_partial2 = is_match(top1, true_code, 2)
        rec_top3 = any(is_match(p, true_code, 6) for p in preds)

        exact_match += rec_exact
        partial_4 += rec_partial4
        partial_2 += rec_partial2
        top3_recall += rec_top3

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
        "n_samples": n,
        "exact_match": exact_match / n if n else 0.0,
        "partial_match_4digit": partial_4 / n if n else 0.0,
        "partial_match_2digit": partial_2 / n if n else 0.0,
        "top3_recall": top3_recall / n if n else 0.0,
        "per_record_results": per_record_results,
    }


def save_markdown_report(metrics: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    lines = [
        "# Baseline Evaluation Result",
        "",
        f"- Model: `{metrics['model']}`",
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
        "--output", type=Path, default=Path("docs/baseline_result.md"), help="결과 저장 경로"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    eval_records = load_eval_records(args.eval_file)
    logger.info("eval 레코드 수: %d, 모델: %s", len(eval_records), args.model)
    result_metrics = evaluate(eval_records, args.model)
    save_markdown_report(result_metrics, args.output)
