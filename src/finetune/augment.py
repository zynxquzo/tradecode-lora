"""
data/raw/products_real.csv의 관세율표 법령 문구 description을 실무형(인보이스/무역서류)
문체로 패러프레이징하여 클래스당 샘플 수를 늘리는 증강 스크립트.

핵심 제약: 패러프레이징은 문체만 바꾸고, HS코드 분류를 결정짓는 속성
(소재/섬유, 편물(knitted) vs 직물(woven), 성별/연령 구분 등)은 절대 바꾸지 않는다.
같은 라벨(hs_code)을 유지한 채 문장만 바뀌어야 학습 데이터로서 유효하기 때문이다.

사전 조건: 환경변수 OPENAI_API_KEY 설정 필요.

사용 예 (5개만 테스트):
  python src/finetune/augment.py --input data/raw/products_real.csv \
      --output data/processed/augmented_test.jsonl --limit 5

전체 실행:
  python src/finetune/augment.py --input data/raw/products_real.csv \
      --output data/processed/augmented.jsonl
"""

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

import openai

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"

PARAPHRASE_PROMPT_TEMPLATE = """You are helping build training data for an HS code \
(commodity classification) text classifier.

Below is a product description written in formal customs tariff (legal) language:

"{description}"

Rewrite it {n} different times in the casual, plain style used in real trade \
invoices, packing lists, or e-commerce product titles. Each rewrite must be a \
single short line (no explanations, no HS codes).

STRICT RULE: You must preserve every detail that affects commodity classification \
— material/fiber content (e.g. cotton, synthetic, wool), construction \
(knitted/crocheted vs woven), gender/age category (men's, women's, boys', girls', \
unisex), and any other technical qualifier present in the original text. Do NOT \
drop, add, or change these details. Only change wording, tone, and sentence \
structure.

Respond with ONLY a JSON array of {n} strings, nothing else. Example format:
["rewrite 1", "rewrite 2", "rewrite 3"]
"""


def load_raw_rows(input_path: Path) -> list[dict]:
    with open(input_path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def call_openai_paraphrase(
    client: openai.OpenAI, model: str, description: str, n: int, max_retries: int = 5
) -> list[str]:
    prompt = PARAPHRASE_PROMPT_TEMPLATE.format(description=description, n=n)
    delay = 2.0
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content.strip()
            start, end = text.find("["), text.rfind("]")
            if start == -1 or end == -1:
                raise ValueError(f"JSON 배열을 찾을 수 없음: {text[:200]}")
            paraphrases = json.loads(text[start : end + 1])
            if not isinstance(paraphrases, list) or not all(
                isinstance(p, str) for p in paraphrases
            ):
                raise ValueError(f"예상치 못한 응답 형식: {text[:200]}")
            return paraphrases
        except (openai.RateLimitError, openai.APIStatusError, openai.APIConnectionError) as e:
            last_error = e
            logger.warning(
                "API 오류 (시도 %d/%d), %.1f초 후 재시도: %s", attempt, max_retries, delay, e
            )
            time.sleep(delay)
            delay *= 2
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            logger.warning(
                "응답 파싱 실패 (시도 %d/%d), %.1f초 후 재시도: %s", attempt, max_retries, delay, e
            )
            time.sleep(delay)
            delay *= 2
    logger.error("최대 재시도 초과, 이 행은 증강 스킵: %s", last_error)
    return []


def augment(
    rows: list[dict],
    client: openai.OpenAI,
    model: str,
    n_per_item: int,
    sleep_between_calls: float,
) -> list[dict]:
    output_records = []
    for i, row in enumerate(rows, start=1):
        description = row["description"].strip()
        hs_code = row["hs_code"].strip()
        confidence_basis = row.get("confidence_basis", "").strip()

        output_records.append(
            {
                "description": description,
                "hs_code": hs_code,
                "confidence_basis": confidence_basis,
                "is_augmented": False,
            }
        )

        paraphrases = call_openai_paraphrase(client, model, description, n_per_item)
        logger.info("[%d/%d] hs_code=%s 원본 1건 + 증강 %d건", i, len(rows), hs_code, len(paraphrases))
        for p in paraphrases:
            p = p.strip()
            if not p:
                continue
            output_records.append(
                {
                    "description": p,
                    "hs_code": hs_code,
                    "confidence_basis": confidence_basis,
                    "is_augmented": True,
                }
            )

        if sleep_between_calls > 0:
            time.sleep(sleep_between_calls)

    return output_records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/raw/products_real.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/augmented.jsonl"))
    parser.add_argument("--n-per-item", type=int, default=3, help="원본 1건당 생성할 패러프레이즈 수")
    parser.add_argument("--limit", type=int, default=None, help="테스트용: 앞에서 N개 행만 처리")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument(
        "--sleep-between-calls", type=float, default=0.2, help="API 호출 간 대기(초), rate limit 완화용"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rows = load_raw_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    logger.info("증강 대상: %d행, 모델: %s, 행당 증강 수: %d", len(rows), args.model, args.n_per_item)

    client = openai.OpenAI()
    records = augment(rows, client, args.model, args.n_per_item, args.sleep_between_calls)

    n_original = sum(1 for r in records if not r["is_augmented"])
    n_augmented = sum(1 for r in records if r["is_augmented"])
    logger.info("증강 완료: 원본 %d건 + 증강 %d건 = 총 %d건", n_original, n_augmented, len(records))

    write_jsonl(records, args.output)
    logger.info("저장 완료: %s", args.output)
