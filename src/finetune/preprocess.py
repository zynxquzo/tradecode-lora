"""
raw 상품설명 데이터(csv/json)를 instruction/input/output 포맷의 jsonl로 변환하고
6자리 HS코드 기준 stratified split(train/eval, 80/20)을 수행하는 스크립트.

기대하는 raw 데이터 컬럼(csv) 또는 키(json list of dict):
  - description (str): 상품설명 (영문 또는 국문)
  - hs_code (str): 6자리 HS코드 (숫자만, 예: "610910")
  - confidence_basis (str, optional): 분류 근거. 없으면 빈 문자열로 채움.

사용 예:
  python src/finetune/preprocess.py --input data/raw/products.csv --output-dir data/processed
"""

import argparse
import csv
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

INSTRUCTION_TEXT = (
    "다음 상품설명을 보고 가장 적절한 HS코드(6자리)를 Top-3로 추천하고, "
    "각 코드에 대한 분류 근거를 설명하세요."
)

HS_CODE_LEN = 6


def load_raw_records(input_path: Path) -> list[dict]:
    if input_path.suffix.lower() == ".csv":
        with open(input_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            records = list(reader)
    elif input_path.suffix.lower() == ".json":
        with open(input_path, encoding="utf-8") as f:
            records = json.load(f)
        if not isinstance(records, list):
            raise ValueError("JSON 입력은 레코드 리스트(list of dict) 형태여야 합니다.")
    else:
        raise ValueError(f"지원하지 않는 확장자: {input_path.suffix} (csv 또는 json만 지원)")
    return records


def normalize_hs_code(raw_code: str) -> str | None:
    """HS코드를 숫자만 남긴 6자리 문자열로 정규화. 유효하지 않으면 None."""
    if raw_code is None:
        return None
    digits = "".join(ch for ch in str(raw_code) if ch.isdigit())
    if len(digits) < HS_CODE_LEN:
        return None
    return digits[:HS_CODE_LEN]


def to_schema_record(description: str, hs_code: str, confidence_basis: str) -> dict:
    return {
        "instruction": INSTRUCTION_TEXT,
        "input": description.strip(),
        "output": [
            {
                "hs_code": hs_code,
                "confidence_basis": confidence_basis.strip() if confidence_basis else "",
            }
        ],
    }


def convert_records(raw_records: list[dict]) -> tuple[list[dict], int]:
    """raw 레코드를 스키마 포맷으로 변환. (변환된 레코드, 스킵된 개수) 반환."""
    converted = []
    skipped = 0
    for rec in raw_records:
        description = (rec.get("description") or "").strip()
        hs_code = normalize_hs_code(rec.get("hs_code"))
        if not description or not hs_code:
            skipped += 1
            continue
        confidence_basis = rec.get("confidence_basis", "")
        converted.append(to_schema_record(description, hs_code, confidence_basis))
    return converted, skipped


def check_class_imbalance(records: list[dict], top_n: int = 10) -> Counter:
    """6자리 HS코드 기준 클래스 분포를 계산하고 쏠림을 로그로 출력."""
    counts = Counter(rec["output"][0]["hs_code"] for rec in records)
    total = sum(counts.values())
    logger.info("클래스(HS코드) 종류 수: %d, 전체 샘플 수: %d", len(counts), total)

    most_common = counts.most_common(top_n)
    logger.info("상위 %d개 HS코드 분포:", top_n)
    for code, cnt in most_common:
        logger.info("  %s: %d건 (%.1f%%)", code, cnt, 100 * cnt / total)

    if most_common:
        top_share = most_common[0][1] / total
        if top_share > 0.3:
            logger.warning(
                "클래스 불균형 경고: 최다 HS코드(%s)가 전체의 %.1f%%를 차지합니다. "
                "샘플링/가중치 보정을 고려하세요.",
                most_common[0][0],
                100 * top_share,
            )

    singleton_classes = [code for code, cnt in counts.items() if cnt == 1]
    if singleton_classes:
        logger.warning(
            "샘플이 1개뿐인 HS코드가 %d개 있습니다. stratified split 시 해당 클래스는 "
            "train에만 배정됩니다 (eval 배정 불가).",
            len(singleton_classes),
        )
    return counts


def stratified_split(
    records: list[dict], eval_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """6자리 HS코드 기준으로 클래스별 80/20 stratified split."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_class[rec["output"][0]["hs_code"]].append(rec)

    train, eval_ = [], []
    for code, class_records in by_class.items():
        rng.shuffle(class_records)
        if len(class_records) == 1:
            # 샘플이 1개뿐이면 eval로 나눌 수 없으므로 train에 배정
            train.extend(class_records)
            continue
        n_eval = max(1, round(len(class_records) * eval_ratio))
        eval_.extend(class_records[:n_eval])
        train.extend(class_records[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(input_path: Path, output_dir: Path, eval_ratio: float = 0.2, seed: int = 42) -> None:
    logger.info("raw 데이터 로드: %s", input_path)
    raw_records = load_raw_records(input_path)
    logger.info("raw 레코드 수: %d", len(raw_records))

    converted, skipped = convert_records(raw_records)
    logger.info("변환 완료: %d건 성공, %d건 스킵(description/hs_code 누락)", len(converted), skipped)

    if not converted:
        logger.error("변환된 레코드가 없습니다. raw 데이터 형식을 확인하세요.")
        return

    check_class_imbalance(converted)

    train, eval_ = stratified_split(converted, eval_ratio=eval_ratio, seed=seed)
    logger.info("split 완료: train=%d건, eval=%d건", len(train), len(eval_))

    write_jsonl(train, output_dir / "train.jsonl")
    write_jsonl(eval_, output_dir / "eval.jsonl")
    logger.info("저장 완료: %s, %s", output_dir / "train.jsonl", output_dir / "eval.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="raw 데이터 경로 (csv 또는 json)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"), help="출력 디렉토리"
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="eval 비율 (기본 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="셔플 시드")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir, args.eval_ratio, args.seed)
