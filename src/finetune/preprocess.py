"""
data/processed/augmented.jsonl(원본+패러프레이징 증강본)을 instruction/input/output
포맷으로 변환하고, 6자리 HS코드 기준 stratified train/eval(80/20) split을 수행하는 스크립트.

augmented.jsonl의 각 라인 형식(augment.py 출력):
  {"description": str, "hs_code": str(6자리), "confidence_basis": str, "is_augmented": bool}

eval 배정 원칙: 같은 클래스 내에서 원본(is_augmented=false)을 증강본보다 우선 배정한다.
증강 데이터로 평가하면 모델이 학습에 쓴 문장과 유사한 문장으로 평가받게 되어
지표가 실제보다 부풀려질 수 있기 때문이다.

사용 예:
  python src/finetune/preprocess.py --input data/processed/augmented.jsonl \
      --output-dir data/processed
"""

import argparse
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

INSTRUCTION_TEXT = "다음 상품설명에 해당하는 HS코드를 6자리까지 추천하고 근거를 설명하세요."

HS_CODE_LEN = 6


def load_augmented_records(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
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
        "output": {
            "hs_code": hs_code,
            "confidence_basis": confidence_basis.strip() if confidence_basis else "",
        },
    }


def convert_records(raw_records: list[dict]) -> tuple[list[dict], int]:
    """augmented 레코드를 스키마 포맷으로 변환. is_augmented 플래그는 별도로 유지해
    split 단계에서 eval 우선순위 결정에 사용한다. (변환 결과, 스킵 개수) 반환."""
    converted = []
    skipped = 0
    for rec in raw_records:
        description = (rec.get("description") or "").strip()
        hs_code = normalize_hs_code(rec.get("hs_code"))
        if not description or not hs_code:
            skipped += 1
            continue
        schema_rec = to_schema_record(description, hs_code, rec.get("confidence_basis", ""))
        schema_rec["_is_augmented"] = bool(rec.get("is_augmented", False))
        converted.append(schema_rec)
    return converted, skipped


def check_class_imbalance(records: list[dict], top_n: int = 10) -> Counter:
    """6자리 HS코드 기준 클래스 분포를 계산하고 쏠림을 로그로 출력."""
    counts = Counter(rec["output"]["hs_code"] for rec in records)
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
                "클래스 불균형 경고: 최다 HS코드(%s)가 전체의 %.1f%%를 차지합니다.",
                most_common[0][0],
                100 * top_share,
            )

    singleton_classes = [code for code, cnt in counts.items() if cnt == 1]
    if singleton_classes:
        logger.warning(
            "샘플이 1개뿐인 HS코드가 %d개 있습니다. 해당 클래스는 train에만 배정됩니다.",
            len(singleton_classes),
        )
    return counts


def stratified_split(
    records: list[dict], eval_ratio: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict], int]:
    """6자리 HS코드 기준 stratified split. 같은 클래스 내에서는 원본
    (_is_augmented=False)을 증강본보다 eval에 우선 배정한다.
    (train, eval, 스킵된 클래스 수) 반환."""
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_class[rec["output"]["hs_code"]].append(rec)

    train, eval_ = [], []
    skipped_classes = 0
    for code, class_records in by_class.items():
        if len(class_records) == 1:
            train.extend(class_records)
            skipped_classes += 1
            continue

        originals = [r for r in class_records if not r["_is_augmented"]]
        augmented = [r for r in class_records if r["_is_augmented"]]
        rng.shuffle(originals)
        rng.shuffle(augmented)

        n_eval = max(1, round(len(class_records) * eval_ratio))
        eval_pool = originals + augmented  # 원본 우선 배정
        eval_.extend(eval_pool[:n_eval])
        train.extend(eval_pool[n_eval:])

    rng.shuffle(train)
    rng.shuffle(eval_)
    return train, eval_, skipped_classes


def strip_internal_fields(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in rec.items() if k != "_is_augmented"} for rec in records]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(input_path: Path, output_dir: Path, eval_ratio: float = 0.2, seed: int = 42) -> None:
    logger.info("증강 데이터 로드: %s", input_path)
    raw_records = load_augmented_records(input_path)
    logger.info("원본 레코드 수: %d", len(raw_records))

    converted, skipped = convert_records(raw_records)
    logger.info("변환 완료: %d건 성공, %d건 스킵(description/hs_code 누락)", len(converted), skipped)

    if not converted:
        logger.error("변환된 레코드가 없습니다. augmented.jsonl 형식을 확인하세요.")
        return

    check_class_imbalance(converted)

    train, eval_, skipped_classes = stratified_split(converted, eval_ratio=eval_ratio, seed=seed)
    logger.info(
        "split 완료: train=%d건, eval=%d건, 샘플 1개라 split 불가했던 클래스=%d개",
        len(train),
        len(eval_),
        skipped_classes,
    )
    eval_original_ratio = (
        sum(1 for r in eval_ if not r["_is_augmented"]) / len(eval_) if eval_ else 0.0
    )
    logger.info("eval셋 중 원본 비율: %.1f%%", 100 * eval_original_ratio)

    write_jsonl(strip_internal_fields(train), output_dir / "train.jsonl")
    write_jsonl(strip_internal_fields(eval_), output_dir / "eval.jsonl")
    logger.info("저장 완료: %s, %s", output_dir / "train.jsonl", output_dir / "eval.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data/processed/augmented.jsonl"), help="augmented jsonl 경로"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"), help="출력 디렉토리"
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="eval 비율 (기본 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="셔플 시드")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir, args.eval_ratio, args.seed)
