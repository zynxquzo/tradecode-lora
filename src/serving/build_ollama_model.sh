#!/usr/bin/env bash
# 병합된 HF 모델(outputs/merged)을 GGUF로 변환하고 Ollama에 등록한다.
#
# 사전 조건:
#   - src/finetune/merge_adapter.py 실행 완료 (outputs/merged에 병합 모델 존재)
#   - llama.cpp 클론 완료 (아래 LLAMA_CPP_DIR 지정) - convert_hf_to_gguf.py는 순수
#     Python이라 클론만 하면 되고 빌드는 필요 없다
#   - llama-quantize 실행 파일 - 소스 빌드(cmake) 하거나, Windows에서 cmake/MSVC가
#     없으면 GitHub Releases의 미리 빌드된 바이너리를 받아 LLAMA_QUANTIZE_BIN
#     환경변수로 경로를 지정 (아래 안내 참고)
#   - Ollama 설치 및 `ollama serve` 실행 중
#
# 사용법:
#   bash src/serving/build_ollama_model.sh [merged 모델 디렉토리] [양자화 레벨]
#   예: bash src/serving/build_ollama_model.sh outputs/merged Q4_K_M
#
# Windows에서 cmake/Visual Studio Build Tools 없이 진행하는 방법 (권장):
#   1. https://github.com/ggml-org/llama.cpp/releases/latest 에서
#      llama-<build>-bin-win-cpu-x64.zip (CPU-only, CUDA 불필요) 다운로드
#   2. 압축 풀고 그 안의 llama-quantize.exe 경로를 확인
#   3. LLAMA_QUANTIZE_BIN="C:/path/to/llama-quantize.exe" bash src/serving/build_ollama_model.sh ...
#      (git bash라면 경로에 슬래시 사용, 예: /c/Users/.../llama-quantize.exe)

set -euo pipefail

MERGED_DIR="${1:-outputs/merged}"
QUANT_LEVEL="${2:-Q4_K_M}"   # 로컬 서빙 속도/용량 균형을 위한 선택 (Q4_K_M: 품질 손실 적고 용량 대비 속도 우수)
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-./llama.cpp}"
OUT_DIR="outputs/gguf"
MODEL_NAME="tradecode-gemma2"

mkdir -p "$OUT_DIR"

if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "llama.cpp가 없습니다. 아래 명령으로 먼저 클론하세요 (convert_hf_to_gguf.py는"
  echo "순수 Python이라 빌드 불필요, pip install만 하면 됨):"
  echo "  git clone https://github.com/ggml-org/llama.cpp \"$LLAMA_CPP_DIR\""
  echo "  pip install -r \"$LLAMA_CPP_DIR/requirements.txt\""
  echo "llama-quantize 실행 파일은 위 스크립트 상단 주석의 Windows 미리 빌드 바이너리"
  echo "안내를 참고하거나, cmake로 직접 빌드하세요:"
  echo "  cmake -B \"$LLAMA_CPP_DIR/build\" -S \"$LLAMA_CPP_DIR\" && cmake --build \"$LLAMA_CPP_DIR/build\" --config Release -j"
  exit 1
fi

echo "[1/3] HF -> GGUF (f16) 변환"
# 실패 시 흔한 원인: llama.cpp 버전이 오래되면 gemma2 아키텍처를 인식하지 못함
# ("unknown model architecture: gemma2" 에러) -> llama.cpp를 최신으로 pull 후 재시도.
# Ollama도 자체적으로 gemma2 아키텍처를 지원하는지 버전 확인 필요 (오래된 Ollama는 미지원).
python "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MERGED_DIR" \
  --outfile "$OUT_DIR/${MODEL_NAME}-f16.gguf" \
  --outtype f16

echo "[2/3] 양자화 ($QUANT_LEVEL)"
# LLAMA_QUANTIZE_BIN 환경변수가 있으면 그걸 최우선 사용 (Windows 미리 빌드 바이너리용).
# 없으면 소스 빌드 결과물 경로를 순서대로 탐색한다 (빌드 방식/OS에 따라 다름).
if [ -n "${LLAMA_QUANTIZE_BIN:-}" ]; then
  QUANTIZE_BIN="$LLAMA_QUANTIZE_BIN"
else
  QUANTIZE_BIN=""
  for candidate in \
    "$LLAMA_CPP_DIR/build/bin/llama-quantize" \
    "$LLAMA_CPP_DIR/build/bin/Release/llama-quantize.exe" \
    "$LLAMA_CPP_DIR/build/bin/llama-quantize.exe" \
    "$LLAMA_CPP_DIR/llama-quantize" \
    "$LLAMA_CPP_DIR/llama-quantize.exe"; do
    if [ -x "$candidate" ]; then
      QUANTIZE_BIN="$candidate"
      break
    fi
  done
  if [ -z "$QUANTIZE_BIN" ]; then
    echo "llama-quantize 실행 파일을 찾을 수 없습니다."
    echo "LLAMA_QUANTIZE_BIN 환경변수로 경로를 직접 지정하세요 (스크립트 상단 주석 참고)."
    exit 1
  fi
fi
"$QUANTIZE_BIN" \
  "$OUT_DIR/${MODEL_NAME}-f16.gguf" \
  "$OUT_DIR/${MODEL_NAME}-${QUANT_LEVEL}.gguf" \
  "$QUANT_LEVEL"

echo "[3/3] Ollama Modelfile 작성 + 등록"
MODELFILE="$OUT_DIR/Modelfile"
cat > "$MODELFILE" <<EOF
FROM ./${MODEL_NAME}-${QUANT_LEVEL}.gguf
PARAMETER temperature 0.1
SYSTEM "당신은 HS코드 분류 전문가입니다. JSON 형식으로만 답하세요."
EOF

# Ollama는 Modelfile 내 FROM 경로를 실행 디렉토리 기준 상대경로로 해석하므로
# gguf 파일이 있는 OUT_DIR로 이동해서 create를 실행한다.
(cd "$OUT_DIR" && ollama create "$MODEL_NAME" -f Modelfile)

echo "등록 완료. 아래로 로딩 테스트:"
echo "  ollama run $MODEL_NAME \"Men's cotton knitted T-shirts, short sleeve, crew neck\""
echo ""
echo "재평가는 다음 명령으로 실행 (모델명만 바꿔서 baseline_eval.py 재사용):"
echo "  python src/eval/baseline_eval.py --model $MODEL_NAME --prompt-style finetuned --output docs/03-finetuned_result.md"
