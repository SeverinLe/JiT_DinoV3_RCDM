#!/usr/bin/env bash
# =============================================================================
# Download DINOv3 ViT-S/16 pretrained weights for use with RETFound.
#
# OPTION A — Official Meta URL (via GitHub access form)
#   bash scripts/download_dinov3_vits16.sh --url "SIGNED_URL_FROM_EMAIL"
#
# OPTION B — Hugging Face Hub
#   1. Open https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
#      log in, submit the form, and wait until access shows as granted.
#   2. Token must allow gated repos:
#      • Classic token: “Read” is enough.
#      • Fine-grained token: turn ON “Access to public gated repositories”
#        (HF Settings → Access Tokens → edit token → Repository permissions).
#   3. Either:
#        export HF_TOKEN=hf_...
#        bash scripts/download_dinov3_vits16.sh --hf
#      or:
#        bash scripts/download_dinov3_vits16.sh --hf --token hf_...
# =============================================================================

set -euo pipefail

OUTDIR="$(cd "$(dirname "$0")/.." && pwd)/checkpoints"
OUTFILE="${OUTDIR}/dinov3_vits16_pretrain.pth"
HF_REPO="facebook/dinov3-vits16-pretrain-lvd1689m"
TMP_DIR="${OUTDIR}/_dinov3_tmp"

hf_troubleshoot() {
  echo ""
  echo "──────────────── Hugging Face download failed ────────────────"
  echo "If you see 403 Forbidden / “public gated repositories”:"
  echo "  • Accept the model license on the model page while logged in."
  echo "  • For a fine-grained token: enable read access to *public gated*"
  echo "    repos (token settings on huggingface.co), or use a classic token."
  echo "  • Pass a token explicitly:  HF_TOKEN=hf_... $0 --hf"
  echo "     or:  $0 --hf --token hf_..."
  echo "────────────────────────────────────────────────────────────"
  exit 1
}

mkdir -p "${OUTDIR}"

usage() {
  echo "Usage:"
  echo "  $0 --url \"SIGNED_URL\"              # Meta e-mail link (wget)"
  echo "  $0 --hf [--token HF_TOKEN]         # Hugging Face (needs gated access)"
  echo ""
  echo "Env: HF_TOKEN or HUGGING_FACE_HUB_TOKEN (optional if already logged in)."
  exit 1
}

MODE=""
URL=""
TOKEN_ARG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url)
      [[ -n "${2:-}" ]] || { echo "Error: --url needs a URL"; exit 1; }
      MODE=url
      URL="$2"
      shift 2
      ;;
    --hf)
      MODE=hf
      shift
      ;;
    --token)
      [[ -n "${2:-}" ]] || { echo "Error: --token needs a value"; exit 1; }
      TOKEN_ARG="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

[[ -n "${MODE:-}" ]] || usage

TOKEN="${TOKEN_ARG:-${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"

case "$MODE" in
  url)
    echo "Downloading DINOv3 ViT-S/16 weights from signed URL..."
    wget -q --show-progress -O "${OUTFILE}" "$URL"
    echo "Saved to: ${OUTFILE}"
    ;;

  hf)
    echo "Downloading DINOv3 ViT-S/16 from Hugging Face Hub..."
    rm -rf "${TMP_DIR}"
    mkdir -p "${TMP_DIR}"

    set +e
    if [[ -n "$TOKEN" ]]; then
      huggingface-cli download "${HF_REPO}" model.safetensors \
        --local-dir "${TMP_DIR}" --token "$TOKEN" 2>&1
      DL_STAT=$?
    else
      huggingface-cli download "${HF_REPO}" model.safetensors \
        --local-dir "${TMP_DIR}" 2>&1
      DL_STAT=$?
    fi
    set -e
    if [[ "$DL_STAT" -ne 0 ]] || [[ ! -f "${TMP_DIR}/model.safetensors" ]]; then
      hf_troubleshoot
    fi

    echo "Converting safetensors → .pth (PyTorch state_dict)..."
    if ! python3 -c "import torch; import safetensors" 2>/dev/null; then
      echo "Install:  pip install torch safetensors"
      echo "(Use the same environment you will use for RETFound.)"
      exit 1
    fi
    python3 - "${TMP_DIR}/model.safetensors" "${OUTFILE}" <<'PY'
import sys
import torch
from safetensors.torch import load_file

src, dst = sys.argv[1], sys.argv[2]
state_dict = load_file(src)
torch.save(state_dict, dst)
print(f"Saved {len(state_dict)} tensors to {dst}")
PY

    rm -rf "${TMP_DIR}"
    echo "Saved to: ${OUTFILE}"
    ;;
esac

echo ""
echo "Done. File: ${OUTFILE}"
echo ""
echo "RETFound train.sh:"
echo "  MODEL=\"Dinov3\""
echo "  MODEL_ARCH=\"dinov3_vits16\""
echo "  FINETUNE=\"${OUTFILE}\""
