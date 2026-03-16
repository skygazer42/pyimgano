#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
clone_reference_repos.sh

Clone a curated set of external industrial anomaly detection / inspection repos
for LOCAL STUDY ONLY (shallow clones). The clones are NOT intended to be
committed; keep them in a cache directory ignored by git.

Usage:
  bash tools/clone_reference_repos.sh --dir .cache/pyimgano_refs --jobs 4

Options:
  --dir DIR     Target directory for clones (default: .cache/pyimgano_refs)
  --jobs N      Parallel clone jobs (default: 2)
  --include-heavy  Also clone very large repos (may take a long time)
  --help        Show this help text

Notes:
  - This script may take time and require network access.
  - We learn patterns from these repos and then re-implement ideas using pyimgano
    base classes. Copying small code snippets is allowed only when the upstream
    license is compatible and we keep required notices.
EOF
  return 0
}

DIR=".cache/pyimgano_refs"
JOBS="2"
INCLUDE_HEAVY="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)
      DIR="${2:-}"
      shift 2
      ;;
    --jobs)
      JOBS="${2:-}"
      shift 2
      ;;
    --include-heavy)
      INCLUDE_HEAVY="1"
      shift 1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "$DIR"

declare -a REPOS=(
  # Framework-style (pipelines + config)
  "https://github.com/openvinotoolkit/anomalib.git"
  # Method implementations (conceptual reference only)
  "https://github.com/tiskw/patchcore-ad.git"
  "https://github.com/byungjae89/SPADE-pytorch.git"
  "https://github.com/byungjae89/MahalanobisAD-pytorch.git"
  "https://github.com/xiahaifeng1995/PaDiM-Anomaly-Detection-Localization-master.git"
  "https://github.com/DoMaLi94/industrial-image-anomaly-detection.git"
  "https://github.com/DonaldRR/SimpleNet.git"
  "https://github.com/VitjanZ/DRAEM.git"
  "https://github.com/LilitYolyan/CutPaste.git"
  "https://github.com/dammsi/AnomalyDINO.git"
  "https://github.com/guojiajeremy/Dinomaly.git"
  "https://github.com/xrli-U/MuSc.git"
  # Additional industrial/method references (study-only)
  "https://github.com/cnulab/RealNet.git"
  "https://github.com/kaichen-z/RAD.git"
  "https://github.com/FangshuoX/ReinAD.git"
  "https://github.com/EPFL-IMOS/AnomalyAny.git"
  "https://github.com/CASIA-IVA-Lab/AnomalyGPT.git"
  "https://github.com/jam-cc/MMAD.git"
  "https://github.com/tientrandinh/Revisiting-Reverse-Distillation.git"
  "https://github.com/gdwang08/STFPM.git"
  "https://github.com/LeapMind/PUAD.git"
  "https://github.com/BioHPC/WE-PaDiM.git"
  "https://github.com/yzhao062/pyod.git"
  "https://github.com/albumentations-team/albumentations.git"
  "https://github.com/kornia/kornia.git"
  # Indexes / surveys
  "https://github.com/M-3LAB/awesome-industrial-anomaly-detection.git"
  "https://github.com/IHPCRits/IAD-Survey.git"
)

declare -a HEAVY_REPOS=(
  # This repo can be very large due to included assets/weights. Keep it opt-in.
  "https://github.com/amazon-science/patchcore-inspection.git"
  # Foundation weights source repo (torch.hub entrypoint). Study-only, code only.
  "https://github.com/facebookresearch/dinov2.git"
)

clone_one() {
  local url="$1"
  local name
  name="$(basename "$url" .git)"
  local out="$DIR/$name"
  if [[ -d "$out/.git" ]]; then
    echo "[skip] $name already exists in $out"
    return 0
  fi
  echo "[clone] $name"
  # Study-only clones should be as light as possible:
  # - avoid fetching LFS objects (often large weights/assets)
  # - prefer partial clone when supported by remote
  # NOTE: Use `--no-checkout` to avoid downloading large blobs at clone time
  # (some repos store weights/assets directly in git). You can inspect files via:
  #   git -C <repo> show HEAD:path/to/file
  # or enable a sparse checkout later.
  if GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --filter=blob:none --no-checkout "$url" "$out"; then
    echo "[ok] $name"
    return 0
  fi

  # Best-effort: keep the script useful even when some repos move or disappear.
  echo "[fail] $name ($url)" >&2
  rm -rf "$out" || true
  return 0
}

export -f clone_one
export DIR

if command -v xargs >/dev/null 2>&1; then
  printf "%s\n" "${REPOS[@]}" | xargs -n 1 -P "$JOBS" bash -lc 'clone_one "$@"' _
  if [[ "$INCLUDE_HEAVY" == "1" ]]; then
    printf "%s\n" "${HEAVY_REPOS[@]}" | xargs -n 1 -P "$JOBS" bash -lc 'clone_one "$@"' _
  fi
else
  for r in "${REPOS[@]}"; do
    clone_one "$r"
  done
  if [[ "$INCLUDE_HEAVY" == "1" ]]; then
    for r in "${HEAVY_REPOS[@]}"; do
      clone_one "$r"
    done
  fi
fi

echo ""
echo "Done. Cloned repos are in: $DIR"
