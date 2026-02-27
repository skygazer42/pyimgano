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
  --help        Show this help text

Notes:
  - This script may take time and require network access.
  - We learn patterns from these repos and then re-implement ideas using pyimgano
    base classes. Copying small code snippets is allowed only when the upstream
    license is compatible and we keep required notices.
EOF
}

DIR=".cache/pyimgano_refs"
JOBS="2"

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
  "https://github.com/openvinotoolkit/anomalib.git"
  "https://github.com/amazon-science/patchcore-inspection.git"
  "https://github.com/DoMaLi94/industrial-image-anomaly-detection.git"
  "https://github.com/DonaldRR/SimpleNet.git"
  "https://github.com/VitjanZ/DRAEM.git"
  "https://github.com/LilitYolyan/CutPaste.git"
  "https://github.com/guojiajeremy/Dinomaly.git"
  "https://github.com/xrli-U/MuSc.git"
  "https://github.com/yzhao062/pyod.git"
  "https://github.com/albumentations-team/albumentations.git"
  "https://github.com/kornia/kornia.git"
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
  if git clone --depth 1 "$url" "$out"; then
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
else
  for r in "${REPOS[@]}"; do
    clone_one "$r"
  done
fi

echo ""
echo "Done. Cloned repos are in: $DIR"
