#!/usr/bin/env bash
set -euo pipefail

dry_run="false"
skip_install="false"
skip_tests="false"
skip_scan="false"

for arg in "$@"; do
  case "$arg" in
    --dry-run)
      dry_run="true"
      ;;
    --skip-install)
      skip_install="true"
      ;;
    --skip-tests)
      skip_tests="true"
      ;;
    --skip-scan)
      skip_scan="true"
      ;;
    *)
      printf 'Unknown argument: %s\n' "$arg" >&2
      exit 2
      ;;
  esac
done

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYIMGANO_SONAR_PYTHON_BIN:-python3}"
install_cmd="${PYIMGANO_SONAR_INSTALL_COMMAND:-${python_bin} -m pip install --upgrade pip setuptools wheel && pip install -e .[dev,torch,skimage]}"
if [ -n "${PYIMGANO_SONAR_PYTEST_COMMAND:-}" ]; then
  pytest_cmd="${PYIMGANO_SONAR_PYTEST_COMMAND}"
else
  pytest_cmd="$(cat <<'EOF'
coverage erase
pytest -v --cov=pyimgano --cov-report= --cov-append tests/contracts
pytest -v --cov=pyimgano --cov-report= --cov-append tests/test_[a-cA-C]*.py
pytest -v --cov=pyimgano --cov-report= --cov-append tests/test_[d-fD-F]*.py
pytest -v --cov=pyimgano --cov-report= --cov-append tests/test_[g-lG-L]*.py
pytest -v --cov=pyimgano --cov-report= --cov-append tests/test_[m-rM-R]*.py
pytest -v --cov=pyimgano --cov-report= --cov-append tests/test_[s-zS-Z]*.py
coverage xml
coverage report -m
EOF
)"
fi
scanner_image="${SONAR_SCANNER_IMAGE:-sonarsource/sonar-scanner-cli}"
host_url="${SONAR_HOST_URL:-https://sonarcloud.io}"
project_key="${SONAR_PROJECT_KEY:-skygazer42_pyimgano}"
scanner_args="-Dsonar.projectKey=${project_key} -Dsonar.qualitygate.wait=true -Dsonar.scanner.skipJreProvisioning=true"
if [ -n "${SONAR_PROJECT_VERSION:-}" ]; then
  scanner_args="${scanner_args} -Dsonar.projectVersion=${SONAR_PROJECT_VERSION}"
fi
if [ -n "${SONAR_SCANNER_EXTRA_ARGS:-}" ]; then
  scanner_args="${scanner_args} ${SONAR_SCANNER_EXTRA_ARGS}"
fi
scanner_cmd="${PYIMGANO_SONAR_SCAN_COMMAND:-docker run --rm -e SONAR_TOKEN -e SONAR_HOST_URL=${host_url} -v ${repo_root}:/usr/src -w /usr/src ${scanner_image} ${scanner_args}}"

run_cmd() {
  local cmd="$1"
  if [ "$dry_run" = "true" ]; then
    printf '%s\n' "$cmd"
    return 0
  fi
  bash -lc "$cmd"
}

if [ "$dry_run" = "true" ]; then
  if [ "$skip_install" != "true" ]; then
    printf '%s\n' "$install_cmd"
  fi
  if [ "$skip_tests" != "true" ]; then
    printf '%s\n' "$pytest_cmd"
  fi
  if [ "$skip_scan" != "true" ]; then
    printf '%s\n' "$scanner_cmd"
  fi
  exit 0
fi

if [ "$skip_scan" != "true" ] && [ -z "${SONAR_TOKEN:-}" ]; then
  printf 'SONAR_TOKEN is required when scan is enabled.\n' >&2
  exit 1
fi

if [ "$skip_install" != "true" ]; then
  run_cmd "$install_cmd"
fi

if [ "$skip_tests" != "true" ]; then
  run_cmd "$pytest_cmd"
fi

if [ "$skip_scan" = "true" ]; then
  exit 0
fi

run_cmd "$scanner_cmd"
