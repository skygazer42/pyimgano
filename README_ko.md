# pyimgano（한국어）

산업 검사(품질/결함) 시나리오를 위한 **이미지 이상 탐지 / 결함 탐지** 툴킷(이미지 레벨 + 픽셀 레벨).

`pyimgano`는 “현업 적용”을 목표로 합니다:

- **모델 레지스트리**(120+ 모델 엔트리: 구현 + 선택적 백엔드 + 별칭)
- **재현 가능한 CLI 실행**(workbench + report + per-image JSONL)
- **배포 친화적 추론 출력**(JSONL, 선택적으로 결함 mask + connected-component regions)
- **산업용 IO**(numpy-first / 명시적 이미지 포맷 / 고해상도 tiling)

> 최신 내용은 영어 `README.md`를 참고하세요(번역은 늦을 수 있습니다).

## 설치

```bash
pip install pyimgano
```

> PyPI 공개 전에는 소스에서 설치:
>
> ```bash
> git clone https://github.com/skygazer42/pyimgano.git
> cd pyimgano
> pip install -e ".[dev]"
> ```

## 빠른 시작(CLI)

### 학습(workbench) → `infer_config.json` 내보내기

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_defects_fp40.json \
  --export-infer-config
```

### 추론 → JSONL(선택적으로 결함 mask/regions)

```bash
pyimgano-infer \
  --infer-config /path/to/run_dir/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

관련 문서:
- `docs/WORKBENCH.md`
- `docs/CLI_REFERENCE.md`
- `docs/INDUSTRIAL_INFERENCE.md`
- `docs/FALSE_POSITIVE_DEBUGGING.md`

## 모델 탐색(CLI)

```bash
pyimgano-benchmark --list-models
pyimgano-benchmark --list-models --tags numpy,pixel_map
pyimgano-benchmark --model-info vision_patchcore --json
```
