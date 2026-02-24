# pyimgano（日本語）

産業検査向けの **画像異常検知 / 欠陥検知** ツールキット（画像レベル + ピクセルレベル）。

`pyimgano` は「実運用」を意識した設計です：

- **モデルレジストリ**（120+ のモデルエントリ：実装 + 任意バックエンド + エイリアス）
- **再現可能な CLI 実行**（workbench + レポート + per-image JSONL）
- **デプロイ向け推論出力**（JSONL、任意で欠陥 mask + 連結成分 regions）
- **産業向け IO**（numpy-first / 明示的な画像フォーマット / 高解像度 tiling）

> 最新の情報は英語版 `README.md` を参照してください（翻訳は遅れる場合があります）。

## インストール

```bash
pip install pyimgano
```

> PyPI 公開前はソースから：
>
> ```bash
> git clone https://github.com/skygazer42/pyimgano.git
> cd pyimgano
> pip install -e ".[dev]"
> ```

## クイックスタート（CLI）

### 学習（workbench）→ `infer_config.json` をエクスポート

```bash
pyimgano-train \
  --config examples/configs/industrial_adapt_defects_roi.json \
  --export-infer-config
```

### 推論 → JSONL（任意で欠陥 mask/regions）

```bash
pyimgano-infer \
  --infer-config /path/to/run_dir/artifacts/infer_config.json \
  --input /path/to/images \
  --defects \
  --save-masks /tmp/pyimgano_masks \
  --save-jsonl /tmp/pyimgano_results.jsonl
```

関連ドキュメント：
- `docs/WORKBENCH.md`
- `docs/CLI_REFERENCE.md`
- `docs/INDUSTRIAL_INFERENCE.md`

## モデルの探索（CLI）

```bash
pyimgano-benchmark --list-models
pyimgano-benchmark --list-models --tags numpy,pixel_map
pyimgano-benchmark --model-info vision_patchcore --json
```

