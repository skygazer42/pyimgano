# Optional extras + strict import boundaries (40 tasks)

Goal: make `pyimgano` install lightweight by default, while keeping deep/deploy
capabilities available via extras:

- `pyimgano[torch]` for deep backends (torch/torchvision)
- `pyimgano[onnx]` for deploy-style ONNX runtime (onnx/onnxruntime/onnxscript)
- `pyimgano[openvino]` for OpenVINO deployment (future-facing)

Core principle: **base installs must not hard-import optional deps**. Optional
deps should only be imported when the user chooses a feature/model that needs
them, and error messages must be actionable (install-hint points to extras).

---

## Task list (execute all)

### Packaging / metadata
1. [x] Audit current `[project.dependencies]` for heavy deps
2. [x] Define `torch` extra (`torch`, `torchvision`)
3. [x] Define `onnx` extra (`onnx`, `onnxruntime`, `onnxscript`)
4. [x] Define `openvino` extra (`openvino`)
5. [x] Move `torch`/`torchvision` out of core deps
6. [x] Move `matplotlib` out of core deps into `viz` extra
7. [x] Move `scikit-image` out of core deps into `skimage` extra
8. [x] Move `numba` out of core deps into `numba` extra
9. [x] Update `diffusion` extra to depend on `pyimgano[torch]`
10. [x] Update `clip` extra to depend on `pyimgano[torch]`
11. [x] Update `anomalib` extra to depend on `pyimgano[torch]`
12. [x] Update `mamba` extra to depend on `pyimgano[torch]`
13. [x] Update `all` extra to include new extras (torch/onnx/openvino/skimage/numba)
14. [x] Keep `backends` extra stable (clip+faiss+anomalib)
15. [x] Align `requirements.txt` with the new core+extras strategy

### Optional import hardening (runtime)
16. [x] Add/extend missing-dependency → extra mapping for friendly hints
17. [x] `pyimgano.models` lazy ctor: catch `ModuleNotFoundError` and raise `ImportError` with `pyimgano[...]` hint
18. [x] `pyimgano-export-torchscript`: use optional-deps `require("torch", extra="torch")`
19. [x] `pyimgano-export-onnx`: use `require("torch", extra="torch")` + `require("onnx*", extra="onnx")`
20. [x] `torchvision_safe`: use `require(..., extra="torch")` instead of raw imports
21. [x] `torchscript_embed`: use `require("torch", extra="torch")` in `_ensure_ready`
22. [x] `onnx_embed`: use `require("onnxruntime", extra="onnx")` in `_ensure_ready`
23. [x] `openclip_embed`: use `require("torch", extra="torch")` in `_ensure_ready`
24. [x] `torch_infer`: error hint should be `pyimgano[torch]`
25. [x] `workbench.load_run.load_checkpoint_into_detector`: error hint should be `pyimgano[torch]`
26. [x] `preprocessing.enhancer`: torch should be optional (only `gaussian_blur_torch` needs it)
27. [x] Remove top-level `matplotlib` imports from classical models that only need it for debug plotting
28. [x] Update other `pip install torch` hints to `pyimgano[torch]` where appropriate

### Tests
29. [x] Add pyproject extras tests for `torch`/`onnx`/`openvino`
30. [x] Add require-hint unit tests for `torch` and `onnxruntime`
31. [x] Add model-lazy-import error test: missing `torch` ⇒ `pip install 'pyimgano[torch]'`
32. [x] Add feature extractor error test: missing `onnxruntime` ⇒ `pip install 'pyimgano[onnx]'`
33. [x] Add preprocessing import smoke: importing `pyimgano.preprocessing` does not import `torch`
34. [x] Ensure existing lightweight-import subprocess tests still pass (`models`/`features`)

### Docs
35. [x] Update `README.md` optional dependencies section with new extras
36. [x] Update `docs/CLI_REFERENCE.md` with install hints for export/infer (torch/onnx)
37. [x] Update `docs/WORKBENCH.md` / industrial docs if they mention torch as default dep

### Verification
38. [x] Run targeted pytest for changed areas (extras/import/CLI smoke)
39. [x] Subprocess sanity: `import pyimgano; import pyimgano.models; import pyimgano.features; import pyimgano.preprocessing` without heavy roots
40. [x] Final pass: ensure error messages are consistent and actionable
