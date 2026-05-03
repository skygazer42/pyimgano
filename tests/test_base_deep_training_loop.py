from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")


def _loss_value(loss):
    detached = loss.detach() if hasattr(loss, "detach") else loss
    return float(detached.item()) if hasattr(detached, "item") else float(detached)


def test_base_deep_detector_fit_score_predict_smoke() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            # Simple linear autoencoder-ish: map -> same dim.
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            out = self.model(x)
            # Per-sample MSE
            err = torch.mean((out - x) ** 2, dim=1)
            return err.detach().cpu().numpy()

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal(size=(32, 8)).astype(np.float32)
    x_test = rng.standard_normal(size=(8, 8)).astype(np.float32)

    det = DummyDeep(
        contamination=0.1,
        preprocessing=False,
        lr=1e-2,
        epoch_num=1,
        batch_size=8,
        optimizer_name="adam",
        criterion_name="mse",
        device="cpu",
        verbose=0,
    )

    det.fit(x_train)
    assert hasattr(det, "decision_scores_")
    assert np.asarray(det.decision_scores_).shape == (len(x_train),)

    scores = np.asarray(det.decision_function(x_test), dtype=np.float32)
    assert scores.shape == (len(x_test),)
    assert np.isfinite(scores).all()

    preds = np.asarray(det.predict(x_test), dtype=int)
    assert preds.shape == (len(x_test),)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = np.asarray(det.predict_proba(x_test), dtype=np.float32)
    assert proba.shape == (len(x_test), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_base_deep_detector_stops_early_on_training_plateau() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._loss_value = 1.0

        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            del batch_data
            return float(self._loss_value)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        epoch_num=10,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.early_stopping_patience = 2
    det.early_stopping_min_delta = 0.0

    det.fit(X_train)

    assert det.training_epochs_completed_ == 3
    assert det.training_steps_completed_ == 12
    assert det.training_stop_reason_ == "early_stopping"
    assert det.training_loss_history_ == [pytest.approx(1.0)] * 3


def test_base_deep_detector_respects_max_steps_budget() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            del batch_data
            return 0.5

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        epoch_num=10,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.max_steps = 5

    det.fit(X_train)

    assert det.training_steps_completed_ == 5
    assert det.training_epochs_completed_ == 2
    assert det.training_stop_reason_ == "max_steps"


def test_base_deep_detector_emits_live_epoch_timing_metrics(monkeypatch) -> None:
    import torch

    import pyimgano.models.base_deep as base_deep_module
    from pyimgano.models.base_deep import BaseDeepLearningDetector
    from pyimgano.train_progress import TrainProgressReporter, use_train_progress_reporter

    class _Recorder(TrainProgressReporter):
        def __init__(self) -> None:
            self.events: list[dict[str, float | int | bool | None]] = []

        def on_training_epoch(self, *, epoch, total_epochs, metrics, live=False):  # noqa: ANN001
            self.events.append(
                {
                    "epoch": int(epoch),
                    "total_epochs": None if total_epochs is None else int(total_epochs),
                    "loss": float(metrics["loss"]),
                    "lr": float(metrics["lr"]),
                    "epoch_s": float(metrics["epoch_s"]),
                    "elapsed_s": float(metrics["elapsed_s"]),
                    "eta_s": float(metrics["eta_s"]),
                    "train_items": int(metrics["train_items"]),
                    "items_per_s": float(metrics["items_per_s"]),
                    "live": bool(live),
                }
            )

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            out = self.model(x)
            err = torch.mean((out - x) ** 2, dim=1)
            return err.detach().cpu().numpy()

    perf_counter_values = iter([10.0, 13.0, 13.5, 15.5])
    monkeypatch.setattr(
        base_deep_module.time,
        "perf_counter",
        lambda: next(perf_counter_values),
    )

    rng = np.random.default_rng(0)
    x_train = rng.standard_normal(size=(8, 4)).astype(np.float32)
    det = DummyDeep(
        contamination=0.1,
        preprocessing=False,
        lr=1e-2,
        epoch_num=2,
        batch_size=2,
        optimizer_name="adam",
        criterion_name="mse",
        device="cpu",
        verbose=0,
    )
    recorder = _Recorder()

    with use_train_progress_reporter(recorder):
        det.fit(x_train)

    assert recorder.events == [
        {
            "epoch": 1,
            "total_epochs": 2,
            "loss": pytest.approx(recorder.events[0]["loss"]),
            "lr": pytest.approx(det.training_lr_history_[0]),
            "epoch_s": pytest.approx(3.0),
            "elapsed_s": pytest.approx(3.0),
            "eta_s": pytest.approx(3.0),
            "train_items": 8,
            "items_per_s": pytest.approx(8.0 / 3.0),
            "live": True,
        },
        {
            "epoch": 2,
            "total_epochs": 2,
            "loss": pytest.approx(recorder.events[1]["loss"]),
            "lr": pytest.approx(det.training_lr_history_[1]),
            "epoch_s": pytest.approx(2.0),
            "elapsed_s": pytest.approx(5.0),
            "eta_s": pytest.approx(0.0),
            "train_items": 8,
            "items_per_s": pytest.approx(4.0),
            "live": True,
        },
    ]


def test_base_deep_detector_does_not_print_verbose_epoch_progress(capsys) -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            out = self.model(x)
            err = torch.mean((out - x) ** 2, dim=1)
            return err.detach().cpu().numpy()

    rng = np.random.default_rng(9)
    x_train = rng.standard_normal(size=(16, 4)).astype(np.float32)
    det = DummyDeep(
        contamination=0.1,
        preprocessing=False,
        lr=1e-2,
        epoch_num=2,
        batch_size=4,
        optimizer_name="adam",
        criterion_name="mse",
        device="cpu",
        verbose=2,
    )

    det.fit(x_train)
    out = capsys.readouterr().out
    assert out == ""


def test_base_deep_detector_uses_weight_decay_num_workers_and_optimizer_name(
    monkeypatch,
) -> None:
    import torch

    import pyimgano.models.base_deep as base_deep_module
    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    observed: dict[str, object] = {"loaders": []}
    original_dataloader = torch.utils.data.DataLoader
    original_resolve_optimizer = base_deep_module._resolve_optimizer

    def _recording_dataloader(*args, **kwargs):
        observed["loaders"].append(dict(kwargs))
        return original_dataloader(*args, **kwargs)

    def _recording_resolve_optimizer(
        *,
        torch,
        name,
        params,
        lr,
        weight_decay,
        momentum=None,
        nesterov=False,
        dampening=None,
        beta1=None,
        beta2=None,
        amsgrad=False,
        eps=None,
        alpha=None,
        centered=False,
    ):
        observed["optimizer"] = {
            "name": name,
            "lr": lr,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "nesterov": nesterov,
            "dampening": dampening,
            "beta1": beta1,
            "beta2": beta2,
            "amsgrad": amsgrad,
            "eps": eps,
            "alpha": alpha,
            "centered": centered,
        }
        return original_resolve_optimizer(
            torch=torch,
            name=name,
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            dampening=dampening,
            beta1=beta1,
            beta2=beta2,
            amsgrad=amsgrad,
            eps=eps,
            alpha=alpha,
            centered=centered,
        )

    monkeypatch.setattr(torch.utils.data, "DataLoader", _recording_dataloader)
    monkeypatch.setattr(base_deep_module, "_resolve_optimizer", _recording_resolve_optimizer)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        epoch_num=1,
        batch_size=2,
        optimizer_name="adam",
        device="cpu",
        verbose=0,
    )
    det.num_workers = 3
    det.weight_decay = 0.02
    det.optimizer_name = "sgd"
    det.optimizer_momentum = 0.75
    det.optimizer_nesterov = True
    det.optimizer_dampening = 0.0

    det.fit(X_train)

    train_loader_call = next(call for call in observed["loaders"] if call.get("shuffle") is True)
    assert train_loader_call["num_workers"] == 3
    assert observed["optimizer"] == {
        "name": "sgd",
        "lr": pytest.approx(det.lr),
        "weight_decay": pytest.approx(0.02),
        "momentum": pytest.approx(0.75),
        "nesterov": True,
        "dampening": pytest.approx(0.0),
        "beta1": None,
        "beta2": None,
        "amsgrad": False,
        "eps": None,
        "alpha": None,
        "centered": False,
    }


def test_base_deep_detector_supports_adam_amsgrad_and_rmsprop_options(monkeypatch) -> None:
    import torch

    import pyimgano.models.base_deep as base_deep_module
    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    observed: list[dict[str, object]] = []
    original_resolve_optimizer = base_deep_module._resolve_optimizer

    def _recording_resolve_optimizer(
        *,
        torch,
        name,
        params,
        lr,
        weight_decay,
        momentum=None,
        nesterov=False,
        dampening=None,
        beta1=None,
        beta2=None,
        amsgrad=False,
        eps=None,
        alpha=None,
        centered=False,
    ):
        observed.append(
            {
                "name": name,
                "momentum": momentum,
                "nesterov": nesterov,
                "dampening": dampening,
                "beta1": beta1,
                "beta2": beta2,
                "amsgrad": amsgrad,
                "eps": eps,
                "alpha": alpha,
                "centered": centered,
            }
        )
        return original_resolve_optimizer(
            torch=torch,
            name=name,
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            dampening=dampening,
            beta1=beta1,
            beta2=beta2,
            amsgrad=amsgrad,
            eps=eps,
            alpha=alpha,
            centered=centered,
        )

    monkeypatch.setattr(base_deep_module, "_resolve_optimizer", _recording_resolve_optimizer)

    X_train = np.zeros((8, 4), dtype=np.float32)

    adam_det = DummyDeep(
        preprocessing=False,
        epoch_num=1,
        batch_size=2,
        optimizer_name="adamw",
        device="cpu",
        verbose=0,
    )
    adam_det.adam_beta1 = 0.81
    adam_det.adam_beta2 = 0.95
    adam_det.adam_amsgrad = True
    adam_det.optimizer_eps = 1e-6
    adam_det.fit(X_train)

    rms_det = DummyDeep(
        preprocessing=False,
        epoch_num=1,
        batch_size=2,
        optimizer_name="rmsprop",
        device="cpu",
        verbose=0,
    )
    rms_det.optimizer_momentum = 0.2
    rms_det.optimizer_eps = 2e-6
    rms_det.rmsprop_alpha = 0.91
    rms_det.rmsprop_centered = True
    rms_det.fit(X_train)

    assert observed == [
        {
            "name": "adamw",
            "momentum": None,
            "nesterov": False,
            "dampening": None,
            "beta1": pytest.approx(0.81),
            "beta2": pytest.approx(0.95),
            "amsgrad": True,
            "eps": pytest.approx(1e-6),
            "alpha": None,
            "centered": False,
        },
        {
            "name": "rmsprop",
            "momentum": pytest.approx(0.2),
            "nesterov": False,
            "dampening": None,
            "beta1": None,
            "beta2": None,
            "amsgrad": False,
            "eps": pytest.approx(2e-6),
            "alpha": pytest.approx(0.91),
            "centered": True,
        },
    ]


def test_base_deep_detector_applies_ema_weights_after_training() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._value = 1.0

        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size, bias=False)
            with torch.no_grad():
                self.model.weight.zero_()
            return self.model

        def training_forward(self, batch_data):
            del batch_data
            with torch.no_grad():
                self.model.weight.fill_(self._value)
                self._value += 1.0
            return float(self._value)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            value = _loss_value(self.model.weight.mean())
            return torch.full((x.shape[0],), value, dtype=torch.float32)

    X_train = np.zeros((4, 1), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        epoch_num=2,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.ema_enabled = True
    det.ema_decay = 0.5
    det.ema_start_epoch = 2

    det.fit(X_train)

    assert det.training_epochs_completed_ == 2
    assert det.training_steps_completed_ == 4
    assert det.training_ema_updates_ == 2
    assert det.training_ema_applied_ is True
    assert _loss_value(det.model.weight.mean()) == pytest.approx(3.5)
    assert np.asarray(det.decision_scores_).tolist() == pytest.approx([3.5, 3.5, 3.5, 3.5])


def test_base_deep_detector_applies_step_scheduler_and_records_lr_history() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        lr=0.1,
        epoch_num=3,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.scheduler_name = "step"
    det.scheduler_step_size = 1
    det.scheduler_gamma = 0.5

    det.fit(X_train)

    assert det.training_epochs_completed_ == 3
    assert det.training_stop_reason_ == "completed"
    assert det.training_lr_history_ == [
        pytest.approx(0.1),
        pytest.approx(0.05),
        pytest.approx(0.025),
    ]
    assert det.training_last_lr_ == pytest.approx(0.0125)


def test_base_deep_detector_applies_warmup_before_scheduler() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        lr=0.1,
        epoch_num=4,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.scheduler_name = "step"
    det.scheduler_step_size = 1
    det.scheduler_gamma = 0.5
    det.warmup_epochs = 2
    det.warmup_start_factor = 0.5

    det.fit(X_train)

    assert det.training_lr_history_ == [
        pytest.approx(0.05),
        pytest.approx(0.1),
        pytest.approx(0.1),
        pytest.approx(0.05),
    ]
    assert det.training_last_lr_ == pytest.approx(0.025)


def test_base_deep_detector_applies_multistep_scheduler_from_milestones() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        lr=0.1,
        epoch_num=4,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.scheduler_name = "multistep"
    det.scheduler_milestones = [1, 3]
    det.scheduler_gamma = 0.5

    det.fit(X_train)

    assert det.training_lr_history_ == [
        pytest.approx(0.1),
        pytest.approx(0.05),
        pytest.approx(0.05),
        pytest.approx(0.025),
    ]
    assert det.training_last_lr_ == pytest.approx(0.025)


def test_base_deep_detector_applies_plateau_scheduler_from_training_loss() -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._loss_value = 1.0

        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            del batch_data
            return float(self._loss_value)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        lr=0.1,
        epoch_num=4,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.scheduler_name = "plateau"
    det.scheduler_patience = 0
    det.scheduler_factor = 0.5
    det.scheduler_min_lr = 0.02

    det.fit(X_train)

    assert det.training_loss_history_ == [pytest.approx(1.0)] * 4
    assert det.training_lr_history_ == [
        pytest.approx(0.1),
        pytest.approx(0.1),
        pytest.approx(0.05),
        pytest.approx(0.025),
    ]
    assert det.training_last_lr_ == pytest.approx(0.02)


def test_base_deep_detector_passes_plateau_plateau_kwargs_to_scheduler(
    monkeypatch,
) -> None:
    import torch

    import pyimgano.models.base_deep as base_deep_module
    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    observed: dict[str, object] = {}
    original_resolve_scheduler = base_deep_module._resolve_scheduler

    def _recording_resolve_scheduler(
        *,
        torch,
        name,
        optimizer,
        epoch_num,
        milestones,
        step_size,
        gamma,
        t_max,
        eta_min,
        patience,
        factor,
        min_lr,
        cooldown,
        threshold,
        threshold_mode,
        eps,
    ):
        observed["scheduler"] = {
            "name": name,
            "epoch_num": epoch_num,
            "milestones": milestones,
            "step_size": step_size,
            "gamma": gamma,
            "t_max": t_max,
            "eta_min": eta_min,
            "patience": patience,
            "factor": factor,
            "min_lr": min_lr,
            "cooldown": cooldown,
            "threshold": threshold,
            "threshold_mode": threshold_mode,
            "eps": eps,
        }
        return original_resolve_scheduler(
            torch=torch,
            name=name,
            optimizer=optimizer,
            epoch_num=epoch_num,
            milestones=milestones,
            step_size=step_size,
            gamma=gamma,
            t_max=t_max,
            eta_min=eta_min,
            patience=patience,
            factor=factor,
            min_lr=min_lr,
            cooldown=cooldown,
            threshold=threshold,
            threshold_mode=threshold_mode,
            eps=eps,
        )

    monkeypatch.setattr(base_deep_module, "_resolve_scheduler", _recording_resolve_scheduler)

    X_train = np.zeros((8, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        lr=0.1,
        epoch_num=2,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.scheduler_name = "plateau"
    det.scheduler_patience = 1
    det.scheduler_factor = 0.5
    det.scheduler_min_lr = 0.01
    det.scheduler_cooldown = 2
    det.scheduler_threshold = 0.003
    det.scheduler_threshold_mode = "abs"
    det.scheduler_eps = 1e-7

    det.fit(X_train)

    assert observed["scheduler"] == {
        "name": "plateau",
        "epoch_num": 2,
        "milestones": None,
        "step_size": None,
        "gamma": None,
        "t_max": None,
        "eta_min": None,
        "patience": 1,
        "factor": 0.5,
        "min_lr": 0.01,
        "cooldown": 2,
        "threshold": 0.003,
        "threshold_mode": "abs",
        "eps": 1e-7,
    }


def test_base_deep_detector_respects_criterion_shuffle_drop_last_and_loader_memory_flags(
    monkeypatch,
) -> None:
    import torch

    from pyimgano.models.base_deep import BaseDeepLearningDetector

    class DummyDeep(BaseDeepLearningDetector):
        def build_model(self):
            self.model = torch.nn.Linear(self.feature_size, self.feature_size)
            return self.model

        def training_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = self.criterion(out, x)
            loss.backward()
            self.optimizer.step()
            return _loss_value(loss)

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            return torch.zeros((x.shape[0],), dtype=torch.float32)

    observed: dict[str, object] = {"loaders": []}
    original_dataloader = torch.utils.data.DataLoader

    def _recording_dataloader(*args, **kwargs):
        observed["loaders"].append(dict(kwargs))
        return original_dataloader(*args, **kwargs)

    monkeypatch.setattr(torch.utils.data, "DataLoader", _recording_dataloader)

    X_train = np.zeros((5, 4), dtype=np.float32)
    det = DummyDeep(
        preprocessing=False,
        epoch_num=1,
        batch_size=2,
        device="cpu",
        verbose=0,
    )
    det.criterion_name = "mae"
    det.shuffle_train = False
    det.drop_last = True
    det.num_workers = 2
    det.pin_memory = True
    det.persistent_workers = True

    det.fit(X_train)

    train_loader_call = next(call for call in observed["loaders"] if "drop_last" in call)
    assert train_loader_call["shuffle"] is False
    assert train_loader_call["drop_last"] is True
    assert train_loader_call["pin_memory"] is True
    assert train_loader_call["persistent_workers"] is True
    assert det.training_steps_completed_ == 2
    assert det.criterion.__class__.__name__ == "L1Loss"
