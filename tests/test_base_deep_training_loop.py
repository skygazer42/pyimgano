import numpy as np


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
            return float(loss.item())

        def evaluating_forward(self, batch_data):
            x, _y = batch_data
            x = x.to(self.device)
            out = self.model(x)
            # Per-sample MSE
            err = torch.mean((out - x) ** 2, dim=1)
            return err.detach().cpu().numpy()

    rng = np.random.default_rng(0)
    X_train = rng.standard_normal(size=(32, 8)).astype(np.float32)
    X_test = rng.standard_normal(size=(8, 8)).astype(np.float32)

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

    det.fit(X_train)
    assert hasattr(det, "decision_scores_")
    assert np.asarray(det.decision_scores_).shape == (len(X_train),)

    scores = np.asarray(det.decision_function(X_test), dtype=np.float32)
    assert scores.shape == (len(X_test),)
    assert np.isfinite(scores).all()

    preds = np.asarray(det.predict(X_test), dtype=int)
    assert preds.shape == (len(X_test),)
    assert set(np.unique(preds)).issubset({0, 1})

    proba = np.asarray(det.predict_proba(X_test), dtype=np.float32)
    assert proba.shape == (len(X_test), 2)
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)
