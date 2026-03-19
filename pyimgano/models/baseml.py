# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from pyimgano.features.protocols import FeatureExtractor, FittableFeatureExtractor
from pyimgano.features.registry import resolve_feature_extractor

from ._legacy_x import MISSING, resolve_legacy_x_keyword
from .base_detector import BaseDetector


class BaseVisionDetector(BaseDetector):
    """
    闁圭鍋撻柡鍫濐槸鐎垫鎲楅崨顒傚晩缂備礁绻愰崥鈧柡鍫濇惈濞呮帞鈧冻缂氱弧鍕不濡や胶銆婇柣銊ュ椤鎲存径濠勭＝閻㈩垰鎲￠ˉ鍛圭€ｎ亝鐝ら柣銊ュ婵炲﹦鎸掗垾宕囧敤缂侇偅鐪归埀?    """

    @abstractmethod
    def __init__(self, contamination=0.1, feature_extractor=None):
        super(BaseVisionDetector, self).__init__(contamination=contamination)
        # Compatibility: some utilities (e.g. `predict_proba`) expect `_classes` to exist.
        # In unsupervised detection this is always binary.
        self._set_n_classes(None)
        self._feature_cache = None

        if feature_extractor is None:
            # Provide a safe default so classical detectors work out-of-the-box.
            # This trades off accuracy for simplicity, but avoids surprising
            # TypeError/ValueError for new users and for quick benchmarks.
            from pyimgano.utils.image_ops import ImagePreprocessor

            feature_extractor = ImagePreprocessor(
                resize=(224, 224),
                output_tensor=False,
                error_mode="zeros",
            )

        # Allow JSON-friendly specs: {"name": "...", "kwargs": {...}}.
        feature_extractor = resolve_feature_extractor(feature_extractor)
        if not isinstance(feature_extractor, FeatureExtractor):
            raise TypeError("feature_extractor 闊洤鎳橀妴蹇曗偓鍦仧楠?.extract(inputs) -> np.ndarray")
        self._base_feature_extractor = feature_extractor
        self.feature_extractor = feature_extractor
        self._feature_extractor_fitted = False

        self.detector = self._build_detector()

    def set_feature_cache(self, cache_dir: str | Path | None) -> None:
        """Enable/disable disk feature caching.

        Notes
        -----
        When enabled, caching is applied in a best-effort way:
        - If inputs are all paths (str/Path), cache extracted feature rows by file metadata.
        - If inputs are all numpy arrays, cache extracted feature rows by hashing array content.
        - Mixed inputs fall back to no caching.
        """

        if cache_dir is None:
            self._feature_cache = None
            self._array_feature_cache = None
            self.feature_extractor = self._base_feature_extractor
            return

        from pyimgano.cache.array_features import ArrayFeatureCache, CachedArrayFeatureExtractor
        from pyimgano.cache.features import (
            CachedFeatureExtractor,
            FeatureCache,
            fingerprint_feature_extractor,
        )

        cache_root = Path(cache_dir)
        fp = fingerprint_feature_extractor(self._base_feature_extractor)

        self._feature_cache = FeatureCache(
            cache_dir=cache_root / "paths",
            extractor_fingerprint=fp,
        )
        self._array_feature_cache = ArrayFeatureCache(
            cache_dir=cache_root / "arrays",
            extractor_fingerprint=fp,
        )

        extractor = CachedFeatureExtractor(
            base_extractor=self._base_feature_extractor,
            cache=self._feature_cache,
        )
        self.feature_extractor = CachedArrayFeatureExtractor(
            base_extractor=extractor,
            cache=self._array_feature_cache,
        )

    @abstractmethod
    def _build_detector(self):
        """
        閻庤鐭粻鐔肩嵁閹壆绠查柛銉у仒缁斿瓨绋夐鍕緮濞达絾鎸惧▓鎴﹀Υ娴ｈ櫣鐥呴柛蹇撴憸濞堟垵顕ｉ崒姘卞煑婵☆偀鍋撴繛鏉戭儏濞呮帞鈧湱鍋樼欢銉╁Υ?            濞撴艾顑呴々? return sklearn.neighbors.LocalOutlierFactor(...)
        """
        pass

    def fit(self, x: object = MISSING, y=None, **kwargs: object):
        """
        濞达綀娉曢弫銈咁潰閿濆懐鍩楅柣銊ュ閳ь兛鐒﹀Λ銈囩磽濞差亝顏為柣銊ュ濞存﹢宕撹箛鏃€娈堕柟璇″枟濞肩敻骞忛悢閿嬪€ゆ俊顐熷亾婵炴潙顑呭▍鎺楀Υ?
        Parameters
        ----------
        X : list of str, or numpy array
            閺夊牊鎸搁崣鍡涙儍閸曨噮鍞茬紓浣稿暞閻楅亶寮甸濠勭闂侇偅鑹鹃悥鍫曞及椤栨碍绂堥柛宥呯箲閺嬪啯绂掗幆鎵唴鐎垫澘瀚▓鎴﹀礆濡ゅ嫨鈧啴濡?        """
        # 1. 濞达綀娉曢弫銈夊箵閹哄秵顐介柛鏍ㄧ墱濞堟垿鎮ч悷鎵獧闁圭粯鍔曡ぐ鍥闯椤帞绀夐悘蹇撴濞存﹢宕撹箛姘ギ闁硅婢€鐠愮喖鎮ч悷鎵獧闁告碍鍨块崳?
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="fit")
        extractor = self.feature_extractor
        if extractor is None or not isinstance(extractor, FeatureExtractor):
            raise TypeError("feature_extractor 闊洤鎳橀妴蹇曗偓鍦仧楠?.extract(inputs) -> np.ndarray")

        # Optional: allow extractors to learn normalization/projection from the training set.
        if (not self._feature_extractor_fitted) and isinstance(extractor, FittableFeatureExtractor):
            extractor.fit(x_value, y=y)
            self._feature_extractor_fitted = True

        features = self.feature_extractor.extract(x_value)
        # 2. 濞达綀娉曢弫銈夋偋閻熸壆绐欓柛姘灴閸ｆ椽寮堕妷顭戝敳缂備礁鍟崬鎾焾閵娧勭暠缂備礁绻愰崥鈧俊顐熷亾婵炴潙顑呭▍?
        self.detector.fit(features)
        # 3. 閻忓繐妫滈鍕磼閸愩劌鐎婚柡浣规緲閹挸顫㈤妷銉ョ厒 self.decision_scores_闁挎稑濂旀禍鎺撶瑹鐠恒劌鐓戠紒顐ヮ嚙椤︹晠鎮?
        self.decision_scores_ = self.detector.decision_scores_
        # 4. 閻犲鍟伴弫銈夊春閾忕顫﹂柣銊ュ閺岀喎鈻旈弴顏嗙闁煎浜滄慨鈺冩媼閿涘嫮鏆梻鍐ㄧ墕閳ь剛鍘ч幏浼村冀閸モ晩鍔?
        self._process_decision_scores()
        # Compatibility: enable `predict_proba()` by initializing `_classes`.
        self._set_n_classes(y)

        return self

    def decision_function(self, x: object = MISSING, **kwargs: object):
        x_value = resolve_legacy_x_keyword(x, kwargs, method_name="decision_function")
        # 1. 濞寸姴瀛╅弻濠囧炊閹冨壖濞戞搩鍘借ぐ渚€宕ｉ弽顐㈩棗鐎?
        features = self.feature_extractor.extract(x_value)
        # 2. 闁活潿鍔忛鍕磼閸愩劊鍋ㄩ柣銊ュ缁繝宕楅崨濠庢⒕婵炴潙顑呭▍鎺旀媼閿涘嫮鏆€殿喖鍊搁悥鍫曞礆閸℃ɑ娈?
        return self.detector.decision_function(features)

