import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

# 导入三个检测器类
from IsolationForest import IsolationForestStructureDetector
from LOF import LOFStructureAnomalyDetector
from SSIM_struct import MultiTemplatePopupStructDetector
from tqdm import tqdm


class EnsembleVotingDetector:
    """
    集成投票检测器
    整合三个模型的预测结果：
    - IsolationForest (结构异常检测)
    - SSIM Template (模板匹配+弹窗检测)
    - LOF (局部异常因子)

    规则：
    1) SSIM 优先：如果 SSIM 判定“正常”，直接输出正常并返回。
    2) 否则进入加权多数(>=50%)：对当次有效模型做权重归一后投票。
    """

    def __init__(
        self,
        iforest_model_path: str = "iforest_structure_detector.pkl",
        ssim_model_path: str = "multi_template_popup_detector.pkl",
        lof_model_path: str = "lof_structure_detector.pkl",
        weights: Optional[Dict[str, float]] = None,
    ):
        self.model_paths = {
            "iforest": iforest_model_path,
            "ssim": ssim_model_path,
            "lof": lof_model_path,
        }

        # 默认权重（简单投票时各模型权重相等）
        self.weights = weights or {"iforest": 1.0, "ssim": 1.0, "lof": 1.0}

        self.models: Dict[str, Optional[object]] = {}
        self.is_loaded = False
        self._load_all_models()

    def _loaded_model_names(self) -> List[str]:
        return [name for name, model in self.models.items() if model is not None]

    def _load_named_model(self, model_name: str, banner: str, detector_cls, success_label: str) -> None:
        print(f"\n{banner}")
        model_path = self.model_paths[model_name]
        if not os.path.exists(model_path):
            print(f"   ✗ {success_label}文件不存在: {model_path}")
            self.models[model_name] = None
            return

        detector = detector_cls()
        detector.load(model_path)
        self.models[model_name] = detector
        print(f"   ✓ {success_label}加载成功: {model_path}")

    def _normalize_loaded_model_weights(self, loaded_models: List[str]) -> None:
        if len(loaded_models) >= len(self.models):
            return

        print("\n调整模型权重（仅对已加载模型归一）...")
        total_weight = sum(self.weights.get(name, 1.0) for name in loaded_models)
        for name, model in self.models.items():
            if model is None:
                self.weights[name] = 0.0
                print(f"   {name} 权重设为 0")
        if total_weight <= 0:
            return
        for name in loaded_models:
            self.weights[name] = self.weights.get(name, 1.0) / total_weight
            print(f"   {name} 权重调整为 {self.weights[name]:.3f}")

    def _load_all_models(self):
        print("=== 开始加载集成投票模型 ===")
        try:
            self._load_named_model(
                "iforest",
                "1. 加载IsolationForest模型...",
                IsolationForestStructureDetector,
                "IsolationForest模型",
            )
            self._load_named_model(
                "ssim",
                "2. 加载SSIM模板检测模型...",
                MultiTemplatePopupStructDetector,
                "SSIM模型",
            )
            self._load_named_model(
                "lof",
                "3. 加载LOF模型...",
                LOFStructureAnomalyDetector,
                "LOF模型",
            )

            loaded_models = self._loaded_model_names()
            if not loaded_models:
                raise ValueError("没有成功加载任何模型！")

            print("\n=== 模型加载完成 ===")
            print(
                f"成功加载 {len(loaded_models)}/{len(self.models)} 个模型: {', '.join(loaded_models)}"
            )

            self._normalize_loaded_model_weights(loaded_models)
            self.is_loaded = True

        except Exception as e:
            print(f"\n加载模型时出错: {e}")
            raise

    def _init_prediction_result(self, image_path: str) -> Dict:
        return {
            "image": os.path.basename(image_path),
            "model_predictions": {},
            "votes": {"normal": 0, "anomaly": 0},
            "weighted_score": 0.0,
            "is_normal": None,
            "prediction": None,
            "confidence": 0.0,
            "anomaly_types": [],
        }

    def _build_prediction_entry(self, pred_raw: Dict, return_details: bool) -> Dict:
        return {
            "is_normal": pred_raw["is_normal"],
            "prediction": pred_raw["prediction"],
            "confidence": float(pred_raw.get("confidence", 0.5)),
            "raw_result": pred_raw if return_details else None,
        }

    def _append_anomaly_type(self, results: Dict, model_name: str, pred_raw: Dict) -> None:
        if model_name == "ssim":
            if pred_raw.get("has_popup"):
                results["anomaly_types"].append(f"检测到{pred_raw.get('popup_count', 0)}个弹窗")
            else:
                results["anomaly_types"].append("界面结构异常")
            return

        if model_name == "iforest":
            results["anomaly_types"].append(f"结构异常({pred_raw.get('anomaly_level', '未知')})")
            return

        if model_name == "lof":
            results["anomaly_types"].append(f"局部异常({pred_raw.get('anomaly_level', '未知')})")

    def _record_prediction(self, results: Dict, model_name: str, pred_raw: Dict, return_details: bool) -> Dict:
        pred_std = self._build_prediction_entry(pred_raw, return_details)
        results["model_predictions"][model_name] = pred_std
        vote_key = "normal" if pred_raw["is_normal"] else "anomaly"
        results["votes"][vote_key] += 1
        if not pred_raw["is_normal"]:
            self._append_anomaly_type(results, model_name, pred_raw)
        return pred_std

    def _record_prediction_error(self, results: Dict, model_name: str, error: Exception) -> None:
        print(f"模型 {model_name} 预测出错: {error}")
        results["model_predictions"][model_name] = {"error": str(error), "is_normal": None}

    def _loaded_model_count(self) -> int:
        return len(self._loaded_model_names())

    def _apply_ssim_short_circuit(self, results: Dict, ssim_pred: Dict) -> Dict:
        results["weighted_score"] = 1.0
        results["is_normal"] = True
        results["prediction"] = "正常"
        results["confidence"] = ssim_pred["confidence"]
        results["summary"] = {
            "total_models_loaded": self._loaded_model_count(),
            "total_models_valid": 1,
            "votes_normal": results["votes"]["normal"],
            "votes_anomaly": results["votes"]["anomaly"],
            "weighted_score": results["weighted_score"],
            "decision_rule": "SSIM优先：SSIM判正常直接返回",
        }
        return results

    def _run_ssim_priority_prediction(self, results: Dict, image_path: str, return_details: bool) -> Dict | None:
        ssim_model = self.models.get("ssim")
        if ssim_model is None:
            return None

        try:
            ssim_raw = ssim_model.predict(image_path)
            ssim_pred = self._record_prediction(results, "ssim", ssim_raw, return_details)
            if ssim_raw["is_normal"]:
                return self._apply_ssim_short_circuit(results, ssim_pred)
        except Exception as error:
            self._record_prediction_error(results, "ssim", error)

        return None

    def _run_secondary_predictions(self, results: Dict, image_path: str, return_details: bool) -> None:
        for model_name in ("iforest", "lof"):
            model = self.models.get(model_name)
            if model is None:
                continue
            try:
                pred_raw = model.predict(image_path)
                self._record_prediction(results, model_name, pred_raw, return_details)
            except Exception as error:
                self._record_prediction_error(results, model_name, error)

    def _valid_predictions(self, results: Dict) -> List:
        return [
            (name, pred)
            for name, pred in results["model_predictions"].items()
            if pred.get("is_normal") is not None and "error" not in pred
        ]

    def _effective_weights(self, valid: List) -> Dict[str, float]:
        total_weight = sum(self.weights.get(name, 1.0) for name, _ in valid)
        if total_weight <= 0:
            return {name: 1.0 / len(valid) for name, _ in valid}
        return {name: self.weights.get(name, 1.0) / total_weight for name, _ in valid}

    def _finalize_weighted_prediction(self, results: Dict) -> Dict:
        valid = self._valid_predictions(results)
        if not valid:
            raise ValueError("没有模型成功预测！")

        eff_w = self._effective_weights(valid)
        normal_w = sum(eff_w[name] for name, pred in valid if pred["is_normal"])
        results["weighted_score"] = float(normal_w)
        results["is_normal"] = normal_w >= 0.5
        results["prediction"] = "正常" if results["is_normal"] else "异常"

        consistency = abs(normal_w - 0.5) * 2
        avg_conf = 0.0
        for name, pred in valid:
            confidence = float(pred.get("confidence", 0.5))
            agrees_with_result = pred["is_normal"] == results["is_normal"]
            avg_conf += eff_w[name] * (confidence if agrees_with_result else (1 - confidence))
        results["confidence"] = float(0.6 * consistency + 0.4 * avg_conf)
        results["summary"] = {
            "total_models_loaded": self._loaded_model_count(),
            "total_models_valid": len(valid),
            "votes_normal": results["votes"]["normal"],
            "votes_anomaly": results["votes"]["anomaly"],
            "weighted_score": results["weighted_score"],
            "decision_rule": "SSIM优先；若SSIM异常或不可用→加权多数(>=50%)",
        }
        results["anomaly_types"] = list(set(results["anomaly_types"]))
        return results

    def predict(self, image_path: str, return_details: bool = True) -> Dict:
        if not self.is_loaded:
            raise ValueError("模型未加载！")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")

        results = self._init_prediction_result(image_path)
        ssim_result = self._run_ssim_priority_prediction(results, image_path, return_details)
        if ssim_result is not None:
            return ssim_result

        self._run_secondary_predictions(results, image_path, return_details)
        return self._finalize_weighted_prediction(results)

    def _convert_numpy_types(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(v) for v in obj)
        else:
            return obj

    def _list_jpg_files(self, test_folder: str) -> List[str]:
        return [filename for filename in os.listdir(test_folder) if filename.lower().endswith(".jpg")]

    def _print_batch_header(self, test_folder: str, jpg_files: List[str]) -> None:
        print("\n=== 开始批量检测 ===")
        print(f"测试目录: {test_folder}")
        print(f"图片数量: {len(jpg_files)}")
        print(f"使用模型: {', '.join(self._loaded_model_names())}")

    def _update_model_agreement_stats(self, stats: Dict[str, int], weighted_score: float) -> None:
        if weighted_score >= 0.999 or weighted_score <= 0.001:
            stats["full"] += 1
            return
        if weighted_score > 0.5:
            stats["majority"] += 1
            return
        stats["split"] += 1

    def _process_batch_image(
        self,
        test_folder: str,
        filename: str,
        results: List[Dict],
        anomalies: List[Dict],
        model_agreement_stats: Dict[str, int],
    ) -> None:
        try:
            img_path = os.path.join(test_folder, filename)
            result = self.predict(img_path, return_details=False)
            results.append(result)
            if not result["is_normal"]:
                anomalies.append(result)
            self._update_model_agreement_stats(
                model_agreement_stats,
                result.get("weighted_score", 0.0),
            )
        except Exception as error:
            print(f"\n检测 {filename} 出错: {error}")

    def _summarize_results(self, results: List[Dict]) -> tuple[int, int, int]:
        total = len(results)
        normal = sum(1 for result in results if result["is_normal"])
        anomaly = total - normal
        return total, normal, anomaly

    def _print_batch_summary(
        self,
        total: int,
        normal: int,
        anomaly: int,
        model_agreement_stats: Dict[str, int],
    ) -> None:
        print("\n=== 检测结果统计 ===")
        print(f"总数: {total}")
        print(f"正常: {normal} ({(normal / total * 100) if total else 0:.1f}%)")
        print(f"异常: {anomaly} ({(anomaly / total * 100) if total else 0:.1f}%)")

        print("\n=== 模型一致性统计（基于加权得分）===")
        if total <= 0:
            return
        print(f"完全一致: {model_agreement_stats['full']} ({model_agreement_stats['full'] / total * 100:.1f}%)")
        print(f"多数一致: {model_agreement_stats['majority']} ({model_agreement_stats['majority'] / total * 100:.1f}%)")
        print(f"意见分歧: {model_agreement_stats['split']} ({model_agreement_stats['split'] / total * 100:.1f}%)")

    def _collect_model_stats(self, results: List[Dict]) -> Dict:
        model_stats = {
            name: {"normal": 0, "anomaly": 0, "error": 0}
            for name in self.models
            if self.models[name] is not None
        }
        for result in results:
            for model_name, pred in result["model_predictions"].items():
                if "error" in pred:
                    model_stats[model_name]["error"] += 1
                elif pred.get("is_normal") is True:
                    model_stats[model_name]["normal"] += 1
                elif pred.get("is_normal") is False:
                    model_stats[model_name]["anomaly"] += 1
        return model_stats

    def _print_model_stats(self, model_stats: Dict) -> None:
        print("\n=== 各模型检测统计（未加权，仅参考）===")
        for model_name, stats in model_stats.items():
            total_pred = stats["normal"] + stats["anomaly"]
            if total_pred <= 0 and stats["error"] <= 0:
                continue
            print(f"\n{model_name}:")
            if total_pred > 0:
                print(f"  正常: {stats['normal']} ({stats['normal'] / total_pred * 100:.1f}%)")
                print(f"  异常: {stats['anomaly']} ({stats['anomaly'] / total_pred * 100:.1f}%)")
            if stats["error"] > 0:
                print(f"  错误: {stats['error']}")

    def _count_anomaly_types(self, anomalies: List[Dict]) -> Dict[str, int]:
        anomaly_type_count: Dict[str, int] = {}
        for anomaly in anomalies:
            for anomaly_type in anomaly.get("anomaly_types", []):
                anomaly_type_count[anomaly_type] = anomaly_type_count.get(anomaly_type, 0) + 1
        return anomaly_type_count

    def _print_anomaly_distribution(self, anomaly_type_count: Dict[str, int], anomaly_total: int) -> None:
        if not anomaly_type_count or anomaly_total <= 0:
            return

        print("\n=== 异常类型分布 ===")
        for anomaly_type, count in sorted(
            anomaly_type_count.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"{anomaly_type}: {count} ({count / anomaly_total * 100:.1f}%)")

    def _save_batch_results(
        self,
        test_folder: str,
        results: List[Dict],
        total: int,
        normal: int,
        anomaly: int,
        model_agreement_stats: Dict[str, int],
        model_stats: Dict,
        anomaly_type_count: Dict[str, int],
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = f"ensemble_results_{timestamp}.json"
        save_data = {
            "test_folder": test_folder,
            "timestamp": timestamp,
            "statistics": {
                "total": total,
                "normal": normal,
                "anomaly": anomaly,
                "model_agreement": model_agreement_stats,
                "model_stats": model_stats,
                "anomaly_types": anomaly_type_count,
            },
            "model_config": {
                "models": list(self.models.keys()),
                "weights": self.weights,
                "model_paths": self.model_paths,
            },
            "results": results,
        }
        save_data = self._convert_numpy_types(save_data)
        with open(result_file, "w", encoding="utf-8") as file_obj:
            json.dump(save_data, file_obj, ensure_ascii=False, indent=2)
        print(f"\n结果已保存到: {result_file}")

    def batch_predict(self, test_folder: str, save_results: bool = True) -> List[Dict]:
        if not os.path.exists(test_folder):
            raise FileNotFoundError(f"测试目录不存在: {test_folder}")

        results = []
        anomalies = []
        model_agreement_stats = {"full": 0, "majority": 0, "split": 0}

        jpg_files = self._list_jpg_files(test_folder)
        if not jpg_files:
            print(f"在 {test_folder} 中没有找到jpg文件")
            return results

        self._print_batch_header(test_folder, jpg_files)

        for filename in tqdm(jpg_files, desc="检测进度"):
            self._process_batch_image(
                test_folder,
                filename,
                results,
                anomalies,
                model_agreement_stats,
            )

        n_total, n_normal, n_anomaly = self._summarize_results(results)
        self._print_batch_summary(n_total, n_normal, n_anomaly, model_agreement_stats)

        model_stats = self._collect_model_stats(results)
        self._print_model_stats(model_stats)

        anomaly_type_count = self._count_anomaly_types(anomalies)
        self._print_anomaly_distribution(anomaly_type_count, n_anomaly)

        if save_results:
            self._save_batch_results(
                test_folder,
                results,
                n_total,
                n_normal,
                n_anomaly,
                model_agreement_stats,
                model_stats,
                anomaly_type_count,
            )

        return results

    def get_model_info(self) -> Dict:
        info = {
            "loaded_models": [],
            "model_paths": self.model_paths,
            "weights": self.weights,
            "is_loaded": self.is_loaded,
        }
        for name, model in self.models.items():
            if model is not None:
                model_info = {
                    "name": name,
                    "type": type(model).__name__,
                    "is_trained": getattr(model, "is_trained", False),
                }
                if name == "iforest" and hasattr(model, "iforest"):
                    model_info["n_estimators"] = getattr(model.iforest, "n_estimators", None)
                    model_info["contamination"] = getattr(model.iforest, "contamination", None)
                elif name == "ssim":
                    model_info["num_templates"] = len(getattr(model, "templates", {}))
                    model_info["structure_threshold"] = getattr(model, "structure_threshold", None)
                elif name == "lof" and hasattr(model, "lof"):
                    model_info["n_neighbors"] = getattr(model.lof, "n_neighbors", None)
                    model_info["contamination"] = getattr(model.lof, "contamination", None)
                info["loaded_models"].append(model_info)
        return info


# 使用示例
if __name__ == "__main__":
    ensemble = EnsembleVotingDetector(
        iforest_model_path="iforest_structure_detector.pkl",
        ssim_model_path="multi_template_popup_detector.pkl",
        lof_model_path="lof_structure_detector.pkl",
        weights={"iforest": 1.0, "ssim": 1.0, "lof": 1.0},
    )

    print("\n=== 模型信息 ===")
    info = ensemble.get_model_info()
    print(f"已加载模型: {len(info['loaded_models'])}")
    for model in info["loaded_models"]:
        print(f"  - {model['name']} ({model['type']})")

    try:
        test_image = "/data/temp11/正常/青春版/0dd011082dad48e6a782c9b4ea6ee5ea.jpg"
        if os.path.exists(test_image):
            print(f"\n=== 测试单张图片: {test_image} ===")
            result = ensemble.predict(test_image)

            print(f"\n最终预测: {result['prediction']}")
            print(f"置信度: {result['confidence']:.3f}")
            print(
                f"投票(未加权): 正常={result['votes']['normal']}, 异常={result['votes']['anomaly']}"
            )
            print(f"加权得分(支持正常的权重占比): {result['weighted_score']:.3f}")

            print("\n各模型预测:")
            for model_name, pred in result["model_predictions"].items():
                if "error" not in pred:
                    print(
                        f"  - {model_name}: {pred['prediction']} (置信度: {pred['confidence']:.3f})"
                    )

            if result["anomaly_types"]:
                print(f"\n异常类型: {', '.join(result['anomaly_types'])}")
    except Exception as e:
        print(f"测试单张图片出错: {e}")
