import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from typing import Dict, Any, List

class ModelEvaluator:
    @staticmethod
    def evaluate_binary(y_true, y_pred, y_prob) -> Dict[str, Any]:
        """Calculates comprehensive metrics for Binary Logistic Regression."""
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        
        # ROC-AUC calculation
        try:
            auc = float(roc_auc_score(y_true, y_prob))
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            
            # Subsample curve points if too dense (e.g. max 60 points) for fast smooth charting
            if len(fpr) > 60:
                indices = np.linspace(0, len(fpr) - 1, 60, dtype=int)
                roc_points = [
                    {"fpr": round(float(fpr[i]), 4), "tpr": round(float(tpr[i]), 4)}
                    for i in indices
                ]
            else:
                roc_points = [
                    {"fpr": round(float(f), 4), "tpr": round(float(t), 4)}
                    for f, t in zip(fpr, tpr)
                ]
        except Exception:
            auc = 0.5
            roc_points = [{"fpr": 0.0, "tpr": 0.0}, {"fpr": 1.0, "tpr": 1.0}]

        # Confusion Matrix (labels: [0, 1] where 0 is Poor, 1 is Good)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        cm_dict = {
            "matrix": cm.tolist(),
            "labels": ["Poor", "Good"],
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
            "total": int(np.sum(cm)),
            "formatted": [
                {"actual": "Poor", "predicted_poor": int(tn), "predicted_good": int(fp)},
                {"actual": "Good", "predicted_poor": int(fn), "predicted_good": int(tp)},
            ]
        }

        # Classification report
        report_dict = classification_report(
            y_true, y_pred, target_names=["Poor", "Good"], output_dict=True, zero_division=0
        )

        class_metrics = [
            {
                "class": "Poor",
                "precision": round(float(report_dict["Poor"]["precision"]), 4),
                "recall": round(float(report_dict["Poor"]["recall"]), 4),
                "f1": round(float(report_dict["Poor"]["f1-score"]), 4),
                "support": int(report_dict["Poor"]["support"])
            },
            {
                "class": "Good",
                "precision": round(float(report_dict["Good"]["precision"]), 4),
                "recall": round(float(report_dict["Good"]["recall"]), 4),
                "f1": round(float(report_dict["Good"]["f1-score"]), 4),
                "support": int(report_dict["Good"]["support"])
            }
        ]

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "roc_curve": roc_points,
            "confusion_matrix": cm_dict,
            "classification_report": class_metrics,
            "macro_avg": {
                "precision": round(float(report_dict["macro avg"]["precision"]), 4),
                "recall": round(float(report_dict["macro avg"]["recall"]), 4),
                "f1": round(float(report_dict["macro avg"]["f1-score"]), 4),
            },
            "weighted_avg": {
                "precision": round(float(report_dict["weighted avg"]["precision"]), 4),
                "recall": round(float(report_dict["weighted avg"]["recall"]), 4),
                "f1": round(float(report_dict["weighted avg"]["f1-score"]), 4),
            }
        }

    @staticmethod
    def evaluate_multiclass(y_true, y_pred, classes: List[Any]) -> Dict[str, Any]:
        """Calculates comprehensive metrics for Multinomial Logistic Regression."""
        acc = float(accuracy_score(y_true, y_pred))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        # Dynamic multi-class confusion matrix
        labels_list = [c for c in classes]
        labels_str = [str(c) for c in classes]
        cm = confusion_matrix(y_true, y_pred, labels=labels_list)
        
        cm_data = []
        for i, act_label in enumerate(labels_str):
            row_dict = {"actual": f"Quality {act_label}"}
            for j, pred_label in enumerate(labels_str):
                row_dict[f"Quality {pred_label}"] = int(cm[i, j])
            cm_data.append(row_dict)

        cm_dict = {
            "matrix": cm.tolist(),
            "labels": labels_str,
            "formatted": cm_data,
            "total": int(np.sum(cm))
        }

        # Classification report
        report_dict = classification_report(
            y_true, y_pred, labels=labels_list, target_names=labels_str, output_dict=True, zero_division=0
        )

        class_metrics = []
        for cls_label in labels_str:
            if cls_label in report_dict:
                class_metrics.append({
                    "class": f"Quality {cls_label}",
                    "precision": round(float(report_dict[cls_label]["precision"]), 4),
                    "recall": round(float(report_dict[cls_label]["recall"]), 4),
                    "f1": round(float(report_dict[cls_label]["f1-score"]), 4),
                    "support": int(report_dict[cls_label]["support"])
                })

        return {
            "accuracy": round(acc, 4),
            "precision": round(prec_weighted, 4),
            "recall": round(rec_weighted, 4),
            "f1_score": round(f1_weighted, 4),
            "macro_avg": {
                "precision": round(prec_macro, 4),
                "recall": round(rec_macro, 4),
                "f1": round(f1_macro, 4),
            },
            "weighted_avg": {
                "precision": round(prec_weighted, 4),
                "recall": round(rec_weighted, 4),
                "f1": round(f1_weighted, 4),
            },
            "confusion_matrix": cm_dict,
            "classification_report": class_metrics,
            "classes": labels_str
        }
