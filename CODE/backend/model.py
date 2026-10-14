import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
import os
from typing import Dict, Any, Optional
from evaluation import ModelEvaluator

class ModelManager:
    def __init__(self, models_dir: str = "backend/saved_models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.binary_model: Optional[LogisticRegression] = None
        self.multinomial_model: Optional[LogisticRegression] = None
        
        self.binary_threshold: float = 6.0
        self.binary_results: Optional[Dict[str, Any]] = None
        self.multinomial_results: Optional[Dict[str, Any]] = None
        self.multiclass_classes: list = []
        self.trained: bool = False

    def train_models(
        self,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        threshold: float = 6.0,
        max_iter: int = 1000
    ) -> Dict[str, Any]:
        """
        Trains both Binary Logistic Regression and Multinomial Logistic Regression.
        All metrics are calculated dynamically from the predictions.
        """
        self.binary_threshold = threshold
        
        # -------------------------------------------------------------
        # 1. Binary Logistic Regression
        # Convert original quality into Poor (0: < threshold) and Good (1: >= threshold)
        # -------------------------------------------------------------
        y_train_binary = (y_train >= threshold).astype(int)
        y_test_binary = (y_test >= threshold).astype(int)

        self.binary_model = LogisticRegression(
            solver='lbfgs',
            max_iter=max_iter,
            random_state=42
        )
        self.binary_model.fit(X_train_scaled, y_train_binary)
        
        # Predictions & Probabilities
        y_pred_bin = self.binary_model.predict(X_test_scaled)
        
        # Probability of positive class (Good)
        if len(self.binary_model.classes_) == 2:
            y_prob_bin = self.binary_model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob_bin = np.zeros(len(y_test_binary))

        self.binary_results = ModelEvaluator.evaluate_binary(
            y_test_binary, y_pred_bin, y_prob_bin
        )
        self.binary_results["threshold"] = threshold
        self.binary_results["classes"] = ["Poor", "Good"]

        # -------------------------------------------------------------
        # 2. Multinomial Logistic Regression
        # Fits on all unique detected classes using multinomial loss (softmax)
        # -------------------------------------------------------------
        unique_classes = sorted(list(set(y_train.unique()).union(set(y_test.unique()))))
        self.multiclass_classes = unique_classes

        # In modern scikit-learn (1.5+ / 1.8+ / 1.9+), solver='lbfgs' automatically uses multinomial cross-entropy
        self.multinomial_model = LogisticRegression(
            solver='lbfgs',
            max_iter=max_iter,
            random_state=42
        )
        self.multinomial_model.fit(X_train_scaled, y_train)

        y_pred_multi = self.multinomial_model.predict(X_test_scaled)
        
        self.multinomial_results = ModelEvaluator.evaluate_multiclass(
            y_test, y_pred_multi, self.multiclass_classes
        )

        self.trained = True
        
        # Save models to disk
        self.save_models()

        return {
            "status": "success",
            "message": "Both models trained successfully",
            "binary": self.binary_results,
            "multinomial": self.multinomial_results,
            "comparison": self.get_comparison()
        }

    def get_comparison(self) -> Dict[str, Any]:
        """Generates dynamic model comparison and conclusion statement."""
        if not self.trained or not self.binary_results or not self.multinomial_results:
            return {"status": "not_trained"}

        b_acc = self.binary_results["accuracy"]
        b_prec = self.binary_results["precision"]
        b_rec = self.binary_results["recall"]
        b_f1 = self.binary_results["f1_score"]

        m_acc = self.multinomial_results["accuracy"]
        m_prec = self.multinomial_results["precision"]
        m_rec = self.multinomial_results["recall"]
        m_f1 = self.multinomial_results["f1_score"]

        comparison_table = [
            {"metric": "Accuracy", "logistic_regression": b_acc, "multinomial_regression": m_acc},
            {"metric": "Precision", "logistic_regression": b_prec, "multinomial_regression": m_prec},
            {"metric": "Recall", "logistic_regression": b_rec, "multinomial_regression": m_rec},
            {"metric": "F1 Score", "logistic_regression": b_f1, "multinomial_regression": m_f1},
        ]

        # Dynamic best-performing decision
        # Binary classification usually achieves higher raw accuracy because it condenses classes into 2,
        # while Multinomial solves the harder problem of fine-grained quality discrimination.
        if b_f1 > m_f1:
            best_model_name = "Logistic Regression (Binary)"
            best_f1 = b_f1
            diff = round((b_f1 - m_f1) * 100, 2)
            conclusion_text = (
                f"Based on the F1-score ({b_f1:.4f} vs {m_f1:.4f}), Binary Logistic Regression achieved "
                f"a higher overall classification performance on the test set (+{diff}% F1 advantage). "
                f"Binary classification benefits from grouping quality scores into 'Poor' (< {self.binary_threshold}) "
                f"and 'Good' (>= {self.binary_threshold}), reducing boundary ambiguities. "
                f"However, Multinomial Logistic Regression successfully differentiates between all {len(self.multiclass_classes)} "
                f"individual quality levels with {m_acc * 100:.1f}% multi-class accuracy."
            )
        elif m_f1 > b_f1:
            best_model_name = "Multinomial Logistic Regression (Multi-Class)"
            best_f1 = m_f1
            diff = round((m_f1 - b_f1) * 100, 2)
            conclusion_text = (
                f"Based on the weighted F1-score ({m_f1:.4f} vs {b_f1:.4f}), Multinomial Logistic Regression performed "
                f"exceptionally well on this dataset, achieving strong discrimination across all individual quality categories."
            )
        else:
            best_model_name = "Tie (Equal F1-Score)"
            best_f1 = b_f1
            conclusion_text = (
                f"Both Logistic Regression and Multinomial Logistic Regression achieved an identical F1-score of {b_f1:.4f}. "
                f"Binary Logistic Regression offers a clear high-level quality boundary, while Multinomial Regression provides granular quality scoring."
            )

        return {
            "comparison_table": comparison_table,
            "best_model": best_model_name,
            "best_f1_score": best_f1,
            "conclusion": conclusion_text,
            "binary_threshold": self.binary_threshold,
            "num_multiclass_classes": len(self.multiclass_classes),
            "multiclass_classes": [str(c) for c in self.multiclass_classes]
        }

    def save_models(self):
        """Persists trained models to disk."""
        if self.binary_model:
            joblib.dump(self.binary_model, os.path.join(self.models_dir, "binary_logistic.joblib"))
        if self.multinomial_model:
            joblib.dump(self.multinomial_model, os.path.join(self.models_dir, "multinomial_logistic.joblib"))

    def load_models(self):
        """Loads models if they exist on disk."""
        bin_path = os.path.join(self.models_dir, "binary_logistic.joblib")
        multi_path = os.path.join(self.models_dir, "multinomial_logistic.joblib")
        if os.path.exists(bin_path):
            self.binary_model = joblib.load(bin_path)
        if os.path.exists(multi_path):
            self.multinomial_model = joblib.load(multi_path)
