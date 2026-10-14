import numpy as np
import pandas as pd
from typing import Dict, Any, List

class Predictor:
    def __init__(self, model_manager, data_processor):
        self.mm = model_manager
        self.dp = data_processor

    def predict(self, feature_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Executes real prediction using trained binary and multinomial models.
        Applies fitted StandardScaler before inference.
        """
        if not self.mm.trained or self.mm.binary_model is None or self.mm.multinomial_model is None:
            raise ValueError("Models are not trained yet. Please train the models before making predictions.")
        
        if self.dp.scaler is None or not self.dp.feature_columns:
            raise ValueError("Preprocessing scaler not found. Please preprocess the dataset first.")

        # Build feature vector in exact column order
        ordered_features = self.dp.feature_columns
        input_vector = []
        missing_features = []

        for feat in ordered_features:
            if feat in feature_values:
                try:
                    val = float(feature_values[feat])
                    input_vector.append(val)
                except (ValueError, TypeError):
                    raise ValueError(f"Feature '{feat}' has invalid non-numeric value: {feature_values[feat]}")
            else:
                missing_features.append(feat)

        if missing_features:
            raise ValueError(f"Missing required feature values: {', '.join(missing_features)}")

        # Convert to DataFrame with feature column names and scale
        X_df = pd.DataFrame([input_vector], columns=ordered_features)
        X_scaled = self.dp.scaler.transform(X_df)

        # -------------------------------------------------------------
        # 1. Binary Prediction
        # -------------------------------------------------------------
        bin_pred_class = int(self.mm.binary_model.predict(X_scaled)[0])
        bin_probs = self.mm.binary_model.predict_proba(X_scaled)[0]
        # In sklearn binary, classes_ are typically [0, 1]
        classes_bin = list(self.mm.binary_model.classes_)
        prob_poor = float(bin_probs[classes_bin.index(0)]) if 0 in classes_bin else 0.0
        prob_good = float(bin_probs[classes_bin.index(1)]) if 1 in classes_bin else 0.0

        bin_result = {
            "prediction": "GOOD" if bin_pred_class == 1 else "POOR",
            "prediction_code": bin_pred_class,
            "threshold": self.mm.binary_threshold,
            "probability_good": round(prob_good, 4),
            "probability_poor": round(prob_poor, 4),
            "confidence_percent": round(max(prob_good, prob_poor) * 100, 1),
            "summary": f"Classified as {'GOOD' if bin_pred_class == 1 else 'POOR'} wine with {round(max(prob_good, prob_poor) * 100, 1)}% confidence."
        }

        # -------------------------------------------------------------
        # 2. Multi-class Prediction
        # -------------------------------------------------------------
        multi_pred_class = self.mm.multinomial_model.predict(X_scaled)[0]
        multi_probs = self.mm.multinomial_model.predict_proba(X_scaled)[0]
        multi_classes = self.mm.multinomial_model.classes_

        class_probabilities = []
        for cls_label, prob in zip(multi_classes, multi_probs):
            prob_float = float(prob)
            class_probabilities.append({
                "quality": str(cls_label),
                "probability": round(prob_float, 4),
                "percent": round(prob_float * 100, 1),
            })

        # Sort classes by quality score
        try:
            class_probabilities.sort(key=lambda x: float(x["quality"]))
        except Exception:
            pass

        multiclass_result = {
            "predicted_quality": int(multi_pred_class) if isinstance(multi_pred_class, (int, np.integer)) else str(multi_pred_class),
            "class_probabilities": class_probabilities,
            "highest_probability": round(float(np.max(multi_probs)) * 100, 1)
        }

        return {
            "status": "success",
            "binary_prediction": bin_result,
            "multiclass_prediction": multiclass_result,
            "inputs": feature_values
        }

    def get_sample_presets(self) -> List[Dict[str, Any]]:
        """Provides realistic sample feature presets for one-click testing."""
        if self.dp.raw_df is None or not self.dp.feature_columns:
            return []
        
        df = self.dp.raw_df
        target_col = self.dp.target_column
        features = self.dp.feature_columns
        
        presets = []
        
        # 1. Dataset Average sample
        avg_dict = {}
        for f in features:
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f]):
                avg_dict[f] = round(float(df[f].mean()), 2)
        presets.append({
            "name": "Dataset Average Wine",
            "description": "Represents the mathematical average of all physicochemical measurements.",
            "values": avg_dict
        })

        # 2. High Quality sample (Quality >= 7 if available)
        if target_col and target_col in df.columns:
            high_subset = df[df[target_col] >= 7]
            if not high_subset.empty:
                high_sample = high_subset.iloc[0]
                high_dict = {f: round(float(high_sample[f]), 2) for f in features if f in high_sample}
                presets.append({
                    "name": "High Quality Wine (Example)",
                    "description": f"Actual sample from dataset having Quality {int(high_sample[target_col])}.",
                    "values": high_dict
                })

            # 3. Economy / Low Quality sample (Quality <= 5)
            low_subset = df[df[target_col] <= 5]
            if not low_subset.empty:
                low_sample = low_subset.iloc[0]
                low_dict = {f: round(float(low_sample[f]), 2) for f in features if f in low_sample}
                presets.append({
                    "name": "Low Quality Wine (Example)",
                    "description": f"Actual sample from dataset having Quality {int(low_sample[target_col])}.",
                    "values": low_dict
                })

        return presets
