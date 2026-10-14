import os
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple, Optional, List

class DataProcessor:
    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.processed_df: Optional[pd.DataFrame] = None
        self.filename: str = ""
        self.file_size: int = 0
        self.target_column: Optional[str] = None
        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        
        # Preprocessing metadata
        self.rows_before: int = 0
        self.rows_after: int = 0
        self.duplicates_removed: int = 0
        self.missing_imputed: Dict[str, Any] = {}
        
        # Train / Test splits
        self.X_train = None
        self.X_test = None
        self.X_train_scaled = None
        self.X_test_scaled = None
        self.y_train = None
        self.y_test = None

    def load_file(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Loads CSV or Excel (.xlsx/.xls) into a pandas DataFrame."""
        self.filename = filename
        self.file_size = len(file_bytes)
        ext = os.path.splitext(filename)[1].lower()
        if '.xlsx' in filename.lower():
            ext = '.xlsx'
        elif '.xls' in filename.lower():
            ext = '.xls'
        elif '.csv' in filename.lower():
            ext = '.csv'
        
        if ext in ['.xlsx', '.xls']:
            try:
                self.raw_df = pd.read_excel(io.BytesIO(file_bytes))
            except Exception as e:
                raise ValueError(f"Failed to read Excel file: {str(e)}")
        elif ext in ['.csv', '.txt']:
            try:
                # Use python engine with sep=None to auto-detect delimiters like comma, semicolon, tab
                self.raw_df = pd.read_csv(io.BytesIO(file_bytes), sep=None, engine='python')
            except Exception:
                try:
                    self.raw_df = pd.read_csv(io.BytesIO(file_bytes), delimiter=';')
                except Exception as e:
                    raise ValueError(f"Failed to read CSV file: {str(e)}")
        else:
            raise ValueError(f"Unsupported file format '{ext}'. Please upload .csv, .xlsx, or .xls")

        # Strip whitespace from column names
        self.raw_df.columns = [str(c).strip() for c in self.raw_df.columns]
        
        # Detect target column
        self.detect_target_column()
        
        return self.get_dataset_summary()

    def detect_target_column(self) -> Optional[str]:
        """Case-insensitive target column detection for 'quality'."""
        if self.raw_df is None:
            return None
        
        for col in self.raw_df.columns:
            if col.strip().lower() == 'quality':
                self.target_column = col
                return col
        
        self.target_column = None
        return None

    def set_target_column(self, col_name: str) -> Dict[str, Any]:
        """Manually specify the target column."""
        if self.raw_df is None:
            raise ValueError("No dataset loaded.")
        if col_name not in self.raw_df.columns:
            raise ValueError(f"Column '{col_name}' not found in dataset.")
        
        self.target_column = col_name
        return self.get_dataset_summary()

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Returns comprehensive summary metadata of the uploaded dataset."""
        if self.raw_df is None:
            return {"status": "empty", "message": "No dataset uploaded yet"}
        
        df = self.raw_df
        rows, cols = df.shape
        missing_total = int(df.isnull().sum().sum())
        duplicates_count = int(df.duplicated().sum())
        
        column_info = []
        for col in df.columns:
            dtype_str = str(df[col].dtype)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            col_missing = int(df[col].isnull().sum())
            col_unique = int(df[col].nunique())
            
            c_info = {
                "name": col,
                "dtype": dtype_str,
                "is_numeric": is_numeric,
                "missing": col_missing,
                "unique": col_unique,
                "mean": round(float(df[col].mean()), 4) if is_numeric and df[col].count() > 0 else None,
                "min": round(float(df[col].min()), 4) if is_numeric and df[col].count() > 0 else None,
                "max": round(float(df[col].max()), 4) if is_numeric and df[col].count() > 0 else None,
                "std": round(float(df[col].std()), 4) if is_numeric and df[col].count() > 1 else None,
            }
            column_info.append(c_info)

        # Target class distribution if target is detected
        classes = []
        class_counts = {}
        if self.target_column and self.target_column in df.columns:
            unique_classes = sorted(df[self.target_column].dropna().unique().tolist())
            classes = [int(c) if isinstance(c, (int, np.integer)) else c for c in unique_classes]
            counts = df[self.target_column].value_counts().to_dict()
            class_counts = {str(k): int(v) for k, v in counts.items()}

        features = [c for c in df.columns if c != self.target_column]

        return {
            "status": "loaded",
            "filename": self.filename,
            "file_size": self.file_size,
            "rows": rows,
            "columns": cols,
            "missing_total": missing_total,
            "duplicates_count": duplicates_count,
            "target_column": self.target_column,
            "target_found": self.target_column is not None,
            "classes": classes,
            "class_counts": class_counts,
            "num_classes": len(classes),
            "features": features,
            "num_features": len(features),
            "column_info": column_info,
            "first_10_rows": df.head(10).replace({np.nan: None}).to_dict(orient="records"),
        }

    def get_full_data_paginated(self, page: int = 1, page_size: int = 25, search: str = "") -> Dict[str, Any]:
        """Provides paginated rows of the raw/processed dataset with search support."""
        if self.raw_df is None:
            return {"rows": [], "total": 0, "page": page, "page_size": page_size}
        
        df = self.raw_df.copy()
        if search:
            # Simple substring filter across all columns
            mask = df.astype(str).apply(lambda row: row.str.contains(search, case=False).any(), axis=1)
            df = df[mask]
        
        total = len(df)
        start = (page - 1) * page_size
        end = start + page_size
        subset = df.iloc[start:end].replace({np.nan: None})
        
        return {
            "rows": subset.to_dict(orient="records"),
            "columns": df.columns.tolist(),
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": int(np.ceil(total / page_size)) if total > 0 else 1
        }

    def run_preprocessing(self, remove_duplicates: bool = True, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Executes the 6-step preprocessing pipeline:
        Step 1: Check & impute missing values (median for numeric, mode for categorical)
        Step 2: Remove duplicate rows
        Step 3: Separate features X and target y
        Step 4: Categorical encoding if necessary
        Step 5: 80-20 Train-Test split
        Step 6: StandardScaler normalization
        """
        if self.raw_df is None:
            raise ValueError("No dataset loaded.")
        if not self.target_column or self.target_column not in self.raw_df.columns:
            raise ValueError("Target column not identified. Please specify target column.")

        df = self.raw_df.copy()
        self.rows_before = len(df)
        
        # Step 1: Missing values imputation
        self.missing_imputed = {}
        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            if missing_count > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)
                    self.missing_imputed[col] = {"strategy": "median", "value": round(float(fill_val), 4), "imputed_rows": missing_count}
                else:
                    fill_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                    df[col] = df[col].fillna(fill_val)
                    self.missing_imputed[col] = {"strategy": "mode", "value": str(fill_val), "imputed_rows": missing_count}

        # Step 2: Remove duplicates
        if remove_duplicates:
            self.duplicates_removed = int(df.duplicated().sum())
            df = df.drop_duplicates().reset_index(drop=True)
        else:
            self.duplicates_removed = 0
            
        self.rows_after = len(df)
        self.processed_df = df.copy()

        # Step 3: Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # Step 4: Handle any categorical feature encoding
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            
        # Ensure all features in X are numeric
        X = X.astype(float)
        self.feature_columns = X.columns.tolist()

        # Step 5: Train / Test Split
        # Try stratified split if possible, fallback to standard split if class count is small
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except Exception:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        # Step 6: Feature scaling with StandardScaler
        self.scaler = StandardScaler()
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        X_te_scaled = self.scaler.transform(X_te)

        self.X_train = X_tr
        self.X_test = X_te
        self.X_train_scaled = X_tr_scaled
        self.X_test_scaled = X_te_scaled
        self.y_train = y_tr
        self.y_test = y_te

        # Compute scaling parameters (mean, scale/std)
        scaling_params = {}
        for idx, col in enumerate(self.feature_columns):
            scaling_params[col] = {
                "mean": round(float(self.scaler.mean_[idx]), 4),
                "std": round(float(self.scaler.scale_[idx]), 4)
            }

        return {
            "status": "preprocessed",
            "rows_before": self.rows_before,
            "rows_after": self.rows_after,
            "duplicates_removed": self.duplicates_removed,
            "missing_imputed": self.missing_imputed,
            "train_samples": len(X_tr),
            "test_samples": len(X_te),
            "test_size_percent": int(test_size * 100),
            "train_size_percent": int((1 - test_size) * 100),
            "feature_columns": self.feature_columns,
            "num_features": len(self.feature_columns),
            "scaling_params": scaling_params,
            "categorical_encoded": categorical_cols,
        }

    def get_correlation_matrix(self) -> Dict[str, Any]:
        """Calculates correlation matrix for numerical columns."""
        if self.raw_df is None:
            return {"columns": [], "matrix": []}
        
        df = self.processed_df if self.processed_df is not None else self.raw_df
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {"columns": [], "matrix": []}

        corr = numeric_df.corr().round(3)
        cols = corr.columns.tolist()
        matrix = []
        for i, row_col in enumerate(cols):
            row_vals = []
            for j, col_col in enumerate(cols):
                row_vals.append(float(corr.iloc[i, j]))
            matrix.append({"name": row_col, "values": row_vals})

        return {
            "columns": cols,
            "matrix": matrix
        }

    def get_feature_distributions(self) -> Dict[str, Any]:
        """Computes histogram bins for all numeric features + target class distribution."""
        if self.raw_df is None:
            return {"features": {}, "target_distribution": []}
        
        df = self.processed_df if self.processed_df is not None else self.raw_df
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        feature_dists = {}
        for col in numeric_cols:
            if col == self.target_column:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            counts, bin_edges = np.histogram(series, bins=10)
            bins_data = []
            for k in range(len(counts)):
                label = f"{round(bin_edges[k], 2)}-{round(bin_edges[k+1], 2)}"
                bins_data.append({"bin": label, "count": int(counts[k]), "min": float(bin_edges[k]), "max": float(bin_edges[k+1])})
            feature_dists[col] = bins_data

        target_distribution = []
        if self.target_column and self.target_column in df.columns:
            counts = df[self.target_column].value_counts().sort_index()
            for cls_val, cnt in counts.items():
                target_distribution.append({
                    "quality": int(cls_val) if isinstance(cls_val, (int, np.integer)) else str(cls_val),
                    "count": int(cnt)
                })

        return {
            "features": feature_dists,
            "target_distribution": target_distribution
        }

    def export_processed_csv(self) -> str:
        """Returns the preprocessed data as a CSV string."""
        if self.processed_df is None:
            if self.raw_df is not None:
                return self.raw_df.to_csv(index=False)
            return ""
        return self.processed_df.to_csv(index=False)
