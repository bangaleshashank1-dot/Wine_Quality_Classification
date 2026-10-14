import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

from preprocessing import DataProcessor
from model import ModelManager
from prediction import Predictor
from export_report import generate_text_report

app = FastAPI(
    title="Wine Quality Classification API",
    description="Machine Learning API for Logistic Regression & Multinomial Logistic Regression",
    version="1.0.0"
)

# Enable CORS for frontend Vite dev server and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State
data_processor = DataProcessor()
model_manager = ModelManager()
predictor = Predictor(model_manager, data_processor)
preprocess_summary_cache: Dict[str, Any] = {}

# Pydantic Request Models
class SelectTargetRequest(BaseModel):
    target_column: str

class PreprocessRequest(BaseModel):
    remove_duplicates: bool = True
    test_size: float = 0.2
    random_state: int = 42

class TrainRequest(BaseModel):
    threshold: float = 6.0
    max_iter: int = 1000

class PredictRequest(BaseModel):
    features: Dict[str, float]

class LoadSampleRequest(BaseModel):
    sample_type: str = "red"  # "red" or "white"

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "dataset_loaded": data_processor.raw_df is not None,
        "models_trained": model_manager.trained
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Uploads a CSV or Excel (.xlsx/.xls) file and generates dataset summary."""
    filename = file.filename or "uploaded_dataset"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in [".csv", ".xlsx", ".xls", ".txt"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload a CSV or Excel (.xlsx / .xls) file."
        )

    try:
        contents = await file.read()
        summary = data_processor.load_file(contents, filename)
        # Reset previous model training state on new upload
        model_manager.trained = False
        model_manager.binary_results = None
        model_manager.multinomial_results = None
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading dataset: {str(e)}")

@app.post("/load-sample")
def load_sample_dataset(request: LoadSampleRequest):
    """Loads the pre-packaged standard UCI Wine Quality dataset."""
    sample_type = request.sample_type.lower()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if sample_type == "red":
        file_path = os.path.join(base_dir, "data", "winequality-red.csv")
        display_name = "winequality-red.csv"
    elif sample_type == "white":
        file_path = os.path.join(base_dir, "data", "winequality-white.csv")
        display_name = "winequality-white.csv"
    elif sample_type == "red_excel":
        file_path = os.path.join(base_dir, "data", "winequality-red.xlsx")
        display_name = "winequality-red.xlsx"
    else:
        file_path = os.path.join(base_dir, "data", "winequality-red.csv")
        display_name = "winequality-red.csv"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Sample dataset file not found on server.")

    try:
        with open(file_path, "rb") as f:
            contents = f.read()
        summary = data_processor.load_file(contents, display_name)
        model_manager.trained = False
        model_manager.binary_results = None
        model_manager.multinomial_results = None
        return summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(e)}")

@app.get("/dataset")
def get_dataset():
    """Returns current dataset summary, column metadata, and first 10 rows."""
    return data_processor.get_dataset_summary()

@app.get("/dataset/preview")
def get_dataset_preview(page: int = Query(1, ge=1), page_size: int = Query(25, ge=5, le=100), search: str = Query("")):
    """Returns paginated rows of the dataset with search filtering."""
    return data_processor.get_full_data_paginated(page=page, page_size=page_size, search=search)

@app.post("/select-target")
def select_target(request: SelectTargetRequest):
    """Manually select the target column if auto-detection was not 'quality'."""
    try:
        return data_processor.set_target_column(request.target_column)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/preprocess")
def preprocess_data(request: PreprocessRequest):
    """Executes the 6-step data preprocessing pipeline."""
    global preprocess_summary_cache
    if data_processor.raw_df is None:
        raise HTTPException(status_code=400, detail="No dataset uploaded yet. Please upload a dataset first.")

    try:
        result = data_processor.run_preprocessing(
            remove_duplicates=request.remove_duplicates,
            test_size=request.test_size,
            random_state=request.random_state
        )
        preprocess_summary_cache = result
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {str(e)}")

@app.post("/train")
def train_models(request: TrainRequest):
    """Trains Binary Logistic Regression & Multinomial Logistic Regression."""
    global preprocess_summary_cache
    if data_processor.raw_df is None:
        raise HTTPException(status_code=400, detail="Please upload a dataset before training.")

    # Run preprocessing if not already executed
    if data_processor.X_train_scaled is None:
        try:
            preprocess_summary_cache = data_processor.run_preprocessing()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Auto-preprocessing before training failed: {str(e)}")

    try:
        results = model_manager.train_models(
            X_train_scaled=data_processor.X_train_scaled,
            X_test_scaled=data_processor.X_test_scaled,
            y_train=data_processor.y_train,
            y_test=data_processor.y_test,
            threshold=request.threshold,
            max_iter=request.max_iter
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training error: {str(e)}")

@app.get("/results/logistic")
def get_logistic_results():
    """Returns Logistic Regression binary evaluation results."""
    if not model_manager.trained or model_manager.binary_results is None:
        raise HTTPException(status_code=400, detail="Binary Logistic Regression has not been trained yet.")
    return model_manager.binary_results

@app.get("/results/multinomial")
def get_multinomial_results():
    """Returns Multinomial Logistic Regression multi-class evaluation results."""
    if not model_manager.trained or model_manager.multinomial_results is None:
        raise HTTPException(status_code=400, detail="Multinomial Logistic Regression has not been trained yet.")
    return model_manager.multinomial_results

@app.get("/results/comparison")
def get_model_comparison():
    """Returns side-by-side metrics comparison and dynamic conclusion."""
    comparison = model_manager.get_comparison()
    if comparison.get("status") == "not_trained":
        raise HTTPException(status_code=400, detail="Models have not been trained yet.")
    return comparison

@app.post("/predict")
def predict_wine_quality(request: PredictRequest):
    """Predicts wine quality using both Binary and Multinomial models."""
    try:
        return predictor.predict(request.features)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/predict/presets")
def get_prediction_presets():
    """Returns sample feature presets for instant testing."""
    return predictor.get_sample_presets()

@app.get("/visualizations/correlation")
def get_correlation():
    """Returns correlation matrix for numerical features."""
    return data_processor.get_correlation_matrix()

@app.get("/visualizations/distributions")
def get_distributions():
    """Returns histogram distributions for features and target classes."""
    return data_processor.get_feature_distributions()

@app.get("/download/processed")
def download_processed_dataset():
    """Downloads the preprocessed dataset as CSV."""
    csv_str = data_processor.export_processed_csv()
    if not csv_str:
        raise HTTPException(status_code=400, detail="No processed dataset available.")
    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=preprocessed_wine_dataset.csv"}
    )

@app.get("/download/report")
def download_project_report():
    """Downloads comprehensive ML evaluation report as text."""
    dataset_sum = data_processor.get_dataset_summary()
    comparison_sum = model_manager.get_comparison()
    report_text = generate_text_report(
        dataset_summary=dataset_sum,
        preprocess_summary=preprocess_summary_cache,
        comparison_data=comparison_sum,
        binary_res=model_manager.binary_results or {},
        multi_res=model_manager.multinomial_results or {}
    )
    return Response(
        content=report_text,
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=Wine_Quality_ML_Project_Report.txt"}
    )
