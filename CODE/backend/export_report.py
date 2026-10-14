import datetime
from typing import Dict, Any

def generate_text_report(dataset_summary: Dict[str, Any], preprocess_summary: Dict[str, Any], comparison_data: Dict[str, Any], binary_res: Dict[str, Any], multi_res: Dict[str, Any]) -> str:
    """Generates a professional text report of the ML analysis."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("================================================================================")
    report.append("             WINE QUALITY CLASSIFICATION MACHINE LEARNING REPORT")
    report.append("   Logistic Regression (Binary) & Multinomial Logistic Regression (Multi-Class)")
    report.append("================================================================================")
    report.append(f"Generated On : {now_str}")
    report.append(f"Dataset File : {dataset_summary.get('filename', 'N/A')}")
    report.append(f"Target Col   : {dataset_summary.get('target_column', 'quality')}")
    report.append(f"Total Rows   : {dataset_summary.get('rows', 'N/A')}")
    report.append(f"Total Cols   : {dataset_summary.get('columns', 'N/A')}")
    report.append(f"Classes Found: {', '.join([str(c) for c in dataset_summary.get('classes', [])])}")
    report.append("--------------------------------------------------------------------------------\n")
    
    report.append("1. DATA PREPROCESSING SUMMARY")
    report.append("-----------------------------")
    report.append(f"• Rows Before Preprocessing: {preprocess_summary.get('rows_before', 'N/A')}")
    report.append(f"• Duplicates Removed        : {preprocess_summary.get('duplicates_removed', 'N/A')}")
    report.append(f"• Rows After Preprocessing : {preprocess_summary.get('rows_after', 'N/A')}")
    report.append(f"• Train / Test Split Ratio : {preprocess_summary.get('train_size_percent', 80)}% / {preprocess_summary.get('test_size_percent', 20)}%")
    report.append(f"• Training Samples         : {preprocess_summary.get('train_samples', 'N/A')}")
    report.append(f"• Testing Samples          : {preprocess_summary.get('test_samples', 'N/A')}")
    report.append(f"• Feature Scaler           : StandardScaler (zero mean, unit variance)")
    report.append(f"• Features Evaluated       : {', '.join(preprocess_summary.get('feature_columns', []))}")
    report.append("\n")

    report.append("2. BINARY CLASSIFICATION (LOGISTIC REGRESSION)")
    report.append("----------------------------------------------")
    if binary_res:
        report.append(f"• Target Partitioning      : Poor (< {binary_res.get('threshold', 6)}) vs Good (>= {binary_res.get('threshold', 6)})")
        report.append(f"• Accuracy Score           : {binary_res.get('accuracy', 0.0) * 100:.2f}%")
        report.append(f"• Precision                : {binary_res.get('precision', 0.0) * 100:.2f}%")
        report.append(f"• Recall                   : {binary_res.get('recall', 0.0) * 100:.2f}%")
        report.append(f"• F1-Score                 : {binary_res.get('f1_score', 0.0):.4f}")
        report.append(f"• ROC-AUC Score            : {binary_res.get('roc_auc', 0.0):.4f}")
        cm = binary_res.get('confusion_matrix', {})
        report.append(f"• Confusion Matrix (TP/FP/TN/FN): TP={cm.get('tp')}, FP={cm.get('fp')}, TN={cm.get('tn')}, FN={cm.get('fn')}")
    else:
        report.append("• Model not trained yet.")
    report.append("\n")

    report.append("3. MULTI-CLASS CLASSIFICATION (MULTINOMIAL LOGISTIC REGRESSION)")
    report.append("---------------------------------------------------------------")
    if multi_res:
        report.append(f"• Classification Strategy   : Multinomial Softmax Cross-Entropy Loss")
        report.append(f"• Classes Predicted        : {', '.join(multi_res.get('classes', []))}")
        report.append(f"• Overall Accuracy         : {multi_res.get('accuracy', 0.0) * 100:.2f}%")
        report.append(f"• Weighted Precision       : {multi_res.get('precision', 0.0) * 100:.2f}%")
        report.append(f"• Weighted Recall          : {multi_res.get('recall', 0.0) * 100:.2f}%")
        report.append(f"• Weighted F1-Score        : {multi_res.get('f1_score', 0.0):.4f}")
        report.append(f"• Macro F1-Score           : {multi_res.get('macro_avg', {}).get('f1', 0.0):.4f}")
    else:
        report.append("• Model not trained yet.")
    report.append("\n")

    report.append("4. MODEL COMPARISON & VERDICT")
    report.append("-----------------------------")
    if comparison_data and comparison_data.get('status') != 'not_trained':
        report.append(f"• Best Performing Model    : {comparison_data.get('best_model')}")
        report.append(f"• Highest F1-Score         : {comparison_data.get('best_f1_score')}")
        report.append(f"• Comparative Conclusion   :\n  {comparison_data.get('conclusion')}")
    else:
        report.append("• Models not trained.")
    report.append("\n")

    report.append("5. PROJECT METADATA")
    report.append("-------------------")
    report.append("Project Title: Wine Quality Classification Using Logistic Regression and Multinomial Logistic Regression")
    report.append("Degree       : Bachelor of Technology (B.Tech)")
    report.append("Course       : Machine Learning")
    report.append("Academic Year: 2026-27")
    report.append("Students     : S. Venugopal (Roll No: 2520090185)")
    report.append("               Shashank Bangale (Roll No: 2520090086)")
    report.append("Guide        : I V Sai Lakshmi Haritha, Assistant Professor, Department of CSIT")
    report.append("================================================================================")
    
    return "\n".join(report)
