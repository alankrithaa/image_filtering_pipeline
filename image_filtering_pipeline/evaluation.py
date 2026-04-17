"""
================================================================================
EVALUATION SCRIPT
================================================================================
Purpose: Compare pipeline predictions against your manual labels to calculate
         accuracy, precision, recall, and F1-score.

HOW TO USE:
===========
1. First, run the integrated pipeline on your labeled subset:
   - Put your 100 labeled images in a folder
   - Run the pipeline on that folder
   
2. Then run this script:
   - It compares pipeline output vs your labels.csv
   - Calculates metrics for overall and per-filter performance

METRICS EXPLAINED:
==================
- Accuracy: % of correct predictions (both accepts and rejects)
- Precision: When pipeline says KEEP, how often is it correct?
- Recall: Of all images that SHOULD be kept, how many did we keep?
- F1-Score: Harmonic mean of precision and recall

TARGET: >90% accuracy as per assignment requirements

================================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your manual labels CSV
# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to your manual labels CSV
LABELS_CSV_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\labeling_subset\labels_template.csv"

# Path to pipeline results JSON (from running pipeline on labeled subset)
PIPELINE_RESULTS_PATH = r"C:\Users\alank\OneDrive\Documents\image_filtering\image_filtering_pipeline\pipeline_output\pipeline_results.json"
# Alternative: If you ran pipeline on full dataset, specify the labeled folder
LABELED_FOLDER = r"labeling_subset"


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def load_labels(csv_path: str) -> pd.DataFrame:
    """
    Load manual labels from CSV file.
    
    Expected columns: filename, full_body, face_visible, is_adult, not_advertisement, keep
    Values should be 0 or 1.
    """
    print(f"📂 Loading manual labels from: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Ensure binary values
    binary_cols = ['full_body', 'face_visible', 'is_adult', 'not_advertisement', 'keep']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    print(f"   Loaded {len(df)} labels")
    
    # Show distribution
    if 'keep' in df.columns:
        keep_count = df['keep'].sum()
        print(f"   Manual labels: {keep_count} KEEP, {len(df) - keep_count} REJECT")
    
    return df


def load_predictions(json_path: str) -> pd.DataFrame:
    """
    Load pipeline predictions from JSON file.
    """
    print(f"📂 Loading pipeline predictions from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    
    # Convert boolean columns to int for comparison
    bool_cols = ['keep', 'full_body_pass', 'face_visible_pass', 'is_adult_pass', 'not_advertisement_pass']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)
    
    print(f"   Loaded {len(df)} predictions")
    
    if 'keep' in df.columns:
        keep_count = df['keep'].sum()
        print(f"   Pipeline predictions: {keep_count} KEEP, {len(df) - keep_count} REJECT")
    
    return df


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted labels (0 or 1)
        
    Returns:
        Dictionary with accuracy, precision, recall, f1
    """
    # True positives, false positives, true negatives, false negatives
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Metrics
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str = ""):
    """
    Print a nicely formatted confusion matrix.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    print(f"\n   {title} Confusion Matrix:")
    print("   " + "-"*35)
    print(f"   {'':15} | Pred KEEP | Pred REJECT")
    print("   " + "-"*35)
    print(f"   {'Actual KEEP':15} | {tp:9} | {fn:11}")
    print(f"   {'Actual REJECT':15} | {fp:9} | {tn:11}")
    print("   " + "-"*35)


def evaluate_pipeline(labels_df: pd.DataFrame, predictions_df: pd.DataFrame):
    """
    Compare pipeline predictions against manual labels.
    """
    print("\n" + "="*70)
    print("📊 EVALUATION RESULTS")
    print("="*70)
    
    # Merge on filename
    merged = labels_df.merge(
        predictions_df, 
        on='filename', 
        suffixes=('_true', '_pred'),
        how='inner'
    )
    
    print(f"\n   Matched {len(merged)} images between labels and predictions")
    
    if len(merged) == 0:
        print("   ❌ No matching images found! Check filenames.")
        return
    
    # =========================================================================
    # OVERALL KEEP/REJECT EVALUATION
    # =========================================================================
    print("\n" + "-"*70)
    print("📈 OVERALL PERFORMANCE (keep vs reject)")
    print("-"*70)
    
    y_true = merged['keep_true'].values
    y_pred = merged['keep_pred'].values
    
    metrics = calculate_metrics(y_true, y_pred)
    
    print(f"\n   Accuracy:  {metrics['accuracy']*100:.1f}%  {'✅ MEETS TARGET' if metrics['accuracy'] >= 0.9 else '❌ BELOW 90% TARGET'}")
    print(f"   Precision: {metrics['precision']*100:.1f}%  (When we say KEEP, we're right this often)")
    print(f"   Recall:    {metrics['recall']*100:.1f}%  (Of good images, we correctly kept this many)")
    print(f"   F1-Score:  {metrics['f1_score']*100:.1f}%")
    
    print_confusion_matrix(y_true, y_pred, "Overall")
    
    # =========================================================================
    # PER-FILTER EVALUATION
    # =========================================================================
    print("\n" + "-"*70)
    print("📈 PER-FILTER PERFORMANCE")
    print("-"*70)
    
    filter_mappings = [
        ('full_body', 'full_body_pass', 'Full Body'),
        ('face_visible', 'face_visible_pass', 'Face Visible'),
        ('is_adult', 'is_adult_pass', 'Adult (Age)'),
        ('not_advertisement', 'not_advertisement_pass', 'Not Advertisement'),
    ]
    
    for true_col, pred_col, name in filter_mappings:
        true_col_full = true_col + '_true' if true_col + '_true' in merged.columns else true_col
        pred_col_full = pred_col + '_pred' if pred_col + '_pred' in merged.columns else pred_col
        
        if true_col_full not in merged.columns or pred_col_full not in merged.columns:
            # Try without suffix
            if true_col in merged.columns and pred_col in merged.columns:
                true_col_full = true_col
                pred_col_full = pred_col
            else:
                print(f"\n   ⚠️  {name}: Columns not found, skipping")
                continue
        
        y_true_filter = merged[true_col_full].values
        y_pred_filter = merged[pred_col_full].values
        
        filter_metrics = calculate_metrics(y_true_filter, y_pred_filter)
        
        print(f"\n   {name}:")
        print(f"      Accuracy: {filter_metrics['accuracy']*100:.1f}%")
        print(f"      Precision: {filter_metrics['precision']*100:.1f}% | Recall: {filter_metrics['recall']*100:.1f}%")
    
    # =========================================================================
    # ERROR ANALYSIS
    # =========================================================================
    print("\n" + "-"*70)
    print("🔍 ERROR ANALYSIS")
    print("-"*70)
    
    # False positives (we said KEEP but should have REJECTED)
    false_positives = merged[(merged['keep_true'] == 0) & (merged['keep_pred'] == 1)]
    print(f"\n   False Positives ({len(false_positives)}): Images we KEPT but should have REJECTED")
    
    for _, row in false_positives.head(5).iterrows():
        print(f"      - {row['filename']}")
        reasons = []
        if row.get('full_body', row.get('full_body_true', 1)) == 0:
            reasons.append("not full body")
        if row.get('face_visible', row.get('face_visible_true', 1)) == 0:
            reasons.append("no face")
        if row.get('is_adult', row.get('is_adult_true', 1)) == 0:
            reasons.append("child")
        if row.get('not_advertisement', row.get('not_advertisement_true', 1)) == 0:
            reasons.append("advertisement")
        if reasons:
            print(f"        Actual issues: {', '.join(reasons)}")
    
    # False negatives (we said REJECT but should have KEPT)
    false_negatives = merged[(merged['keep_true'] == 1) & (merged['keep_pred'] == 0)]
    print(f"\n   False Negatives ({len(false_negatives)}): Images we REJECTED but should have KEPT")
    
    for _, row in false_negatives.head(5).iterrows():
        print(f"      - {row['filename']}")
        reason = row.get('rejection_reason', 'Unknown')
        print(f"        Pipeline reason: {reason}")
    
    # =========================================================================
    # RECOMMENDATIONS
    # =========================================================================
    print("\n" + "-"*70)
    print("💡 RECOMMENDATIONS")
    print("-"*70)
    
    if metrics['accuracy'] >= 0.9:
        print("\n   ✅ Your pipeline meets the >90% accuracy target!")
        print("   Consider running on the full dataset.")
    else:
        print(f"\n   ❌ Accuracy is {metrics['accuracy']*100:.1f}%, below the 90% target.")
        print("\n   Suggestions to improve:")
        
        # Analyze which filter is causing most issues
        if len(false_negatives) > len(false_positives):
            print("   - Pipeline is too strict (rejecting too many good images)")
            print("   - Try lowering thresholds:")
            print("     • KEYPOINT_CONFIDENCE_THRESHOLD (default 0.3)")
            print("     • MIN_KEYPOINTS_VISIBLE (default 13)")
            print("     • FACE_CONFIDENCE_THRESHOLD (default 0.5)")
        else:
            print("   - Pipeline is too lenient (accepting bad images)")
            print("   - Try raising thresholds or adding stricter checks")
    
    return metrics


def run_pipeline_on_labeled_subset(labeled_folder: str, labels_csv: str):
    """
    Helper to run the pipeline on the labeled subset and evaluate.
    """
    print("\n🔄 Running pipeline on labeled subset...")
    
    # Import and run pipeline
    from importlib import import_module
    
    # You would run the pipeline here, but for now we'll just provide instructions
    print("""
    To evaluate your pipeline:
    
    1. Copy your labeled images to a folder (already in 'labeling_subset/')
    
    2. Run the pipeline on that folder:
       - Open 06_integrated_pipeline.py
       - Change DATASET_PATH to your labeling_subset folder
       - Run the script
    
    3. Then run this evaluation script:
       python 07_evaluation.py
    """)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("📊 PIPELINE EVALUATION")
    print("="*70)
    
    # Check if files exist
    if not os.path.exists(LABELS_CSV_PATH):
        print(f"\n❌ Labels file not found: {LABELS_CSV_PATH}")
        print("\n   Make sure you:")
        print("   1. Completed labeling your 100 images")
        print("   2. Saved the CSV as 'labels.csv' (not 'labels_template.csv')")
        print("   3. Updated LABELS_CSV_PATH in this script")
        
        # Check for template
        template_path = LABELS_CSV_PATH.replace('labels.csv', 'labels_template.csv')
        if os.path.exists(template_path):
            print(f"\n   Found template at: {template_path}")
            print("   Rename it to 'labels.csv' after filling it out.")
        exit(1)
    
    if not os.path.exists(PIPELINE_RESULTS_PATH):
        print(f"\n❌ Pipeline results not found: {PIPELINE_RESULTS_PATH}")
        print("\n   Make sure you:")
        print("   1. Run the pipeline on your labeled subset first:")
        print("      python 06_integrated_pipeline.py")
        print("   2. Check that results are saved to pipeline_output/")
        print("   3. Update PIPELINE_RESULTS_PATH in this script")
        exit(1)
    
    # Load data
    labels_df = load_labels(LABELS_CSV_PATH)
    predictions_df = load_predictions(PIPELINE_RESULTS_PATH)
    
    # Evaluate
    metrics = evaluate_pipeline(labels_df, predictions_df)
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)
