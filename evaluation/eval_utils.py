"""
Common utilities for VAD evaluation protocols (SWI and SPI).
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_curve, accuracy_score
)
from typing import Dict, Tuple


def overlapping_segments(
    audio: tf.Tensor,
    percent_overlap: float = 0.875,
    segment_duration: float = 0.63,
    sr: int = 16000
) -> tf.Tensor:
    """
    Create overlapping segments from audio for frame-level predictions.
    
    This ensures full coverage of the audio, including short clips.
    
    Args:
        audio: Input audio tensor
        percent_overlap: Overlap percentage (0.875 = 87.5%)
        segment_duration: Segment length in seconds
        sr: Sample rate
        
    Returns:
        Tensor of overlapping audio segments
    """
    frame_length = int(segment_duration * sr)
    frame_step = int(segment_duration * (1 - percent_overlap) * sr)
    
    # Create overlapping frames
    segments = tf.signal.frame(
        audio,
        frame_length=frame_length,
        frame_step=frame_step,
        pad_end=False
    )
    
    # Add last segment to ensure full coverage
    last_segment = tf.signal.frame(
        audio[-frame_length:],
        frame_length=frame_length,
        frame_step=frame_length,
        pad_end=False
    )
    
    return tf.concat([segments, last_segment], axis=0)


def tpr_at_fpr(y_true, y_score, target_fpr: float = 0.315) -> float:
    """
    Calculate True Positive Rate at a specific False Positive Rate.
    
    This metric is commonly used in speech detection evaluation.
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted scores
        target_fpr: Target false positive rate (default: 0.315)
        
    Returns:
        TPR value at the target FPR
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    # Sort to ensure monotonic increase
    idx = np.argsort(fpr)
    
    # Interpolate TPR at target FPR
    return float(np.interp(target_fpr, fpr[idx], tpr[idx]))


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    aggregation_method: str = 'median'
) -> Dict:
    """
    Calculate comprehensive binary classification metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        aggregation_method: 'median' or 'mean'
        
    Returns:
        Dictionary of metrics
    """
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_pred)
    
    # Precision, Recall, F-scores
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # F2 score (emphasizes recall over precision)
    _, _, f2, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', beta=2, zero_division=0
    )
    
    # TPR at FPR=0.315
    tpr_0315 = tpr_at_fpr(y_true, y_pred)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Fix support None issue (happens when no positive samples exist)
    if support is None:
        support = 0

    try:
        support = int(support)
    except Exception:
        support = 0
    
    return {
        f'Accuracy_{aggregation_method}': float(accuracy),
        f'AUROC_{aggregation_method}': float(auroc),
        f'F1_{aggregation_method}': float(f1),
        f'F2_{aggregation_method}': float(f2),
        f'Precision_{aggregation_method}': float(precision),
        f'Recall_{aggregation_method}': float(recall),
        f'TPR@FPR=0.315_{aggregation_method}': float(tpr_0315),
        f'Support_{aggregation_method}': int(support),
        f'Confusion_Matrix_{aggregation_method}': conf_matrix.tolist()
    }


def calculate_per_class_metrics(
    results_df,
    exclude_class: str = 'NO_SPEECH'
) -> Dict:
    """
    Calculate metrics for each class separately (multi-class to binary).
    
    For each speech class, compares it against non-speech.
    
    Args:
        results_df: DataFrame with 'label', 'predicted_median', 'predicted_mean'
        exclude_class: Class to treat as negative (usually 'NO_SPEECH')
        
    Returns:
        Dictionary of per-class metrics
    """
    classes = list(results_df['label'].unique())
    
    if exclude_class in classes:
        classes.remove(exclude_class)
    
    metrics = {}
    
    for cls in classes:
        # Binary problem: current class vs. no speech
        cls_df = results_df[results_df['label'].isin([cls, exclude_class])]
        
        y_true = (cls_df['label'] == cls).astype(int).values
        y_pred_median = cls_df['predicted_median'].values
        y_pred_mean = cls_df['predicted_mean'].values
        
        # Calculate metrics for both aggregation methods
        metrics[cls] = {
            'Median': calculate_binary_metrics(y_true, y_pred_median, 'median'),
            'Mean': calculate_binary_metrics(y_true, y_pred_mean, 'mean')
        }
    
    return metrics


def calculate_overall_metrics(results_df) -> Dict:
    """
    Calculate overall binary VAD metrics (speech vs. non-speech).
    
    Args:
        results_df: DataFrame with 'ground_truth', 'predicted_median', 'predicted_mean'
        
    Returns:
        Dictionary of overall metrics for median and mean aggregation
    """
    y_true = results_df['ground_truth'].values
    y_pred_median = results_df['predicted_median'].values
    y_pred_mean = results_df['predicted_mean'].values
    
    return {
        'Median': calculate_binary_metrics(y_true, y_pred_median, 'median'),
        'Mean': calculate_binary_metrics(y_true, y_pred_mean, 'mean')
    }


def print_metrics(metrics: Dict, title: str = "Metrics"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics section
    """
    print("\n" + "=" * 60)
    print(f"{title}")
    print("=" * 60)
    
    import pprint
    pprint.pprint(metrics, width=80, compact=False)
    print("=" * 60 + "\n")


def aggregate_fold_results(all_results: list, metric_names: list) -> Dict:
    """
    Aggregate results across multiple folds.
    
    Args:
        all_results: List of result dictionaries from each fold
        metric_names: List of metric names to aggregate
        
    Returns:
        Dictionary with mean and std for each metric
    """
    aggregated = {}
    
    for metric_name in metric_names:
        values = [result[metric_name] for result in all_results if metric_name in result]
        
        if values:
            aggregated[f'{metric_name}_mean'] = float(np.mean(values))
            aggregated[f'{metric_name}_std'] = float(np.std(values))
    
    return aggregated