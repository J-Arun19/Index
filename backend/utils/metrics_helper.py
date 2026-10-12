import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    precision_recall_curve, auc, balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
)

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Computes the Expected Calibration Error (ECE) for binary classification.
    
    Args:
        y_true (list or np.ndarray): Ground truth binary labels (0 or 1).
        y_prob (list or np.ndarray): Predicted probability of the positive class (class 1).
        n_bins (int): Number of bins to partition the probability space.
        
    Returns:
        float: Expected Calibration Error.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    if total_samples == 0:
        return 0.0
        
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Mask for samples falling in the current bin
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
            
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            # Predicted class is based on the standard 0.5 threshold for calibration evaluation
            preds = (y_prob[in_bin] >= 0.5).astype(int)
            acc = np.mean(preds == y_true[in_bin])
            # Confidence is max(p, 1-p) for binary classification
            conf = np.mean(np.maximum(y_prob[in_bin], 1.0 - y_prob[in_bin]))
            
            ece += (bin_size / total_samples) * np.abs(acc - conf)
            
    return float(ece)

def optimize_threshold(y_true, y_prob):
    """
    Finds the classification threshold that maximizes Macro-F1.
    
    Args:
        y_true (list or np.ndarray): Ground truth binary labels (0 or 1).
        y_prob (list or np.ndarray): Predicted probability of the positive class (class 1).
        
    Returns:
        tuple: (best_threshold, best_macro_f1)
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    best_threshold = 0.5
    best_macro_f1 = 0.0
    
    # Grid search over thresholds from 0.01 to 0.99
    thresholds = np.linspace(0.01, 0.99, 99)
    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        macro_f1 = f1_score(y_true, preds, average='macro', zero_division=0)
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = t
            
    return float(best_threshold), float(best_macro_f1)

def compute_all_metrics(y_true, y_prob, threshold=0.5):
    """
    Computes a comprehensive suite of metrics for binary classification.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    
    total = len(y_true)
    if total == 0:
        return {}
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    real_pred_pct = float(np.mean(y_pred == 0))
    fake_pred_pct = float(np.mean(y_pred == 1))
    
    acc = float(np.mean(y_pred == y_true)) * 100
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    real_prec, real_rec, real_f1 = float(prec[0]), float(rec[0]), float(f1[0])
    fake_prec, fake_rec, fake_f1 = float(prec[1]), float(rec[1]), float(f1[1])
    
    macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        roc_auc = 0.0
        
    try:
        p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = float(auc(r_curve, p_curve))
    except Exception:
        pr_auc = 0.0
        
    ece = compute_ece(y_true, y_prob)
    
    class_collapse = bool(real_pred_pct >= 0.98 or fake_pred_pct >= 0.98 or real_rec < 0.05 or fake_rec < 0.05)
    
    res = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision": fake_prec,
        "recall": fake_rec,
        "f1_score": fake_f1,
        "macro_f1": macro_f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "ece": ece,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "false_positive_real_count": int(fp),
        "false_negative_fake_count": int(fn),
        "real_precision": real_prec,
        "real_recall": real_rec,
        "real_f1": real_f1,
        "fake_precision": fake_prec,
        "fake_recall": fake_rec,
        "fake_f1": fake_f1,
        "real_pred_pct": real_pred_pct,
        "fake_pred_pct": fake_pred_pct,
        "class_collapse": class_collapse
    }
    res["matrix_verification"] = verify_confusion_matrix_consistency(res["confusion_matrix"], res)
    return res


def verify_confusion_matrix_consistency(
    confusion_matrix_val,
    metrics_dict: dict,
    tolerance: float = 1e-3
) -> dict:
    """
    Verifies that the provided confusion matrix mathematically matches the reported
    classification metrics within a small numerical tolerance.

    Args:
        confusion_matrix_val (list or np.ndarray): 2x2 confusion matrix [[tn, fp], [fn, tp]].
        metrics_dict (dict): Dictionary containing reported metrics.
        tolerance (float): Maximum acceptable absolute difference.

    Returns:
        dict: Consistency audit result with keys:
            - is_consistent (bool)
            - derived_metrics (dict)
            - discrepancies (dict)
            - error_messages (list of str)
    """
    cm = np.array(confusion_matrix_val)
    if cm.shape != (2, 2):
        return {
            "is_consistent": False,
            "error_messages": [f"Expected 2x2 confusion matrix, got shape {cm.shape}"],
            "discrepancies": {},
            "derived_metrics": {}
        }

    tn = float(cm[0, 0])
    fp = float(cm[0, 1])
    fn = float(cm[1, 0])
    tp = float(cm[1, 1])

    total_samples = tn + fp + fn + tp
    if total_samples <= 0:
        return {
            "is_consistent": False,
            "error_messages": ["Confusion matrix has zero total samples."],
            "discrepancies": {},
            "derived_metrics": {}
        }

    real_total = tn + fp
    fake_total = fn + tp
    pred_real_total = tn + fn
    pred_fake_total = fp + tp

    derived_real_rec = tn / real_total if real_total > 0 else 0.0
    derived_real_prec = tn / pred_real_total if pred_real_total > 0 else 0.0
    derived_real_f1 = (
        2.0 * (derived_real_prec * derived_real_rec) / (derived_real_prec + derived_real_rec)
        if (derived_real_prec + derived_real_rec) > 0 else 0.0
    )

    derived_fake_rec = tp / fake_total if fake_total > 0 else 0.0
    derived_fake_prec = tp / pred_fake_total if pred_fake_total > 0 else 0.0
    derived_fake_f1 = (
        2.0 * (derived_fake_prec * derived_fake_rec) / (derived_fake_prec + derived_fake_rec)
        if (derived_fake_prec + derived_fake_rec) > 0 else 0.0
    )

    derived_macro_f1 = (derived_real_f1 + derived_fake_f1) / 2.0
    derived_acc_pct = (tn + tp) / total_samples * 100.0

    derived_metrics = {
        "total_samples": int(total_samples),
        "real_count": int(real_total),
        "fake_count": int(fake_total),
        "accuracy": derived_acc_pct,
        "real_precision": derived_real_prec,
        "real_recall": derived_real_rec,
        "real_f1": derived_real_f1,
        "fake_precision": derived_fake_prec,
        "fake_recall": derived_fake_rec,
        "fake_f1": derived_fake_f1,
        "macro_f1": derived_macro_f1,
    }

    discrepancies = {}
    error_messages = []

    checks = [
        ("real_precision", derived_real_prec),
        ("real_recall", derived_real_rec),
        ("real_f1", derived_real_f1),
        ("fake_precision", derived_fake_prec),
        ("fake_recall", derived_fake_rec),
        ("fake_f1", derived_fake_f1),
        ("macro_f1", derived_macro_f1),
    ]

    for key, derived_val in checks:
        if key in metrics_dict and metrics_dict[key] is not None:
            rep_val = float(metrics_dict[key])
            diff = abs(derived_val - rep_val)
            discrepancies[key] = diff
            if diff > tolerance:
                error_messages.append(
                    f"Discrepancy in '{key}': matrix derives {derived_val:.4f}, but reported {rep_val:.4f} (diff={diff:.4f} > {tolerance})"
                )

    if "accuracy" in metrics_dict and metrics_dict["accuracy"] is not None:
        rep_acc = float(metrics_dict["accuracy"])
        if rep_acc <= 1.0 and total_samples > 1:
            rep_acc *= 100.0
        acc_diff = abs(derived_acc_pct - rep_acc)
        discrepancies["accuracy"] = acc_diff
        if acc_diff > (tolerance * 100.0):
            error_messages.append(
                f"Discrepancy in 'accuracy': matrix derives {derived_acc_pct:.2f}%, but reported {rep_acc:.2f}% (diff={acc_diff:.2f})"
            )

    is_consistent = (len(error_messages) == 0)

    return {
        "is_consistent": is_consistent,
        "derived_metrics": derived_metrics,
        "discrepancies": discrepancies,
        "error_messages": error_messages
    }


