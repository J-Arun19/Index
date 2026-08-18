import numpy as np
from sklearn.metrics import f1_score

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
