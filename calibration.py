import numpy as np


def calibration_curve(y_true, y_pred_prob, num_bins=15, thresh=0.5):
    pred_y = y_pred_prob > thresh
    correct = (pred_y == y_true).astype(np.float32)

    b = np.linspace(start=0, stop=1.0, num=num_bins)
    bins = np.digitize(y_pred_prob, bins=b, right=True)

    # expected calibration error, see "On Calibration of Modern Neural Networks."
    ece = 0

    conf = np.zeros(shape=(num_bins,))
    observed_rel_freq = np.zeros(shape=(num_bins,))
    for bin_idx in range(num_bins):
        mask = bins == bin_idx
        if np.any(mask):
            ece += np.abs(np.sum(correct[mask] - y_pred_prob[mask]))
        count = sum(mask)
        if count > 0:
            conf[bin_idx] = np.sum(y_pred_prob[mask]) / count
            observed_rel_freq[bin_idx] = np.sum(y_true[mask]) / count
        else:
            conf[bin_idx] = np.nan
            observed_rel_freq[bin_idx] = np.nan

    ece = ece / len(y_true)

    # Adaptive calibration error (ACE), see Dusenberry et al. (2020): Measuring Calibration in Deep Learning.
    # Here, bins are simply chosen such that each bin contains an equal number of probability outcomes.
    ace = 0
    b = np.quantile(y_pred_prob, b)
    b = np.unique(b)
    num_bins = len(b)
    bins = np.digitize(y_pred_prob, bins=b, right=True)
    for bin_idx in range(num_bins):
        mask = bins == bin_idx
        if np.any(mask):
            ace += np.abs(np.sum(correct[mask] - y_pred_prob[mask]))

    ace = ace / len(y_true)

    return conf, observed_rel_freq, ece, ace
