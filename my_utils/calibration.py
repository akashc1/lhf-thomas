from sklearn.calibration import calibration_curve


def compute_calibration(y, p_mean, num_bins, num_classes):
    cals = {}
    for c in range(num_classes):
        y_true = (y==c).cpu().numpy()
        y_prob = p_mean[:, c].cpu().numpy()
        reliability_diag = calibration_curve(y_true, y_prob, n_bins=num_bins)
        cals[c] = {'reliability_diag': reliability_diag}
    return cals