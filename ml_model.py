import numpy as np
from sklearn.cluster import KMeans

def cluster_assets(returns, n_clusters=3):
    # Use last window for clustering
    if len(returns) < 10:
        return np.zeros(returns.shape[1])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(returns[-30:].T)
    return labels

def compute_market_pulse(rolling_corr, volatility, pca_explained):
    # Pulse: high corr + low vol + high explained variance = stable
    avg_corr = np.nanmean(rolling_corr.values[-1]) if len(rolling_corr) else 0
    avg_vol = np.nanmean(volatility.values[-1]) if len(volatility) else 0
    pca_var = pca_explained[0] if len(pca_explained) else 0
    # Normalize to [0,1]
    pulse = 0.5 * avg_corr + 0.3 * (1 - avg_vol) + 0.2 * pca_var
    return pulse

def detect_divergence(rolling_corr, threshold=0.3):
    # Returns True if mean correlation drops below threshold
    if len(rolling_corr) == 0:
        return False
    mean_corr = np.nanmean(rolling_corr.values[-1])
    return mean_corr < threshold
