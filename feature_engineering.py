import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def compute_returns(prices):
    return prices.pct_change().dropna()

def compute_rolling_corr(returns, window=30):
    # Compute rolling correlation and ensure MultiIndex for unstacking
    corr = returns.rolling(window).corr()
    # Add date as first level of index for unstacking in visualization
    corr = corr.reset_index()
    corr = corr.rename(columns={'level_0': 'date', 'level_1': 'asset1'})
    corr = corr.set_index(['date', 'asset1'])
    # Now, columns are asset2
    return corr

def compute_volatility(returns, window=30):
    return returns.rolling(window).std().dropna()

def compute_pca(returns, n_components=2):
    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(returns.fillna(0))
    explained = pca.explained_variance_ratio_
    return factors, explained, pca

def zscore(series):
    return (series - series.mean()) / series.std()
