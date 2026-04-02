import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

ASSETS = {
    'S&P 500': '^GSPC',
    'NASDAQ': '^IXIC',
    'Bitcoin': 'BTC-USD',
    'Gold': 'GC=F',
    'Oil': 'CL=F',
    'Ethereum': 'ETH-USD',
    'Apple': 'AAPL',
    'Tesla': 'TSLA',
    'EuroStoxx': '^STOXX50E',
    'Nikkei': '^N225',
}

MOCK_DATES = pd.date_range(end=datetime.today(), periods=365)
MOCK_DATA = pd.DataFrame(
    np.cumsum(np.random.randn(len(MOCK_DATES), len(ASSETS)), axis=0) + 100,
    index=MOCK_DATES,
    columns=ASSETS.keys()
)

def fetch_data(start=None, end=None, use_mock=False):
    if use_mock:
        return MOCK_DATA.copy()
    try:
        start = start or (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
        end = end or datetime.today().strftime('%Y-%m-%d')
        df = yf.download(list(ASSETS.values()), start=start, end=end)['Adj Close']
        df.columns = [k for k in ASSETS.keys() if ASSETS[k] in df.columns or ASSETS[k] in df.columns.get_level_values(1)]
        if df.isnull().sum().sum() > 0:
            df = df.fillna(method='ffill').fillna(method='bfill')
        if df.shape[1] < len(ASSETS):
            # Fallback to mock if too many missing
            return MOCK_DATA.copy()
        return df
    except Exception:
        return MOCK_DATA.copy()
