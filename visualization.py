import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec
from utils import neon_line

plt.style.use('dark_background')

NEON_COLORS = ['#00FFFF', '#9B5DE5', '#00FF9C', '#FF4D4D', '#FFD700', '#FF00C8', '#00FFEA', '#FF4D4D']

class Dashboard:
    def __init__(self, prices, returns, rolling_corr, volatility, pca_factors, pca_explained, pulse_scores, divergence_flags):
        self.prices = prices
        self.returns = returns
        self.rolling_corr = rolling_corr
        self.volatility = volatility
        self.pca_factors = pca_factors
        self.pca_explained = pca_explained
        self.pulse_scores = pulse_scores
        self.divergence_flags = divergence_flags
        self.assets = prices.columns.tolist()
        self.n_assets = len(self.assets)
        self.fig = plt.figure(figsize=(16, 10))
        self.gs = gridspec.GridSpec(3, 3, height_ratios=[2,1,1])
        self.ax_main = self.fig.add_subplot(self.gs[0, :2])
        self.ax_corr = self.fig.add_subplot(self.gs[0, 2])
        self.ax_pulse = self.fig.add_subplot(self.gs[1, :2])
        self.ax_pca = self.fig.add_subplot(self.gs[1, 2])
        self.ax_div = self.fig.add_subplot(self.gs[2, :])
        self.lines = []
        self.ani = None
        self._init_plot()

    def _init_plot(self):
        # Main panel: normalized prices
        norm_prices = (self.prices - self.prices.iloc[0]) / self.prices.iloc[0]
        for i, asset in enumerate(self.assets):
            l = neon_line(self.ax_main, norm_prices.index, norm_prices[asset], color=NEON_COLORS[i % len(NEON_COLORS)], label=asset)
            self.lines.append(l)
        self.ax_main.set_title('Normalized Asset Prices', fontsize=14)
        self.ax_main.legend(loc='upper left', fontsize=9)
        self.ax_main.grid(False)
        # Correlation heatmap
        self.corr_img = self.ax_corr.imshow(np.zeros((self.n_assets, self.n_assets)), vmin=-1, vmax=1, cmap='cool')
        self.ax_corr.set_title('Rolling Correlation', fontsize=12)
        self.ax_corr.set_xticks(range(self.n_assets))
        self.ax_corr.set_yticks(range(self.n_assets))
        self.ax_corr.set_xticklabels(self.assets, rotation=90, fontsize=7)
        self.ax_corr.set_yticklabels(self.assets, fontsize=7)
        # Pulse indicator
        self.pulse_line = neon_line(self.ax_pulse, [], [], color='#00FF9C', label='Pulse Score')
        self.ax_pulse.set_title('Market Pulse Score', fontsize=12)
        self.ax_pulse.set_ylim(0, 1)
        self.ax_pulse.grid(False)
        # PCA factor
        self.pca_line = neon_line(self.ax_pca, [], [], color='#9B5DE5', label='1st PC')
        self.ax_pca.set_title('First Principal Component', fontsize=12)
        self.ax_pca.grid(False)
        # Divergence detector
        self.div_line = neon_line(self.ax_div, [], [], color='#FF4D4D', label='Divergence')
        self.ax_div.set_title('Divergence Detector', fontsize=12)
        self.ax_div.set_ylim(-0.1, 1.1)
        self.ax_div.grid(False)

    def animate(self, i):
        # Animate all panels
        window = min(i+1, len(self.prices))
        norm_prices = (self.prices.iloc[:window] - self.prices.iloc[0]) / self.prices.iloc[0]
        for j, l in enumerate(self.lines):
            l[0].set_data(norm_prices.index, norm_prices.iloc[:, j])
        # Correlation heatmap
        if window > 30:
            # Get the rolling correlation matrix for the current window as a square matrix
            corr_series = self.rolling_corr.iloc[window-1]
            if hasattr(corr_series, 'unstack'):
                corr_matrix = corr_series.unstack().reindex(index=self.assets, columns=self.assets).values
            else:
                corr_matrix = np.zeros((self.n_assets, self.n_assets))
            self.corr_img.set_data(corr_matrix)
        # Pulse
        self.pulse_line[0].set_data(range(window), self.pulse_scores[:window])
        # Color zones
        if window > 10:
            last = self.pulse_scores[window-1]
            if last > 0.7:
                self.ax_pulse.set_facecolor('#003300')
            elif last > 0.4:
                self.ax_pulse.set_facecolor('#333300')
            else:
                self.ax_pulse.set_facecolor('#330000')
        # PCA
        self.pca_line[0].set_data(range(window), self.pca_factors[:window, 0])
        # Divergence
        self.div_line[0].set_data(range(window), self.divergence_flags[:window])
        return self.lines + [self.corr_img, self.pulse_line[0], self.pca_line[0], self.div_line[0]]

    def run(self):
        self.ani = FuncAnimation(self.fig, self.animate, frames=len(self.prices), interval=60, blit=False, repeat=False)
        plt.tight_layout()
        plt.show()
