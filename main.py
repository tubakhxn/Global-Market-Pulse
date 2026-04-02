import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from data_loader import fetch_data
plt.style.use('dark_background')

# Load data
data = fetch_data()
assets = data.columns.tolist()
norm_prices = (data - data.iloc[0]) / data.iloc[0]

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#00FFFF', '#9B5DE5', '#00FF9C', '#FF4D4D', '#FFD700', '#FF00C8', '#00FFEA', '#FF4D4D']
lines = []
for i, asset in enumerate(assets):
    # Thin neon lines
    l, = ax.plot([], [], color=colors[i % len(colors)], linewidth=1.5, label=asset)
    lines.append(l)
ax.set_title('Normalized Asset Prices', fontsize=28)
ax.legend(loc='upper right', fontsize=9, bbox_to_anchor=(1.0, 1.0), frameon=False)
ax.set_xlim(norm_prices.index[0], norm_prices.index[-1])
ax.set_ylim(norm_prices.min().min() - 0.05, norm_prices.max().max() + 0.05)
ax.grid(False)

def animate(i):
    for j, l in enumerate(lines):
        l.set_data(norm_prices.index[:i+1], norm_prices.iloc[:i+1, j])
    return lines

ani = FuncAnimation(fig, animate, frames=len(norm_prices), interval=30, blit=True, repeat=False)
plt.tight_layout()
plt.show()
