import matplotlib.pyplot as plt
import numpy as np

def neon_line(ax, x, y, color, label=None):
    # Glow effect: thick transparent line + main line
    l1, = ax.plot(x, y, color=color, linewidth=5, alpha=0.2, solid_capstyle='round')
    l2, = ax.plot(x, y, color=color, linewidth=2.2, label=label, solid_capstyle='round')
    return (l2, l1)
