import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Data (from your table)
# ----------------------------
datasets = ["Beers", "Hospital", "Flights", "Walmart", "Shuttle", "Tax_20k", "Tax_200k"]

training = {
    "Las Training":   [149.33, 153.28, 155.96, 319.02, 622.89, 183.44, 214.91],
    "GIDCL Training": [231.11, 246.27, 171.16, 420.15, 745.92, 323.25, 496.13],
}

repair = {
    "Las Repair":      [6.4, 0.8, 10.78, 5.61, 100.45, 1.13, 2.63],
    "GIDCL Repair":    [20.25, 20.41, 18.87, 24.32, 95.87, 15.81, 22.58],
    "Jellyfish Repair":[20.12, 3.42, 28.89, 21.78, 39.42, 7.43, 19.25],
}

# ----------------------------
# Plot helper: grouped bar chart
# ----------------------------
def rgb255(r, g, b):
    return (r / 255., g / 255., b / 255.)

def plot_grouped_bars(ax, x_labels, series_dict, title, ylabel):
    n_groups = len(x_labels)
    n_series = len(series_dict)

    x = np.arange(n_groups)
    total_width = 0.82
    bar_w = total_width / n_series
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * bar_w

    style_map = {
        "Las Training":   dict(color=rgb255(228,147,67), edgecolor="black", hatch="///", linewidth=1.0, alpha=1.0),
        "GIDCL Training": dict(color=rgb255(145,219,210), edgecolor="black", hatch="x", linewidth=1.0, alpha=1.0),

        "Las Repair":       dict(color=rgb255(228,147,67), edgecolor="black", hatch="///", linewidth=1.0, alpha=1.0),
        "GIDCL Repair":     dict(color=rgb255(145,219,210), edgecolor="black", hatch="x", linewidth=1.0, alpha=1.0),
        "Jellyfish Repair": dict(color=rgb255(235,202,89), edgecolor="black", hatch=".",  linewidth=1.0, alpha=1.0),
    }

    for i, (name, vals) in enumerate(series_dict.items()):
        style = style_map.get(name, {})  # 找不到就用默认样式
        ax.bar(
            x + offsets[i],
            vals,
            width=bar_w,
            label=name,
            **style,              # 2) 把自定义样式注入到 bar 里
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
    ax.margins(x=0.02)

# ----------------------------
# Figure 1: Training comparison
# ----------------------------
fig1, ax1 = plt.subplots(figsize=(10.5, 4.2))
plot_grouped_bars(
    ax1,
    datasets,
    training,
    title="Training Time Comparison",
    ylabel="Time"
)
plt.tight_layout()
plt.savefig("training_comparison.png", dpi=300)
plt.show()

# ----------------------------
# Figure 2: Repair comparison
# ----------------------------
fig2, ax2 = plt.subplots(figsize=(10.5, 4.2))
plot_grouped_bars(
    ax2,
    datasets,
    repair,
    title="Repair Time Comparison",
    ylabel="Time"
)
plt.tight_layout()
plt.savefig("repair_comparison.png", dpi=300)
print(1)
plt.show()
