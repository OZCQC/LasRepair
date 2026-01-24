import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (UPDATED)
# ---------------------------
datasets = ["CAN", "WebTable", "UK_SG", "USA"]
methods = ["JOSIE", "LSH", "DeepJoin", "Snoopy", "Omnimatch", "HyperJoin"]

data = {
    "WebTable": {
        "JOSIE":     [0.353, 0.251, 0.159, 0.109, 0.232, 0.242],
        "LSH":       [0.587, 0.514, 0.467, 0.174, 0.403, 0.525],
        "DeepJoin":  [0.693, 0.558, 0.459, 0.221, 0.510, 0.681],
        "Snoopy":    [0.713, 0.553, 0.463, 0.228, 0.500, 0.683],
        "Omnimatch": [0.700, 0.591, 0.453, 0.218, 0.556, 0.687],
        "HyperJoin": [0.813, 0.747, 0.536, 0.259, 0.698, 0.807],
    },
    "USA": {
        "JOSIE":     [0.760, 0.747, 0.686, 0.177, 0.522, 0.799],
        "LSH":       [0.810, 0.721, 0.720, 0.185, 0.418, 0.505],
        "DeepJoin":  [0.870, 0.687, 0.488, 0.202, 0.481, 0.571],
        "Snoopy":    [0.860, 0.747, 0.638, 0.199, 0.519, 0.740],
        "Omnimatch": [0.630, 0.747, 0.704, 0.146, 0.522, 0.820],
        "HyperJoin": [0.920, 0.953, 0.836, 0.213, 0.664, 0.971],
    },
    "CAN": {
        "JOSIE":     [0.567, 0.524, 0.441, 0.131, 0.358, 0.495],
        "LSH":       [0.687, 0.657, 0.607, 0.136, 0.357, 0.464],
        "DeepJoin":  [0.640, 0.533, 0.424, 0.147, 0.368, 0.476],
        "Snoopy":    [0.793, 0.651, 0.587, 0.180, 0.450, 0.667],
        "Omnimatch": [0.607, 0.624, 0.565, 0.133, 0.410, 0.614],
        "HyperJoin": [0.833, 0.818, 0.659, 0.189, 0.559, 0.742],
    },
    "UK_SG": {
        "JOSIE":     [0.270, 0.287, 0.232, 0.088, 0.270, 0.349],
        "LSH":       [0.260, 0.263, 0.264, 0.060, 0.144, 0.222],
        "DeepJoin":  [0.660, 0.550, 0.422, 0.196, 0.485, 0.602],
        "Snoopy":    [0.720, 0.603, 0.440, 0.208, 0.535, 0.640],
        "Omnimatch": [0.560, 0.483, 0.392, 0.165, 0.418, 0.553],
        "HyperJoin": [0.940, 0.890, 0.702, 0.272, 0.771, 0.988],
    }
}



# ---------------------------
# Visual styles (UPDATED for single HyperJoin)
# ---------------------------
palette = {
    "JOSIE":     "#1f77b4",
    "LSH":       "#ff7f0e",
    "DeepJoin":  "#2ca02c",
    "Snoopy":    "#d62728",
    "Omnimatch": "#9467bd",
    "HyperJoin": "#8c564b",
}

linestyles = {
    "JOSIE":     "-.",
    "LSH":       "--",
    "DeepJoin":  "--",
    "Snoopy":    "-.",
    "Omnimatch": ":",
    "HyperJoin": "-",
}

markers = {
    "JOSIE":     "o",
    "LSH":       "s",
    "DeepJoin":  "^",
    "Snoopy":    "D",
    "Omnimatch": "v",
    "HyperJoin": "p",
}

ks = np.array([5, 15, 25])

yticks = np.array([0.0, 0.25, 0.5, 0.75, 1.0])


# Matplotlib style settings
plt.rcParams.update({
    # --- Font: Times New Roman ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",   # 数学符号更像 Times
    "pdf.fonttype": 42,           # 让 PDF 里字体以 TrueType 方式嵌入
    "ps.fonttype": 42,

    # --- your existing settings ---
    "font.size": 18,
    "axes.titlesize": 18,
    "axes.labelsize": 18,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
    "axes.linewidth": 1.2,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "grid.linewidth": 0.5
})


# Create figure with better proportions (wider, shorter subplots)
fig = plt.figure(figsize=(20, 7))
gs = fig.add_gridspec(2, 4, wspace=0.35, hspace=0.50, 
                      left=0.05, right=0.98, top=0.88, bottom=0.12)

axes = []
for r in range(2):
    for c in range(4):
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

# Plotting
plot_idx = 0
subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
for i, dataset in enumerate(datasets):
    # P@k subplot
    ax = axes[plot_idx]
    for m in methods:
        y = data[dataset][m][:3]  # P@5, P@15, P@25
        ax.plot(ks, y, 
                label=m,
                color=palette[m],
                linestyle=linestyles[m],
                linewidth=2.2,
                marker=markers[m],
                markersize=7,
                markerfacecolor=palette[m],
                markeredgecolor=palette[m],
                markeredgewidth=1.5,
                alpha=0.9)
    
    # Remove title from top, will add at bottom
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel("P@k", fontsize=18)
    ax.set_xticks(ks)
    ax.set_xlim(3, 27)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])

    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    # Keep all spines visible (add frame)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('#333333')
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=1)
    
    # Add title below the plot
    ax.text(0.5, -0.30, f"{subplot_labels[plot_idx]} P@k on {dataset}", 
            transform=ax.transAxes, ha='center', va='top', 
            fontsize=18)
    plot_idx += 1

    # R@k subplot
    ax = axes[plot_idx]
    for m in methods:
        y = data[dataset][m][3:]  # R@5, R@15, R@25
        ax.plot(ks, y,
                label=m,
                color=palette[m],
                linestyle=linestyles[m],
                linewidth=2.2,
                marker=markers[m],
                markersize=7,
                markerfacecolor=palette[m],
                markeredgecolor=palette[m],
                markeredgewidth=1.5,
                alpha=0.9)
    
    # Remove title from top, will add at bottom
    ax.set_xlabel("k", fontsize=18)
    ax.set_ylabel("R@k", fontsize=18)
    ax.set_xticks(ks)
    ax.set_xlim(3, 27)
    ax.set_ylim(-0.02, 1.05)
    ax.set_yticks(yticks)
    ax.set_yticklabels(["0.00", "0.25", "0.50", "0.75", "1.00"])

    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
    # Keep all spines visible (add frame)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_edgecolor('#333333')
    ax.tick_params(axis='both', which='major', direction='out', length=4, width=1)
    
    # Add title below the plot
    ax.text(0.5, -0.30, f"{subplot_labels[plot_idx]} R@k on {dataset}", 
            transform=ax.transAxes, ha='center', va='top', 
            fontsize=18)
    plot_idx += 1

# Create unified legend at the top
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, 
                   loc='upper center', 
                   ncol=6,
                   frameon=True, 
                   fancybox=False,
                   shadow=False,
                   edgecolor='#CCCCCC',
                   framealpha=1.0,
                   bbox_to_anchor=(0.5, 0.97),
                   columnspacing=1.5,
                   handlelength=2.5,
                   handletextpad=0.5)

# Add figure caption
# fig.text(0.5, 0.01,
#          "Figure: Effectiveness on datasets (P@k / R@k).",
#          ha='center', fontsize=14, fontweight='bold')

# Save outputs
plt.savefig("main_results.png", dpi=300, bbox_inches="tight", facecolor='white')
plt.savefig("main_results.pdf", bbox_inches="tight", facecolor='white')

plt.show()