import matplotlib.pyplot as plt
import numpy as np
import os

# Directory setup
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'exp_10.txt')

# Robust text reading
def read_text(path):
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            with open(path, 'r', encoding=enc) as fh:
                text = fh.read()
            if text.strip():
                return text
        except Exception:
            continue
    with open(path, 'rb') as fh:
        return fh.read().decode('utf-8', errors='ignore')

content = read_text(data_path)
lines = [line for line in content.splitlines() if line.strip()]
if len(lines) < 2:
    raise ValueError("exp_10.txt does not contain enough data rows.")

# Parse header for dataset names
header_parts = [part for part in lines[0].split('\t') if part]
dataset_names = header_parts[1:]
if not dataset_names:
    raise ValueError("Failed to parse dataset names from header.")

# Parse ablation rows
variant_names = []
values_rows = []
for line in lines[1:]:
    parts = [part for part in line.split('\t') if part]
    if len(parts) < 2:
        continue
    variant_names.append(parts[0])
    try:
        values = [float(x) for x in parts[1:1 + len(dataset_names)]]
    except ValueError:
        raise ValueError(f"Invalid numeric value encountered in line: {line}")
    values_rows.append(values)

if not variant_names:
    raise ValueError("No ablation rows parsed from exp_10.txt.")

values = np.array(values_rows)
num_variants, num_datasets = values.shape

# Matplotlib style setup
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.major.width'] = 0.8
plt.rcParams['ytick.major.width'] = 0.8
plt.rcParams['lines.linewidth'] = 1.5

# Color mapping based on variant names (matching the image)
color_map = {
    'w/o LLM': '#808080',      # Grey
    'w/o EM': '#87CEEB',       # Light blue (Sky blue)
    'w/o CL': '#FFD700',       # Yellow (Gold)
    'full version': '#FFA500', # Orange
}

bar_width = min(0.18, 0.8 / num_variants)
x = np.arange(num_datasets)

fig, ax = plt.subplots(figsize=(6.6, 4.6))

for idx, (variant, row_values) in enumerate(zip(variant_names, values)):
    positions = x + (idx - (num_variants - 1) / 2) * bar_width
    # Get color from mapping, fallback to grey if not found
    color = color_map.get(variant, '#808080')
    ax.bar(
        positions,
        row_values,
        width=bar_width,
        label=variant,
        color=color,
        edgecolor='black',
        linewidth=0.6
    )

ax.set_xlabel('Dataset', fontsize=11, fontweight='bold')
ax.set_ylabel('F1-Score', fontsize=11, fontweight='bold')
ax.set_title('Ablation Study on EMCL', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(dataset_names, rotation=20, ha='center')
ax.set_ylim(0, 1.05)
yticks = np.arange(0, 1.051, 0.1)
yticks = [tick for tick in yticks if round(tick, 10) < 1.05]
ax.set_yticks(yticks)
ax.grid(axis='y', linestyle='--', alpha=0.3, linewidth=0.5)

ax.legend(
    loc='upper right',
    frameon=True,
    fancybox=True,
    shadow=False,
    fontsize=9,
    framealpha=0.9,
    borderpad=0.6
)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()
output_path = os.path.join(script_dir, 'exp_10.pdf')
fig.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
plt.close(fig)

print(f"Bar chart saved to {output_path}")

