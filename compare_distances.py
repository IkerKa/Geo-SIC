import json
import numpy as np
import matplotlib.pyplot as plt

# File paths
file_paths = [
    './Experiments/PreTrainEv/5epochs/evaluation_5.json',
    './Experiments/PreTrainEv/50epochs/evaluation_50.json',
    './Experiments/PreTrainEv/100epochs/evaluation_100.json'
]

# Read JSON files and extract distance values
distances = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        distances.append(data['distance'])

# Plotting the distances as bars
labels = ['5 epochs', '50 epochs', '100 epochs']
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(x, distances, color=['blue', 'orange', 'green'])

# Add text labels on bars
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f}', ha='center', va='bottom', fontsize=10)

# Adjust y-axis limits to zoom in
y_min = min(distances) * 0.99 
y_max = max(distances) * 1.01 
ax.set_ylim(y_min, y_max)

ax.set_xlabel('Training Epochs')
ax.set_ylabel('Distance')
ax.set_title('Comparison of Distances for Different Training Epochs')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display the plot
plt.tight_layout()
plt.show()
