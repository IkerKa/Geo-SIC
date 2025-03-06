import json
import numpy as np
import matplotlib.pyplot as plt

# File paths
file_paths = [
    './Experiments/PreTrainEv/5epochs/evaluation_5.json',
    './Experiments/PreTrainEv/50epochs/evaluation_50.json',
    './Experiments/PreTrainEv/100epochs/evaluation_100.json',
    './Experiments/PreTrainEv/250epochs/evaluation_250.json'
]

distances = []
momentum_means = []
momentum_maxs = []
momentum_mins = []
ssim_values = []
labels = ['5 epochs', '50 epochs', '100 epochs', '250 epochs']

for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        distances.append(data['distance'])
        momentum_means.append(data['momentum_mean'])
        momentum_maxs.append(data['momentum_max'])
        momentum_mins.append(data['momentum_min'])
        ssim_values.append(data['ssim'])

x = np.arange(len(labels))

fig, axes = plt.subplots(2, 2, figsize=(12, 12))


bars = axes[0, 0].bar(x, distances, color=['blue', 'orange', 'green', 'red'])

for bar in bars:
    yval = bar.get_height()
    axes[0,0].text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.6f}', ha='center', va='bottom', fontsize=10)

y_min = min(distances) * 0.99 
y_max = max(distances) * 1.01 
axes[0,0].set_ylim(y_min, y_max)

axes[0,0].set_xlabel('Training Epochs')
axes[0,0].set_ylabel('Distance')
axes[0,0].set_title('Comparison of Distances for Different Training Epochs')
axes[0,0].set_xticks(x)
axes[0,0].set_xticklabels(labels)
axes[0,0].grid(axis='y', linestyle='--', alpha=0.7)


axes[0,1].bar(x, momentum_means, color=['blue', 'orange', 'green', 'red'])
for i, v in enumerate(momentum_means):
    axes[0,1].text(i, v, f'{v:.2e}', ha='center', va='bottom', fontsize=10)
axes[0,1].set_ylabel('Momentum Mean')
axes[0,1].set_title('Momentum Mean vs. Training Epochs')

# Momentum max/min plot
axes[1,0].plot(labels, momentum_maxs, marker='o', linestyle='-', color='red', label='Momentum Max')
axes[1,0].plot(labels, momentum_mins, marker='o', linestyle='-', color='blue', label='Momentum Min')
axes[1,0].fill_between(labels, momentum_mins, momentum_maxs, color='gray', alpha=0.3)
for i, (max_v, min_v) in enumerate(zip(momentum_maxs, momentum_mins)):
    axes[1,0].text(i, max_v, f'{max_v:.2e}', ha='center', va='bottom', fontsize=10)
    axes[1,0].text(i, min_v, f'{min_v:.2e}', ha='center', va='top', fontsize=10)
axes[1,0].set_ylabel('Momentum Max/Min')
axes[1,0].set_title('Momentum Max/Min vs. Training Epochs')
axes[1,0].legend()

# SSIM plot
axes[1,1].bar(x, ssim_values, color=['blue', 'orange', 'green', 'red'])
for i, v in enumerate(ssim_values):
    axes[1,1].text(i, v, f'{v:.6f}', ha='center', va='bottom', fontsize=10)

#focus on the y-axis
y_min = min(ssim_values) * 0.9
y_max = max(ssim_values) * 1.1
axes[1,1].set_ylim(y_min, y_max)

axes[1,1].set_ylabel('SSIM')
axes[1,1].set_title('SSIM vs. Training Epochs')

for ax in axes.flatten():
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()





# imgs_path = [
#     './Experiments/PreTrainEv/5epochs/registered_image_5.png',
#     './Experiments/PreTrainEv/50epochs/registered_image_50.png',
#     './Experiments/PreTrainEv/100epochs/registered_image_100.png',
#     './Experiments/PreTrainEv/250epochs/registered_image_250.png'
# ]

# #stack them in a 2x2 grid
# fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# for i, img_path in enumerate(imgs_path):
#     with open(img_path, 'r') as file:
#         img = plt.imread(img_path)
#         axes[i//2, i%2].imshow(img, cmap='gray')
#         axes[i//2, i%2].set_title(f'Registered Image - {labels[i]}')

# plt.tight_layout()

# plt.show()



# diff_imgs_path = [
#     './Experiments/PreTrainEv/5epochs/diff_image_5.png',
#     './Experiments/PreTrainEv/50epochs/diff_image_50.png',
#     './Experiments/PreTrainEv/100epochs/diff_image_100.png',
#     './Experiments/PreTrainEv/250epochs/diff_image_250.png'
# ]


# fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# for i, img_path in enumerate(diff_imgs_path):
#     with open(img_path, 'r') as file:
#         img = plt.imread(img_path)
#         axes[i//2, i%2].imshow(img, cmap='jet')
#         axes[i//2, i%2].set_title(f'Difference Image - {labels[i]}')

# plt.tight_layout()

# plt.show()
