import json
import numpy as np
import matplotlib.pyplot as plt

def plot_metrics(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    executions = list(data.keys())
    ssim_values = [data[ex]['ssim'] for ex in executions]
    rmse_values = [data[ex]['rmse'] for ex in executions]
    dice_values = [data[ex]['dice'] for ex in executions]
    
    avg_ssim = np.mean(ssim_values)
    avg_rmse = np.mean(rmse_values)
    avg_dice = np.mean(dice_values)
    
    plt.figure(figsize=(10, 5))
    plt.plot(executions, ssim_values, marker='o', label='SSIM')
    plt.plot(executions, rmse_values, marker='s', label='RMSE')
    plt.plot(executions, dice_values, marker='^', label='DICE')
    plt.axhline(avg_ssim, color='blue', linestyle='dashed', label=f'Avg SSIM: {avg_ssim:.3f}')
    plt.axhline(avg_rmse, color='orange', linestyle='dashed', label=f'Avg RMSE: {avg_rmse:.3f}')
    plt.axhline(avg_dice, color='green', linestyle='dashed', label=f'Avg DICE: {avg_dice:.3f}')
    
    plt.xlabel('Execution Index')
    plt.ylabel('Metric Values')
    plt.title('Performance Metrics per Execution')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    plt.savefig(f'{json_file}_metrics_plot.png')
    plt.show()
    
    print(f"Average SSIM: {avg_ssim:.3f}")
    print(f"Average RMSE: {avg_rmse:.3f}")
    print(f"Average DICE: {avg_dice:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot metrics from JSON file.')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing metrics.')
    args = parser.parse_args()
    
    plot_metrics(args.json_file)

