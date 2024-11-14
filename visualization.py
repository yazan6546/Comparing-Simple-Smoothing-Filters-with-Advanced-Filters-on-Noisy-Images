import matplotlib.pyplot as plt
import cv2



def plot_metric_vs_kernel(metric_dict_outer, ylabel, noise_levels, filter_types, kernel_sizes, number):
    num_noise_types = len(metric_dict_outer)
    num_noise_levels = len(noise_levels)
    
    fig, axes = plt.subplots(num_noise_types, num_noise_levels, figsize=(15, 5 * num_noise_types), sharey=True)
    
    for i, (noise_type, metric_dict_inner) in enumerate(metric_dict_outer.items()):
        for j, noise_level in enumerate(noise_levels):
            ax = axes[i, j] if num_noise_types > 1 else axes[j]
            for filter_type in filter_types:
                if noise_level in metric_dict_inner and filter_type in metric_dict_inner[noise_level]:
                    metric_values = metric_dict_inner[noise_level][filter_type]
                    ax.plot(kernel_sizes, metric_values, label=filter_type)
            
            ax.set_title(f'{noise_type} - {noise_level}')
            ax.set_xlabel('Kernel Size')
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)

    fig.suptitle(f'{ylabel} vs Kernel Size for Image {number}', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjust layout to make room for the suptitle
    fig.subplots_adjust(top=0.85)  # Fine-tune the spacing below the title

    
    plt.tight_layout()
    plt.show()
