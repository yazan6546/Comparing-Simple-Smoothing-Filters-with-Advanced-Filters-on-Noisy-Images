import matplotlib.pyplot as plt
import math


def plot_metric_vs_kernel(metric_dict_outer, ylabel, noise_levels, filter_types, kernel_sizes, number):
    """
    Plot the metric values against the kernel sizes for different noise levels and types.

    Parameters:
    - metric_dict_outer: Dictionary containing the metric values for different noise types.
    - ylabel: Label for the y-axis.
    - noise_levels: List of noise levels.
    - filter_types: List of filter types.
    - kernel_sizes: List of kernel sizes.
    - number: Image number.
    """

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



def plot_images(list_of_images, titles):
    """
    Plot a list of images with their corresponding titles.

    Parameters:
    - list_of_images: List of images to be displayed.
    - titles: List of titles for the images.
    """
    
    # Display the image in the notebook
    fig, axes = plt.subplots(math.ceil(len(list_of_images)/3), 3, figsize=(15, 5))
    for i, img in enumerate(list_of_images):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    plt.show()


def plot_time_vs_kernel(times):
    # Unstack the DataFrame to have 'Kernel Size' as the index and 'Filter Type' as columns
    df_unstacked = times['Time'].unstack(level=0)
    
    # Plot using pandas' plot method
    ax = df_unstacked.plot(kind='line', figsize=(10, 6), marker='o')
    ax.set_yscale('log')
    
    ax.set_xlabel('Kernel Size')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Time vs Kernel Size for Different Filters')
    ax.grid(True)
    
    plt.show()



def visualize_edges(original_image, edges_image):
    """
    Visualize the original image and the edges detected by the Canny edge detector.

    Parameters:
    - original_image: The original grayscale image.
    - edges_image: The image with edges detected by the Canny edge detector.
    """
    plt.figure(figsize=(10, 5))

    # Display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Display the edges image
    plt.subplot(1, 2, 2)
    plt.imshow(edges_image, cmap='gray')
    plt.title('Edges Image')
    plt.axis('off')

    plt.show()



def plot_original_noisy_images(dataframes, noise_types, intensity):
    """
    Plot the original image and the noisy variants side by side.

    Parameters:
    - dataframes: List of DataFrames containing the noisy images.
    - noise_types: List of noise types.
    """
    
# Iterate over all images
    for i, df in enumerate(dataframes):

        image = df.loc['no_noise', 'Image']

        _, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        # Display the original image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original')
        axes[0].axis('off')
    
        # Display the noisy variants
        for j, noise_type in enumerate(noise_types):
            noisy_image = dataframes[i].loc['noisy_image ({intensity})', 'Image']
            axes[j + 1].imshow(noisy_image, cmap='gray')
            axes[j + 1].set_title('{noise_type} ({intensity})')
            axes[j + 1].axis('off')
    
        plt.show()