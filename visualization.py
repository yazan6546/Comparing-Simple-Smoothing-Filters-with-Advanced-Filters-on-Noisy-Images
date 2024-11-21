import matplotlib.pyplot as plt
import math
import utilities as utils
import cv2


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
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
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
    """
    Plot the time taken to apply different filters against the kernel sizes.

    Parameters:
    - times: DataFrame containing the time taken to apply different filters.

    """

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




def plot_original_noisy_images(df, noise_types, intesities, type):
    """
    Plot the original image and the noisy variants side by side.

    Parameters:
    - dataframes: List of DataFrames containing the noisy images.
    - noise_types: List of noise types.
    """

    fontsize_subtitle = 23
    
    image = df.loc['no_noise', 'Image']
    fig, axes = plt.subplots(3, 3, figsize=(30, 20))

    # Iterate over all images
    for i, intensity, in enumerate(intesities):
        
         # Display the original image
        axes[i, 0].imshow(image, cmap='gray')
        axes[i, 0].set_title('Original', fontsize=fontsize_subtitle)
        axes[i, 0].axis('off')
        
        # Display the noisy variants
        for j, noise in enumerate(noise_types):
            
            noisy_image_key = f'{noise} ({intensity})'

            noisy_image = df.loc[noisy_image_key, 'Image']
            axes[i, j + 1].imshow(noisy_image, cmap='gray')
            axes[i, j + 1].set_title(noisy_image_key, fontsize=fontsize_subtitle)
            axes[i, j + 1].axis('off')
        
    fig.suptitle(f'Different noises on {type} image' , fontsize=35)
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust layout to make room for the suptitle
    plt.subplots_adjust(hspace=0.1)  # Increase the height space between rows
    plt.show()


def plot_original_noisy_filtered_images(df, noise_type, noise_intensity, filter_types, original_image_name, kernel_size):

    """
    Plot the original image, noisy image, and filtered images side by side.

    Parameters:
    - df: DataFrame containing the images.
    - noise_type: Type of noise.
    - noise_intensity: Intensity of noise.
    - filter_types: List of filter types.
    - original_image_name: Name of the original image.
    - kernel_size: 
    """

    fontsize_subtitle=26
    
    image = df.loc['no_noise', 'Image']
    noisy_image = df.loc[f'{noise_type} Noise ({noise_intensity})', 'Image']
    
    fig, axes = plt.subplots(3, 3, figsize=(30, 20))
    # Display original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image', fontsize=fontsize_subtitle)
    axes[0, 0].axis('off')

    # Display noisy image (Gaussian medium)
    axes[0, 1].imshow(noisy_image, cmap='gray')
    axes[0, 1].set_title(f'Noisy Image ({noise_type} {noise_intensity})', fontsize=fontsize_subtitle)
    axes[0, 1].axis('off')

    # Hide the third subplot in the first row
    axes[0, 2].axis('off')

    # Display filtered images in the second row
    for j, filter_type in enumerate(filter_types[:3]):

        image_path = utils.get_path_filtered(original_image_name, noise_intensity, noise_type, filter_type, kernel_size, create_dir=False)

        filtered_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        axes[1, j].imshow(filtered_image, cmap='gray')
        axes[1, j].set_title(f'{filter_type.replace("_", " ").title()}', fontsize=fontsize_subtitle)
        axes[1, j].axis('off')

    # Display filtered images in the third row
    for j, filter_type in enumerate(filter_types[3:]):

        image_path = utils.get_path_filtered(original_image_name, noise_intensity, noise_type, filter_type, kernel_size, create_dir=False)

        filtered_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        axes[2, j].imshow(filtered_image, cmap='gray')
        axes[2, j].set_title(f'{filter_type.replace("_", " ").title()}', fontsize=fontsize_subtitle)
        axes[2, j].axis('off')

    fig.suptitle(f'Filtering the image corrupted with {noise_type} ({noise_intensity})' , fontsize=35)
    plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust layout to make room for the suptitle
    plt.subplots_adjust(hspace=0.1)  # Increase the height space between rows
    plt.show()    

    