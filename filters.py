import cv2
import os
import numpy as np

# Function to create directory structure and save images
def save_filtered_images(df, image_name, kernel_sizes=[3, 5, 7], gaussian_std=0):

    # Define the base directory
    base_dir = 'Images_filtered'

    # Define noise levels, noise types, and filters
    noise_levels = ['low', 'medium', 'high']
    noise_types = ['Gaussian', 'Salt and Pepper']
    
    filters = ['box_filter',
               'median_filter',
               'gaussian_filter',
               'adaptive_median_filter',
               'bilateral_filter',
               'adaptive_mean_filter']
    
    for noise_level in noise_levels:
        for noise_type in noise_types:

            key = f'{noise_type} Noise ({noise_level})'

            noisy_image = df.loc[key, 'Image']

            for filter_type in filters:
                for k in kernel_sizes:
                    # Create the directory path
                    dir_path = os.path.join(base_dir, image_name, noise_level, noise_type, filter_type)
                    os.makedirs(dir_path, exist_ok=True)
                    
                    # Apply the filter (example with GaussianBlur)
                    if filter_type == 'box_filter':
                        filtered_image = cv2.blur(noisy_image, (k, k))
                    elif filter_type == 'median_filter':
                        filtered_image = cv2.medianBlur(noisy_image, k)
                    elif filter_type == 'gaussian_filter':
                        filtered_image = cv2.GaussianBlur(noisy_image, (k, k), k * 3)
                    elif filter_type == 'adaptive_median_filter':
                        filtered_image = adaptive_median_filter(noisy_image, max_kernel_size=k)
                    elif filter_type == 'bilateral_filter':
                        filtered_image = cv2.bilateralFilter(noisy_image, k, 75, 75)
                    elif filter_type == 'adaptive_mean_filter':
                        global_variance = np.var(noisy_image)
                        filtered_image = adaptive_mean_filter(noisy_image, k, global_variance)


                    if (filtered_image.shape != noisy_image.shape):
                        raise ValueError(f"Filtered image shape {filtered_image.shape} does not match noisy image shape {noisy_image.shape}")
                    
                    # Save the filtered image
                    file_name = f"{image_name}_{noise_type}_{noise_level}_{filter_type}_k{k}.png"
                    file_path = os.path.join(dir_path, file_name)
                    cv2.imwrite(file_path, filtered_image)


def adaptive_median_filter(image, max_kernel_size=3):
    """
    Apply adaptive median filter to the input image.

    Parameters:
    - image: Input grayscale image (2D numpy array).
    - max_kernel_size: Maximum size of the kernel (must be an odd number).

    Returns:
    - output_image: Filtered image.
    """
    # Ensure the max_kernel_size is odd
    if max_kernel_size % 2 == 0:
        raise ValueError("max_kernel_size must be an odd number")

    # Pad the image to handle borders
    pad_size = max_kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    output_image = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            kernel_size = 3  # Start with the smallest kernel size
            pixel_filtered = image[i, j]  # Default to original pixel value
            
            while kernel_size <= max_kernel_size:
                # Extract the local neighborhood
                local_region = padded_image[i:i + kernel_size, j:j + kernel_size]
                
                # Calculate median, minimum, and maximum values in the neighborhood
                local_median = np.median(local_region)
                local_min = np.min(local_region)
                local_max = np.max(local_region)
                
                # Adaptive median filter conditions
                if local_min < local_median < local_max:
                    if local_min < image[i, j] < local_max:
                        pixel_filtered = image[i, j]  # Keep original pixel
                    else:
                        pixel_filtered = local_median  # Use median value
                    break  # Exit while loop as we have a valid result
                else:
                    # Increase kernel size
                    kernel_size += 2
            
            # Store the filtered pixel value
            output_image[i, j] = pixel_filtered

    return output_image




def adaptive_mean_filter(image, kernel_size, global_variance):
    """
    Apply adaptive mean filter to the input image.

    Parameters:
    - image: Input grayscale image (2D numpy array).
    - kernel_size: Size of the kernel.
    - global_variance: Global variance of the image.

    Returns:
    - output_image: Filtered image.
    """

    pad_size = kernel_size // 2
    padded_image = np.pad(image, pad_size, mode='constant', constant_values=0)
    output_image = np.copy(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            local_region = padded_image[i:i + kernel_size, j:j + kernel_size]
            local_mean = np.mean(local_region)
            local_variance = np.var(local_region)

            ratio = global_variance / local_variance if local_variance > global_variance else 1

            filtered_value = image[i, j] - ratio * (image[i, j] - local_mean)
            output_image[i, j] = filtered_value

    return output_image