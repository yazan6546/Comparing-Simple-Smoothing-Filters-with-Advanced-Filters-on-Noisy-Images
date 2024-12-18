import utilities as util
import os
import numpy as np
import cv2
import time
import pandas as pd
import filters

def calculate_psnr(original_image, filtered_image):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the original and filtered images.

    Parameters:
    - original_image: Original grayscale image (2D numpy array).
    - filtered_image: Filtered grayscale image (2D numpy array).

    Returns:
    - psnr: PSNR value.
    """

    mse = np.mean((original_image - filtered_image) ** 2)
    return float('inf') if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))

def get_metric_values_for_filter_and_noise(metric, base_dir, original_image_name, original_image, noise_level, noise_type, filter_types):
    """
    Get the metric values for different filters and noise types.

    Parameters:
    - metric: Metric function to calculate.
    - base_dir: Base directory containing the filtered images.
    - original_image_name: Name of the original image.
    - original_image: Original grayscale image (2D numpy array).
    - noise_level: Noise level (e.g., 'low', 'medium', 'high').
    - noise_type: Type of noise (e.g., 'Gaussian', 'Salt and Pepper').
    - filter_types: List of filter types.

    Returns:
    - metric_values: Dictionary containing the metric values for different filter types.
    - kernels: List of kernel sizes used for filtering.
    """

    metric_values = {filter_type: [] for filter_type in filter_types}
    kernels = []

    for filter_type in filter_types:
        dir_path = os.path.join(base_dir, original_image_name, noise_level, noise_type, filter_type)
        if not os.path.exists(dir_path):
            continue
        
        # Sort the files based on the kernel size
        files = os.listdir(dir_path)
        sorted_files = sorted(files, key=util.kernel_size_from_name)
                
        for file_name in sorted_files:

            if file_name.endswith('.png'):

                # Extract kernel size from the filename
                kernel_size = util.kernel_size_from_name(file_name)
                        
                # Construct the file path
                file_path = os.path.join(dir_path, file_name)
                        
                # Load the filtered image
                filtered_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        
                # Calculate MSE
                metric_value = metric(original_image.flatten(), filtered_image.flatten())

                if kernel_size not in kernels:   
                    kernels.append(kernel_size)

                metric_values[filter_type].append(metric_value)


    return metric_values, kernels



def collect_metric_values_for_all_filters_and_noise_types(metric, base_dir, original_image_name, original_image):
    """
    Collect the metric values for different filters and noise types.

    Parameters:
    - metric: Metric function to calculate.
    - base_dir: Base directory containing the filtered images.
    - original_image_name: Name of the original image.
    - original_image: Original grayscale image (2D numpy array).

    Returns:
    - metric_dict_outer: Dictionary containing the metric values for different noise types and noise levels.
    - kernels: List of kernel sizes used for filtering.
    """

    noise_levels = os.listdir(os.path.join(base_dir, original_image_name))
    noise_types = os.listdir(os.path.join(base_dir, original_image_name, noise_levels[0]))
    filter_types = os.listdir(os.path.join(base_dir, original_image_name, noise_levels[0], noise_types[0]))

    metric_dict_outer = {}

    for noise_type in noise_types:
        metric_dict_inner = {}
        for noise_level in noise_levels:
            metric_values, kernels = get_metric_values_for_filter_and_noise(metric, base_dir, original_image_name, original_image, noise_level, noise_type, filter_types)
            metric_dict_inner[noise_level] = metric_values
        metric_dict_outer[noise_type] = metric_dict_inner

    return metric_dict_outer, kernels


def measure_filter_time(image, filter_type, kernel_size, gaussian_std=0):
    """
    Measure the time taken to apply a filter on the input image.

    Parameters:
    - image: Input grayscale image (2D numpy array).
    - filter_type: Type of filter to apply.
    - kernel_size: Size of the kernel.
    - gaussian_std: Standard deviation for Gaussian filter.

    Returns:
    - elapsed_time: Time taken to apply the filter.
    """

    start_time = time.perf_counter()

    if filter_type in filters.FILTER_MAPPING:
        if filter_type == 'gaussian_filter':  # Special case for Gaussian filter
            filtered_image = filters.FILTER_MAPPING[filter_type](image, kernel_size, gaussian_std)
        else:
            filtered_image = filters.FILTER_MAPPING[filter_type](image, kernel_size)
    else:
        raise ValueError("Unsupported filter type")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return elapsed_time



def collect_filter_times(image, filter_types, kernel_sizes, gaussian_std=0):
    """"
    Collect the time taken to apply different filters on the input image.

    Parameters:
    - image: Input grayscale image (2D numpy array).
    - filter_types: List of filter types.
    - kernel_sizes: List of kernel sizes.
    - gaussian_std: Standard deviation for Gaussian filter.

    Returns:
    - df: DataFrame containing the time taken to apply different filters.
    """

    data = []

    for filter_type in filter_types:
        for kernel_size in kernel_sizes:
            elapsed_time = measure_filter_time(image, filter_type, kernel_size, gaussian_std)
            data.append({
                'Filter Type': filter_type,
                'Kernel Size': kernel_size,
                'Time': elapsed_time
            })

    # Create a DataFrame from the data
    df = pd.DataFrame(data)
    return df