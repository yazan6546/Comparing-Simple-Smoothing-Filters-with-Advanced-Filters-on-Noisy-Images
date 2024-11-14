import utilities as util
import os
import numpy as np
import cv2

def calculate_psnr(original_image, filtered_image):

    mse = np.mean((original_image - filtered_image) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def get_metric_values_for_filter_and_noise(metric, base_dir, original_image_name, original_image, noise_level, noise_type, filter_types):
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
