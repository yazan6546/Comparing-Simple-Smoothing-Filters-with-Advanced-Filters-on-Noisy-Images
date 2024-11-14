import image_processing as ip
import pandas as pd
import os
import shutil
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def create_dataframe_image(original_image):
  # Define noise parameters


  gaussian_std = [15, 50, 120]
  salt_pepper_probs  = [0.01, 0.03, 0.1]
  values = ['low', 'medium', 'high']

  # Create lists for DataFrame
  index = []
  noisy_images = []

  index.append('no_noise')
  noisy_images.append(original_image)

  # Apply Gaussian noise with different variances
  for value, sigma in zip(values, gaussian_std):
      noisy_img = ip.add_gaussian_noise(original_image, sigma=sigma)
      index.append(f'Gaussian Noise ({value})')
      noisy_images.append(noisy_img)

  # Apply Salt-and-Pepper noise with different probabilities
  for value, prob in zip(values, salt_pepper_probs):
      noisy_img = ip.add_salt_and_pepper_noise(original_image, prob=prob)
      index.append(f'Salt and Pepper Noise ({value})')
      noisy_images.append(noisy_img)

  # Create DataFrame
  df = pd.DataFrame({'Noise_Type': index, 'Image': noisy_images}).set_index('Noise_Type')
  return df


# Function to create a directory, replacing it if it already exists
def create_or_replace_dir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)

def kernel_size_from_name(file_name):

    # Extract kernel size from the filename
    name_without_extension = os.path.splitext(file_name)[0]
    parts = name_without_extension.split('_')[-1]
    kernel_size = int(parts[1:])

    return kernel_size

# Function to create directory structure and save images
def save_filtered_images(df, image_name, kernel_sizes=[3, 5, 7]):

    # Define the base directory
    base_dir = 'Images_filtered'

    # Define noise levels, noise types, and filters
    noise_levels = ['low', 'medium', 'high']
    noise_types = ['Gaussian', 'Salt and Pepper']
    filters = ['box_filter', 'median_filter', 'gaussian_filter']
    
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
                        filtered_image = cv2.GaussianBlur(noisy_image, (k, k), 0)


                    if (filtered_image.shape != noisy_image.shape):
                        raise ValueError(f"Filtered image shape {filtered_image.shape} does not match noisy image shape {noisy_image.shape}")
                    
                    # Save the filtered image
                    file_name = f"{image_name}_{noise_type}_{noise_level}_{filter_type}_k{k}.png"
                    file_path = os.path.join(dir_path, file_name)
                    cv2.imwrite(file_path, filtered_image)

# # Example usage
# image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']
# for image_path in image_paths:
#     image_name = os.path.splitext(os.path.basename(image_path))[0]
#     save_filtered_images(image_path, image_name)



def get_mse_values_for_filter_and_noise(base_dir, original_image_name, original_image, noise_level, noise_type, filter_types):
    mse_values = {filter_type: [] for filter_type in filter_types}
    kernels = []

    for filter_type in filter_types:
        dir_path = os.path.join(base_dir, original_image_name, noise_level, noise_type, filter_type)
        if not os.path.exists(dir_path):
            continue
        
        # Sort the files based on the kernel size
        files = os.listdir(dir_path)
        sorted_files = sorted(files, key=kernel_size_from_name)
                
        for file_name in sorted_files:

            if file_name.endswith('.png'):

                # Extract kernel size from the filename
                kernel_size = kernel_size_from_name(file_name)
                        
                # Construct the file path
                file_path = os.path.join(dir_path, file_name)
                        
                # Load the filtered image
                filtered_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        
                # Calculate MSE
                mse = mean_squared_error(original_image.flatten(), filtered_image.flatten())

                if kernel_size not in kernels:   
                    kernels.append(kernel_size)

                mse_values[filter_type].append(mse)


    return mse_values, kernels



def collect_mse_values_for_all_filters_and_noise_types(base_dir, original_image_name, original_image):

    noise_levels = os.listdir(os.path.join(base_dir, original_image_name))
    noise_types = os.listdir(os.path.join(base_dir, original_image_name, noise_levels[0]))
    filter_types = os.listdir(os.path.join(base_dir, original_image_name, noise_levels[0], noise_types[0]))

    mse_dict_outer = {}

    for noise_type in noise_types:
        mse_dict_inner = {}
        for noise_level in noise_levels:
            mse_values, kernels = get_mse_values_for_filter_and_noise(base_dir, original_image_name, original_image, noise_level, noise_type, filter_types)
            mse_dict_inner[noise_level] = mse_values
        mse_dict_outer[noise_type] = mse_dict_inner

    return mse_dict_outer, kernels




def plot_mse_vs_kernel(mse_dict_outer, noise_levels, filter_types, kernel_sizes):
    num_noise_types = len(mse_dict_outer)
    num_noise_levels = len(noise_levels)
    
    fig, axes = plt.subplots(num_noise_types, num_noise_levels, figsize=(15, 5 * num_noise_types), sharey=True)
    
    for i, (noise_type, mse_dict_inner) in enumerate(mse_dict_outer.items()):
        for j, noise_level in enumerate(noise_levels):
            ax = axes[i, j] if num_noise_types > 1 else axes[j]
            for filter_type in filter_types:
                if noise_level in mse_dict_inner and filter_type in mse_dict_inner[noise_level]:
                    mse_values = mse_dict_inner[noise_level][filter_type]
                    ax.plot(kernel_sizes, mse_values, label=filter_type)
            
            ax.set_title(f'{noise_type} - {noise_level}')
            ax.set_xlabel('Kernel Size')
            if j == 0:
                ax.set_ylabel('Mean Squared Error (MSE)')
            ax.legend()
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()

# Example usage
base_dir = 'Images_filtered'
original_image_name = 'image_0'
original_image= cv2.imread('images/image_0.png', cv2.IMREAD_GRAYSCALE)
filter_types = ['box_filter', 'median_filter', 'gaussian_filter']

mse_dict_outer, kernels = collect_mse_values_for_all_filters_and_noise_types(base_dir, original_image_name, original_image)
# print(mse_dict_outer)

noise_levels = ['low', 'medium', 'high']

print(f'ker = {kernels}')

print(mse_dict_outer['Gaussian']['low']['box_filter'])

plot_mse_vs_kernel(mse_dict_outer, noise_levels, filter_types, kernels)