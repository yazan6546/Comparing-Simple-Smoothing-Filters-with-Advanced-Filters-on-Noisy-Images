import image_processing as ip
import pandas as pd
import os
import cv2

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
      index.append(f'Salt and Pepper ({value})')
      noisy_images.append(noisy_img)

  # Create DataFrame
  df = pd.DataFrame({'Noise_Type': index, 'Image': noisy_images}).set_index('Noise_Type')
  return df

# Define the base directory
base_dir = 'Images_filtered'

# Define noise levels, noise types, and filters
noise_levels = ['low', ' medium', 'high']
noise_types = ['Gaussian', 'Salt and Pepper']
filters = ['box_filter', 'median_filter', 'gaussian_filter']
kernel_sizes = [3, 5, 7]

# Function to create directory structure and save images
def save_filtered_images(df, image_name):

    # Define the base directory
    base_dir = 'Images_filtered'

    # Define noise levels, noise types, and filters
    noise_levels = ['low', ' medium', 'high']
    noise_types = ['Gaussian', 'Salt and Pepper']
    filters = ['box_filter', 'median_filter', 'gaussian_filter']
    kernel_sizes = [3, 5, 7]
    
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
                    
                    # Save the filtered image
                    file_name = f"{image_name}_{noise_type}_{noise_level}_{filter_type}_k{k}.png"
                    file_path = os.path.join(dir_path, file_name)
                    cv2.imwrite(file_path, filtered_image)

# # Example usage
# image_paths = ['path_to_image1.jpg', 'path_to_image2.jpg']
# for image_path in image_paths:
#     image_name = os.path.splitext(os.path.basename(image_path))[0]
#     save_filtered_images(image_path, image_name)


