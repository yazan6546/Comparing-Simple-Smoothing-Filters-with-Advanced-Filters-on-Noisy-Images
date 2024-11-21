import noise
import pandas as pd
import os
import shutil

# Define the base directory
BASE_DIR = 'Images_filtered'

def create_dataframe_image(original_image):
    """
    Create a DataFrame containing the original image and noisy images with different noise levels.

    Parameters:
    - original_image: Original grayscale image (2D numpy array).

    Returns:
    - df: DataFrame
    """
  # Define noise parameters


    gaussian_std = [10, 25, 50]
    salt_pepper_probs  = [0.01, 0.03, 0.10]
    values = ['low', 'medium', 'high']

    # Create lists for DataFrame
    index = []
    noisy_images = []

    index.append('no_noise')
    noisy_images.append(original_image)

    # Apply Gaussian noise with different variances
    for value, sigma in zip(values, gaussian_std):
        noisy_img = noise.add_gaussian_noise(original_image, sigma=sigma)
        index.append(f'Gaussian Noise ({value})')
        noisy_images.append(noisy_img)

    # Apply Salt-and-Pepper noise with different probabilities
    for value, prob in zip(values, salt_pepper_probs):
        noisy_img = noise.add_salt_and_pepper_noise(original_image, prob=prob)
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
  """
    Extract the kernel size from the filename.

    Parameters:
    - file_name: Name of the file.

    Returns:
    - kernel_size: Size of the kernel used for filtering.
    """

  # Extract kernel size from the filename
  name_without_extension = os.path.splitext(file_name)[0]
  parts = name_without_extension.split('_')[-1]
  return int(parts[1:])


def get_path_filtered(image_name, noise_level, noise_type, filter_type, kernel_size, create_dir=True):
    """
    Construct the file path for a filtered image.

    Parameters:
    - base_dir: Base directory for filtered images.
    - image_name: Name of the original image.
    - noise_level: Noise level (e.g., 'low', 'medium', 'high').
    - noise_type: Type of noise (e.g., 'Gaussian', 'Salt and Pepper').
    - filter_type: Type of filter applied.
    - kernel_size: Size of the kernel used for filtering.
    - create_dir: Whether to create the directory if it does not exist.

    Returns:
    - file_path: Constructed file path for the filtered image.
    """

    dir_path = os.path.join(BASE_DIR, image_name, noise_level, noise_type, filter_type)

    if create_dir:
        os.makedirs(dir_path, exist_ok=True)
    
    file_name = f"{image_name}_{noise_type}_{noise_level}_{filter_type}_k{kernel_size}.png"
    file_path = os.path.join(dir_path, file_name)
    return file_path

