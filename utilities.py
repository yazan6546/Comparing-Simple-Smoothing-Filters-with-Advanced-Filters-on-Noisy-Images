import noise
import pandas as pd
import os
import shutil


def create_dataframe_image(original_image):
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

  # Extract kernel size from the filename
  name_without_extension = os.path.splitext(file_name)[0]
  parts = name_without_extension.split('_')[-1]
  return int(parts[1:])