import image_processing as ip
import pandas as pd

def create_dataframe_image(original_image):
  # Define noise parameters


  gaussian_std = [5, 25, 75]
  salt_pepper_probs  = [0.05, 0.10, 0.25]

  # Create lists for DataFrame
  index = []
  noisy_images = []

  index.append('no_noise')
  noisy_images.append(original_image)

  # Apply Gaussian noise with different variances
  for sigma in gaussian_std:
      noisy_img = ip.add_gaussian_noise(original_image, sigma=sigma)
      index.append(f'Gaussian_Noise_sigma_{sigma}')
      noisy_images.append(noisy_img)

  # Apply Salt-and-Pepper noise with different probabilities
  for prob in salt_pepper_probs:
      noisy_img = ip.add_salt_and_pepper_noise(original_image, prob=prob)
      index.append(f'Salt_and_Pepper_prob_{prob}')
      noisy_images.append(noisy_img)

  # Create DataFrame
  df = pd.DataFrame({'Noise_Type': index, 'Image': noisy_images}).set_index('Noise_Type')
  return df
