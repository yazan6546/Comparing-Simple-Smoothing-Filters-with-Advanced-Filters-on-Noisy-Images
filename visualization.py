

import matplotlib.pyplot as plt
import cv2

def display_images_noises(original_image, gaussian_noise, salt_pepper_noise, intensity):

  # Assuming you have the original image and noisy images
  images = [original_image, gaussian_noise, salt_pepper_noise]
  titles = ["Original", "Gaussian Noise " + intensity, "Salt-and-Pepper Noise " + intensity]

  # Create a figure with subplots
  fig, axes = plt.subplots(1, 4, figsize=(20, 5))  # 1 row, 4 columns

  # Loop through the images and titles
  for ax, img, title in zip(axes, images, titles):
      ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
      ax.set_title(title)
      ax.axis('off')  # Turn off axes for better presentation

  plt.tight_layout()
  plt.show()


def to_numpy(tensor):

  # Convert _TakeDataset to list of tensors
  dataset_list = list(tensor)

  # Convert list of tensors to list of NumPy arrays
  numpy_list = [item.numpy() for item in dataset_list]

  # Convert list of NumPy arrays to a single NumPy array
  numpy_array = np.array(numpy_list)

  return numpy_array
