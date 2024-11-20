import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy_image = np.clip(image + gauss, 0, 255).astype(np.uint8)
    return noisy_image

def add_salt_and_pepper_noise(image, prob=0.05):
    noisy_image = np.copy(image)
    total_pixels = image.size
    num_salt = int(total_pixels * prob * 0.5)
    num_pepper = int(total_pixels * prob * 0.5)
    salt_coords = [np.random.randint(0, i-1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    pepper_coords = [np.random.randint(0, i-1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0
    return noisy_image
