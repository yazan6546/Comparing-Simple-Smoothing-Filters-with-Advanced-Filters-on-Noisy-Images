import cv2
import os


# Function to create directory structure and save images
def save_filtered_images(df, image_name, kernel_sizes=[3, 5, 7], gaussian_std=0):

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
                        filtered_image = cv2.GaussianBlur(noisy_image, (k, k), gaussian_std)


                    if (filtered_image.shape != noisy_image.shape):
                        raise ValueError(f"Filtered image shape {filtered_image.shape} does not match noisy image shape {noisy_image.shape}")
                    
                    # Save the filtered image
                    file_name = f"{image_name}_{noise_type}_{noise_level}_{filter_type}_k{k}.png"
                    file_path = os.path.join(dir_path, file_name)
                    cv2.imwrite(file_path, filtered_image)
                    


def adaptive_median_filter(image, max_kernel_size=7):
    # Initialize output image
    output_image = np.zeros_like(image)
    padded_image = np.pad(image, max_kernel_size // 2, mode='reflect')
    
    # Process each pixel in the image
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