# Image Filtering and Visualization

This repository contains code for comparing the performance of various image smoothing filters applied to noisy images. The filters include simple smoothing filters such as Box filter, Gaussian filter, and Median filter, as well as advanced filters like Bilateral filter, Adaptive Median filter, and Adaptive Mean filter. The evaluation focuses on noise removal effectiveness, edge preservation, computational efficiency, and the influence of kernel size on the performance of each filter. Quantitative metrics such as Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR) are used for comparison.

## Table of Contents
- [Features](#features)
- [Filters](#filters)
  - [Box Filter](#box-filter)
  - [Median Filter](#median-filter)
  - [Gaussian Filter](#gaussian-filter)
  - [Bilateral Filter](#bilateral-filter)
  - [Adaptive Median Filter](#adaptive-median-filter)
  - [Adaptive Mean Filter](#adaptive-mean-filter)
- [Running the Code](#running-the-code)
- [Jupyter Notebook](#jupyter-notebook)
  - [Converting Notebook to PDF](#converting-notebook-to-pdf)
- [Repository Contents](#repository-contents)
- [Directory Structure](#directory-structure)
- [Contributing](#contributing)

## Features

- **Image Filtering**: Apply different filters to noisy images.
- **Performance Metrics**: Measure the performance of filters using MSE and PSNR.
- **Visualization**: Visualize the original, noisy, and filtered images.
- **Edge Detection**: Visualize edge detection results for original and noisy images.

## Filters

### Box Filter
A simple linear filter that averages the pixel values within a defined kernel size.

### Median Filter
A non-linear filter that replaces each pixel's value with the median value of its neighboring pixels.

### Gaussian Filter
A linear filter that applies a Gaussian function to the neighboring pixels, giving more weight to closer pixels.

### Bilateral Filter
An advanced filter that considers both spatial distance and intensity difference between pixels.

### Adaptive Median Filter
Adapts the size of the filtering window based on the local characteristics of the image.

### Adaptive Mean Filter
Reduces noise by adjusting the filtering process based on local variance.


## Running the Code

1.  Clone this repository to your local machine.
2.  Run this command:
   
    ```bash
    pip install requirements.txt
    ```
3.  Open the `maain.ipynb` notebook in Google Colab or a Jupyter environment.
  
5.  Make sure you have a folder called `images` in the same directory, add images of your choice to it,
    with each image having the name `image_{number}.png`.
    
6.  Execute the code cells sequentially to perform the analysis and generate visualizations.

## Jupyter Notebook

The repository includes a Jupyter Notebook (`main.ipynb`) that serves as the main project file. It demonstrates the usage of the filtering and visualization functions, providing a step-by-step guide to applying filters, measuring their performance, and visualizing the results. The notebook also includes examples of edge detection and the effect of different kernel sizes on filtering performance.

## Converting Notebook to PDF
To convert the Jupyter Notebook to a PDF file with a table of contents, use the following command:

   ```bash
   jupyter nbconvert --to pdf --template toc_template.tplx [main.ipynb](http://_vscodecontentref_/0)
   ```
The pdf is already included in `reports` directory for convenience.

## Repository Contents

- `filters.py`: Contains functions for applying various image filters.
- `metrics.py`: Contains functions for measuring the performance of filters using MSE and PSNR.
- `visualization.py`: Contains functions for visualizing the original, noisy, and filtered images, as well as edge detection results.
- `utilities.py`: Contains utility functions for path management and directory creation.
- `main.ipynb`: Jupyter Notebook demonstrating the usage of the filtering and visualization functions.
- `requirements.txt`: List of required dependencies for the project.
- `images/`: Directory containing the input images.
- `Images_filtered/`: Directory structure for storing the filtered images.

## Directory Structure

The Images_filtered directory contains the filtered images organized in a structured manner. The directory structure is as follows:

    - Images_filtered/
    ├── image_name/
    │   ├── noise_level/
    │   │   ├── noise_type/
    │   │   │   ├── filter_type/
    │   │   │   │   ├── image_name_noise_type_noise_level_filter_type_k3.png
    │   │   │   │   ├── image_name_noise_type_noise_level_filter_type_k5.png
    │   │   │   │   ├── ...

### Explanation
- `image_name`: The name of the original image.
- `noise_level`: The level of noise applied to the image (e.g., low, medium, high).
- `noise_type`: The type of noise applied to the image (e.g., Gaussian, Salt and Pepper).
- `filter_type`: The type of filter applied to the noisy image (e.g., box_filter, median_filter).
- `image_name_noise_type_noise_level_filter_type_k{X}.png`: The filtered image file, where kX indicates the kernel size used for filtering.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.
