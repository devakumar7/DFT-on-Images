# Image Processing and Data Update

This Python script performs image processing on a set of images and updates corresponding data in an Excel file.

## Features

1. **Generate DFT Images:** Computes the Discrete Fourier Transform (DFT) of grayscale images and saves the magnitude to new image files.

2. **Detect and Highlight Points:** Applies thresholding to detect regions in images, highlights them, and saves the processed images. Updates an Excel file with the count of detected points for each image at various threshold levels.

3. **Update Excel with Orientation Angle at Different Thresholds:** Applies edge detection and Hough Line Transform to calculate the orientation angle of lines in images. Updates the Excel file with the calculated orientation angles.

## Usage

1. Install the required libraries:

   ```bash
   pip install opencv-python numpy xlrd xlwt
   
cropped_images_folder_path = 'path_to_cropped_images_folder'
dft_on_cropped_images_folder_path = 'path_to_dft_output_folder'
detected_feather_points_folder_path = 'path_to_detected_points_output_folder'

Sure, let's draft a GitHub README for your code:

markdown

# Image Processing and Data Update

This Python script performs image processing on a set of images and updates corresponding data in an Excel file.

## Features

1. **Generate DFT Images:** Computes the Discrete Fourier Transform (DFT) of grayscale images and saves the magnitude to new image files.

2. **Detect and Highlight Points:** Applies thresholding to detect regions in images, highlights them, and saves the processed images. Updates an Excel file with the count of detected points for each image at various threshold levels.

3. **Update Excel with Orientation Angle:** Applies edge detection and Hough Line Transform to calculate the orientation angle of lines in images. Updates the Excel file with the calculated orientation angles.

## Usage

1. Install the required libraries:

   **by using command: **
    pip install opencv-python numpy xlrd xlwt
   
    Set the paths for input and output folders in the script:

    cropped_images_folder_path = 'path_to_cropped_images_folder'
    dft_on_cropped_images_folder_path = 'path_to_dft_output_folder'
    detected_feather_points_folder_path = 'path_to_detected_points_output_folder'

Run the script:

bash

    python script_name.py

Dependencies

    OpenCV: Image processing library.
    NumPy: Library for numerical operations.
    xlrd, xlwt: Libraries for working with Excel files.

Make sure to replace `'path_to_cropped_images_folder'`, `'path_to_dft_output_fold
