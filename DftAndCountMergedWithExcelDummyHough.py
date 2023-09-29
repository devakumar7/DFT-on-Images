import cv2
import numpy as np
import os
import xlrd
from xlwt import Workbook

# Declare the path for the dummy Excel file globally
DUMMY_EXCEL_PATH = 'C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\temp_dummy_example_with_data.xls'

def generate_dft_images(input_folder_path, destination_folder_path):
    # Create the destination folder if it doesn't exist
    print(f"1111111111111111111111111111")
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        # Construct the full path of the input image file
        input_image_path = os.path.join(input_folder_path, image_file)
        
        print(">>>>>>>>>>>>>>>>>>>>>", image_file)
        
        #update_excel_with_orientation_angle('C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\temp.xls', image_file, input_image_path)
        
        # Load the image in grayscale
        gray = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)      

        # Compute the discrete Fourier Transform of the image
        fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)

        # Shift the zero-frequency component to the center of the spectrum
        fourier_shift = np.fft.fftshift(fourier)

        # Calculate the magnitude of the Fourier Transform
        magnitude = 20 * np.log(cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))

        # Scale the magnitude for display
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

        # Save the magnitude of the Fourier Transform to a file
        output_filename = os.path.splitext(image_file)[0] + '_Fourier_Magnitude.png'
        output_path = os.path.join(destination_folder_path, output_filename)
        cv2.imwrite(output_path, magnitude)

        print(f"Saved the Fourier Transform magnitude of {image_file} to {output_filename}")

def detect_and_highlight_points(input_folder_path, destination_folder_path, threshold_values):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for threshold_value in threshold_values:
        for image_file in image_files:
                
            print(f"Name of the {image_file} ")

            # Split the string at "_cropped"
            parts = image_file.split('_cropped')

            # Take the first part (before "_cropped")
            desired_part = parts[0]

            # Print the desired part
            print(desired_part)
        
            # Construct the full path of the input image file
            input_image_path = os.path.join(input_folder_path, image_file)

            # Load the input image
            input_image = cv2.imread(input_image_path)

            # Convert to grayscale
            gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            
            # Apply thresholding
            ret, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours or rectangles on the original image
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save the processed image with detected areas
            output_file = os.path.splitext(image_file)[0] + f'_{threshold_value}_detected.png'
            output_path = os.path.join(destination_folder_path, output_file)
            cv2.imwrite(output_path, input_image)

            # Count the detected regions
            num_detected_regions = len(contours)
            
            # Create an object with desired_part and detectedpoints
            my_object = {
                "desired_part": desired_part,
                "detected_points": num_detected_regions,
                "threshold_value":threshold_value,
                #"orientation_angle": orientation_angle
            }
            
            update_excel_with_image_name('C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\temp.xls', my_object)
            print(f"Number of detected regions in {image_file} with threshold {threshold_value}: {num_detected_regions}")

            print(f"Processed image with highlighted areas (threshold {threshold_value}) saved as {output_path}")

def update_excel_with_image_name(file_path, my_object):
    """
    Update an Excel file at the given path with new data for the specified image name.
    """
    
    image_name = my_object["desired_part"]    
    detected_points_value = my_object["detected_points"]
    column_name="detectedAt"+str(my_object["threshold_value"])
        
    # Open the existing Excel file for reading
    workbook = xlrd.open_workbook(file_path, formatting_info=True)

    # Open the first sheet (assuming it's 'Sheet 1')
    sheet = workbook.sheet_by_index(0)

    # Find the row with the specified image name
    target_row_index = None
    for row_index in range(1, sheet.nrows):  # Start from the second row (headers are in the first row)
        if sheet.cell_value(row_index, 0) == image_name:
            target_row_index = row_index
            break
            

    if target_row_index is not None:
        # Create a new workbook for writing
        new_wb = Workbook()
        new_sheet = new_wb.add_sheet('Sheet 1')

        # List of header rows
        headers = sheet.row_values(0)

        # Write the headers to the new sheet
        for col, header in enumerate(headers):
            new_sheet.write(0, col, header)

        print(headers.index(column_name))
        column_name_temp=headers.index(column_name)
        print(headers.index(column_name))

        # Copy data from the existing file to the new file
        for row_index in range(1, sheet.nrows):
            if row_index == target_row_index:
                # Update data for the target row and specified column
                for col in range(sheet.ncols):
                    # if col == 0:
                        # new_sheet.write(target_row_index, col, image_name+"Updated")  # Update imageName
                    if col == headers.index(column_name):
                        # Update the specified column with new data
                        new_sheet.write(target_row_index, col, detected_points_value)  # Update detectedAt* column
                    else:
                        new_sheet.write(target_row_index, col, sheet.cell_value(row_index, col))
            else:
                # Copy unchanged data for other rows
                for col in range(sheet.ncols):
                    new_sheet.write(row_index, col, sheet.cell_value(row_index, col))

        # Save the new workbook with updated data to a temporary file
        temp_file_path = 'C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\temp_dummy_example_with_data.xls'
        new_wb.save(DUMMY_EXCEL_PATH)

        # Close the existing file
        workbook.release_resources()

        # Remove the original file
        os.remove(file_path)

        # Rename the temporary file to the original file's name
        os.rename(DUMMY_EXCEL_PATH, file_path)

    else:
        print(f"Record with imageName='{image_name}' not found in the Excel file.")


def update_excel_with_orientation_angle(input_folder_path, file_path):
    # Get a list of all image files in the input folder
    image_files = [f for f in os.listdir(input_folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for image_file in image_files:
        # Construct the full path of the input image file
        input_image_path = os.path.join(input_folder_path, image_file)

        print(">>>>>>>>>>>>>>>>>>>>>", image_file)

        # Load the image in grayscale
        gray = cv2.imread(input_image_path, cv2.COLOR_BGR2GRAY)

        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Line Transform
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)

        if lines is not None:
            # Initialize the angle variable
            angle = None
            for rho, theta in lines[:, 0]:
                # Calculate angle in degrees
                angle = theta * 180 / np.pi
        else:
            print("No lines detected")

        print(">>>>>>>>>>>>>>>>>>>>>", angle)

        # Split the string at "_cropped"
        parts = image_file.split('_cropped')

        # Take the first part (before "_cropped")
        desired_part = parts[0]

        # Print the desired part
        print(desired_part)
        image_name = desired_part

        # Open the existing Excel file for reading
        workbook = xlrd.open_workbook(file_path, formatting_info=True)

        # Open the first sheet (assuming it's 'Sheet 1')
        sheet = workbook.sheet_by_index(0)

        # Find the row with the specified image name
        target_row_index = None
        for row_index in range(1, sheet.nrows):  # Start from the second row (headers are in the first row)
            if sheet.cell_value(row_index, 0) == image_name:
                target_row_index = row_index
                break

        if target_row_index is not None:
            # Create a new workbook for writing
            new_wb = Workbook()
            new_sheet = new_wb.add_sheet('Sheet 1')

            # List of header rows
            headers = sheet.row_values(0)

            # Write the headers to the new sheet
            for col, header in enumerate(headers):
                new_sheet.write(0, col, header)

            # Copy data from the existing file to the new file
            for row_index in range(1, sheet.nrows):
                if row_index == target_row_index:
                    # Update data for the target row and specified column
                    for col in range(sheet.ncols):
                        if col == headers.index("orientationAngle"):
                            # Update the specified column with new data (assuming 'detected_points_value' exists)
                            new_sheet.write(target_row_index, col, angle)  # Update detectedAt* column
                        else:
                            new_sheet.write(target_row_index, col, sheet.cell_value(row_index, col))
                else:
                    # Copy unchanged data for other rows
                    for col in range(sheet.ncols):
                        new_sheet.write(row_index, col, sheet.cell_value(row_index, col))

            # Save the new workbook with updated data to a temporary file
            temp_file_path = 'C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\temp_dummy_example_with_data.xls'
            new_wb.save(DUMMY_EXCEL_PATH)

            # Close the existing file
            workbook.release_resources()

            # Remove the original file
            os.remove(file_path)

            # Rename the temporary file to the original file's name
            os.rename(DUMMY_EXCEL_PATH, file_path)

        else:
            print(f"Record with imageName='{image_name}' not found in the Excel file.")
		

if __name__ == "__main__":
   
    cropped_images_folder_path = 'C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\detected'
    dft_on_cropped_images_folder_path = 'C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\dftOnBoofCv'
    detected_feather_points_folder_path = 'C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\featherPoints'

    generate_dft_images(cropped_images_folder_path, dft_on_cropped_images_folder_path)
    
    # Define a list of threshold values to use
    threshold_values = [100, 110, 120, 125, 130, 140, 150, 160, 170, 175, 180, 190, 200, 210, 220, 225, 230, 240, 250]  # Add your desired threshold values here
    detect_and_highlight_points(dft_on_cropped_images_folder_path, detected_feather_points_folder_path, threshold_values)
    update_excel_with_orientation_angle(cropped_images_folder_path, 'C:\\Deva\\PythonWorkspace\\sirsImages\\checking\\temp.xls')
