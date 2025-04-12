import cv2 as cv
import os
import numpy as np

# Define input and output folder paths
input_folders = ["Dataset/Hex", "Dataset/Machine", "Dataset/Philips", "Dataset/Torx"]
output_folders = ["Processed/Hex", "Processed/Machine", "Processed/Philips", "Processed/Torx"]
cropped_folders = ["Cropped/Hex", "Cropped/Machine", "Cropped/Philips", "Cropped/Torx"]

# Create output folders if they don't exist
for folder in output_folders + cropped_folders:
    os.makedirs(folder, exist_ok=True)

# Define a kernel for dilation
kernel = np.ones((7,7), np.uint8)  # Adjust size if needed

# Process each dataset
for input_folder, output_folder, cropped_folder in zip(input_folders, output_folders, cropped_folders):
    for filename in os.listdir(input_folder):
        if filename.endswith((".jpg")):
            # Read the image
            img_path = os.path.join(input_folder, filename)
            img = cv.imread(img_path)

            # Convert to grayscale
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            # Apply thresholding
            _, thresh = cv.threshold(gray, 80, 255, cv.THRESH_BINARY_INV)  # Adjust threshold value if needed

            # Apply dilation
            dilated = cv.dilate(thresh, kernel, iterations=2)  # Adjust iterations for stronger effect

            # Save the processed image
            output_path = os.path.join(output_folder, filename)
            cv.imwrite(output_path, dilated)

            # Find contours in the mask
            contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            # Loop through each detected object and crop it from the original image
            for i, contour in enumerate(contours):
                x, y, w, h = cv.boundingRect(contour)  # Get bounding box
                cropped = img[y:y+h, x:x+w]  # Crop from original image

                # Save cropped image
                cropped_filename = f"{filename.split('.')[0]}_crop_{i}.png"
                cropped_path = os.path.join(cropped_folder, cropped_filename)
                cv.imwrite(cropped_path, cropped)

