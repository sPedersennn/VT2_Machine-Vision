
import cv2 as cv
import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision import transforms

# Assume these images are from your processing pipeline
original = cv.imread("Raw\Philips\image_168.jpg")
gray = cv.cvtColor(original, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv.threshold(blurred, 50, 255, cv.THRESH_BINARY_INV)
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
dilated = cv.dilate(thresh, kernel, iterations=2)

# Step 1: Find center of white pixels in the threshold image
coords = cv.findNonZero(dilated)  # Returns Nx1x2 array of points
mean = cv.mean(coords)  # (x, y, _) with last element unused
center_x, center_y = int(mean[0]), int(mean[1])

# Step 2: Define crop dimensions (adjust as needed)
crop_width, crop_height = 256, 256
x1 = max(center_x - crop_width // 2, 0)
y1 = max(center_y - crop_height // 2, 0)
x2 = min(x1 + crop_width, original.shape[1])
y2 = min(y1 + crop_height, original.shape[0])

# Step 3: Crop from the original image
cropped = original[y2 - crop_height:y2, x2 - crop_width:x2]
cropped_resized = cv.resize(cropped, (original.shape[1], original.shape[0]))

# Convert all images to 3-channel format (so make_grid works)
def ensure_3channel(img):
    if len(img.shape) == 2:  # Grayscale
        return cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    else:  # Already 3-channel BGR
        return img

# List of images in OpenCV BGR format
images_bgr = [
    original,
    ensure_3channel(gray),
    ensure_3channel(blurred),
    ensure_3channel(thresh),
    ensure_3channel(dilated),
    ensure_3channel(cropped_resized)
]

# Convert BGR to RGB, then to tensor [C, H, W] and normalize to [0, 1]
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1]
])

images_tensor = [transform(cv.cvtColor(img, cv.COLOR_BGR2RGB)) for img in images_bgr]

# Stack into a grid
grid = make_grid(images_tensor, nrow=3, padding=5)

# Plot with matplotlib
plt.figure(figsize=(12, 6))
plt.imshow(grid.permute(1, 2, 0))  # Convert [C, H, W] to [H, W, C] for plt
plt.axis('off')
plt.title("Image Processing Stages")
plt.show()