import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

# Image URL
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRmKTFI_7r2TKR0sknfK_7GJX8sHItxbf-zh_jOJIBde2-L69K29IAzFLrD&s"

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            norm_image[i, j] = (image[i, j] - min_val) * 255 / (max_val - min_val)
    return norm_image

# Fetch the image from the URL
response = requests.get(url)
img_data = response.content

# Open the image using PIL
img = Image.open(BytesIO(img_data))

# Convert the image to grayscale
image = np.array(img.convert('L'))  # Convert to grayscale

# Compute FFT
f = np.fft.fft2(image)
fshift = np.fft.fftshift(f)

# Create a mask with a low-pass filter
rows, cols = image.shape
crow, ccol = rows // 2, cols // 2
mask = np.zeros((rows, cols), np.uint8)
mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # Creating a simple low-pass filter mask

# Apply mask and inverse FFT
fshift_masked = fshift * mask
f_ishift = np.fft.ifftshift(fshift_masked)
image_filtered = np.fft.ifft2(f_ishift)
image_filtered = np.abs(image_filtered)

log_image = np.log(fshift_masked  + 1) 
norm_dft_image = normalize(log_image)

# Display the filtered image
plt.imshow(image_filtered, cmap='gray')
plt.title('Low-Pass Filtered Image')
plt.show()