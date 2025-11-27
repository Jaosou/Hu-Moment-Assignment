#Hough Transform

import numpy as np
import cv2
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO
# Image URL
# url = "https://upload.wikimedia.org/wikipedia/commons/7/7d/Lenna_%28test_image%29.png"
image = cv2.imread('image/object.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Perform Hough Transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
# Display the result
plt.imshow(image)
plt.title('Hough Transform Line Detection')
plt.show()