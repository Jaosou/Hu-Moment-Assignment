import numpy as np
import matplotlib.pyplot as plt
import cv2

# --------------------------
# ฟังก์ชัน Convolution แบบเขียนเอง
# --------------------------
def conv2d_manual(image, kernel):
    h, w = image.shape
    kh, kw = kernel.shape
    
    # flip kernel (ตามนิยาม convolution)
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    # padding
    pad_h = kh // 2
    pad_w = kw // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # เตรียมผลลัพธ์
    result = np.zeros((h, w))
    
    # convolution ด้วย loop
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel_flipped)
    
    return result


x = np.arange(-5, 6)
y = np.arange(-5, 6)
X, Y = np.meshgrid(x, y)

sigma_x2 = 5
sigma_y2 = 5
A = 1

kernel = A * np.exp(
    - ( (X**2)/(2*sigma_x2) + (Y**2)/(2*sigma_y2) )
)

image = cv2.imread('image/nature.jpg', cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (128, 128))

conv_img = conv2d_manual(image, kernel)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("After Convolution")
plt.imshow(conv_img, cmap='gray')
plt.colorbar()

plt.show()
