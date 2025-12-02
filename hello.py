import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

def conv2d_manual(image, kernel):
    # ขนาดภาพ
    h, w = image.shape
    
    # ขนาด kernel
    kh, kw = kernel.shape
    
    # ต้อง flip kernel สำหรับ convolution
    kernel_flipped = np.flipud(np.fliplr(kernel))
    
    # padding รอบภาพ
    pad_h = kh // 2
    pad_w = kw // 2
    
    # สร้างภาพ padding
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    
    # เตรียมที่เก็บผลลัพธ์
    result = np.zeros((h, w))
    
    # convolution ด้วยลูป
    for i in range(h):
        for j in range(w):
            # ตัดส่วนภาพที่ทับกับ kernel
            region = padded[i : i + kh, j : j + kw]
            
            # คูณ element-wise แล้วรวม
            result[i, j] = np.sum(region * kernel_flipped)
    
    return result

# กำหนดพิกัด x, y
x = np.arange(-5, 6)   # -5 ถึง 5
y = np.arange(-5, 6)

X, Y = np.meshgrid(x, y)

# พารามิเตอร์
sigma_x2 = 1
sigma_y2 = 1
x0 = 0
y0 = 0
A = 1

# ฟังก์ชัน Gaussian 2D
f = A * np.exp(
    - ( ((X - x0)**2) / (2 * sigma_x2) + ((Y - y0)**2) / (2 * sigma_y2) )
)
img_size = 128
img = cv2.imread('nature.jpg')
img_resized = cv2.resize(img, (img_size, img_size))

# เริ่ม plot 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, f, edgecolor='none')

# ตกแต่งกราฟ
ax.set_title("3D Gaussian Surface")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("f(x, y)")

plt.show()
