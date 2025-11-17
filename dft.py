import numpy as np
import cv2
import matplotlib.pyplot as plt

size = 48
cutoff_frequency = 10

# โหลดภาพขาวดำ
gray = cv2.imread('image/dog.jfif', cv2.IMREAD_GRAYSCALE)
gray = cv2.resize(gray, (size, size))

rows,cols = gray.shape
print("Image dimensions:", rows, "x", cols)
F = np.zeros((rows, cols), dtype=complex)

def IDFT2D(F):
    M, N = F.shape
    img_reconstructed = np.zeros((M, N), dtype=complex)
    for x in range(M):
        for y in range(N):
            sum_val = 0
            for u in range(M):
                for v in range(N):
                    e = np.exp(2j * np.pi * ((u * x / M) + (v * y / N)))
                    sum_val += F[u, v] * e
            img_reconstructed[x, y] = sum_val / (M * N)
    return np.real(img_reconstructed)

def inverse_dft(freq_domain):
    return np.real(np.fft.ifft2(np.fft.ifftshift(freq_domain)))


def DFT2D(image,u,v):
    M, N = image.shape
    for x in range(M):
        for y in range(N):
            e = np.exp(-2j * np.pi * ((u * x / M) + (v * y / N)))
            F[u, v] += image[x, y] * e
    return F[u, v]

def compute_DFT(image):
    for u in range(rows):
        for v in range(cols):
            F[u, v] = DFT2D(image, u, v)
    return F

def LowPassFilter(image, cutoff):
    M, N = image.shape
    filtered = np.zeros((M, N), dtype=complex)
    center_u, center_v = M // 2, N // 2
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - center_u) ** 2 + (v - center_v) ** 2)
            if D <= cutoff:
                filtered[u, v] = image[u, v]
            else:
                filtered[u, v] = 0
    return filtered

def normalize(image):
    min_val = np.min(image)
    max_val = np.max(image)
    norm_image = np.zeros_like(image)
    for i in range(rows):
        for j in range(cols):
            norm_image[i, j] = (image[i, j] - min_val) * 255 / (max_val - min_val)
    return norm_image

dft_image = compute_DFT(gray)
abs_image = np.abs(dft_image)
shifted = np.fft.fftshift(abs_image)
log_image = np.log(shifted  + 1)   # ป้องกัน log(0)
norm_dft_image = normalize(log_image)

low_pass_filtered = LowPassFilter(norm_dft_image, cutoff_frequency)
reconstructed = inverse_dft(low_pass_filtered)
reconstructed = normalize(reconstructed)

print(norm_dft_image)

plt.figure(figsize=(16,4))

# แสดงภาพ
plt.subplot(1,3,1)
plt.title('Original Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')

# แสดง DFT
plt.subplot(1,3,2)
plt.title('DFT Magnitude')
plt.imshow(norm_dft_image, cmap='gray')
plt.axis('off')

# แสดง Phase
plt.subplot(1,3,3)
plt.title('DFT Phase')
plt.imshow(np.angle(dft_image), cmap='gray')
plt.axis('off')

# แสดง Low Pass Filtered
plt.figure(figsize=(8, 4))
plt.subplot(1,2,1)
plt.title('Low Pass Filtered')
plt.imshow(np.abs(low_pass_filtered), cmap='gray')
plt.axis('off')

# แสดง Reconstructed
plt.subplot(1,2,2)
plt.title('Reconstructed Image')
plt.imshow(reconstructed, cmap='gray')
plt.axis('off')

plt.show()