import cv2
import matplotlib.pyplot as plt
import numpy as np

# โหลดภาพแบบขาวดำ
img = cv2.imread('image/road.webp', cv2.IMREAD_GRAYSCALE)
img_size = 256
img = cv2.resize(img, (img_size, img_size))

# Canny edge detection
edges = cv2.Canny(img, 100, 200)

h ,w = edges.shape
diag_len = int(np.sqrt(h**2 + w**2))

rhos = np.arange(-diag_len, diag_len + 1)
thetas = np.deg2rad(np.arange(0, 180))

accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)

ys, xs = np.nonzero(edges)

for x, y in zip(xs, ys): 
    for t_idx, theta in enumerate(thetas): 
        rho = int(x * np.cos(theta) + y * np.sin(theta)) 
        rho_idx = rho + diag_len 
        accumulator[rho_idx, t_idx] += 1
        

def seach_peak():
    N = 20   # จำนวน peak ที่ต้องการ

    # flatten แล้วหา index ของค่ามากที่สุด N ตัว
    flat = accumulator.ravel()
    idxs = np.argpartition(flat, -N)[-N:]      # หาตัวใหญ่สุด N ตัว (unordered)

    # จัดเรียงตามค่าจริง (ใหญ่ → เล็ก)
    idxs = idxs[np.argsort(flat[idxs])[::-1]]

    peaks = []
    for idx in idxs:
        rho_i, theta_i = np.unravel_index(idx, accumulator.shape)
        rho_val = rhos[rho_i]
        theta_val = thetas[theta_i]
        peaks.append((rho_val, theta_val))
        
        
    for peak in peaks:
        threshold = 10
        rho_val, theta_val = peak
        for rho_target, theta_target in peaks:
            if abs(rho_val-rho_target) < threshold:
                peaks.remove((rho_target, theta_target))

    return peaks


def convert_rho_theta_to_line(rho, theta):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    return (x1, y1, x2, y2)

peaks = seach_peak()
print(peaks)

for rho, theta in peaks:
    x1, y1, x2, y2 = convert_rho_theta_to_line(rho, theta)
    print(rho, np.rad2deg(theta), "->", (x1, y1, x2, y2))
    cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

# peak_idx = np.unravel_index(accumulator.argmax(), accumulator.shape) 
# rho_peak = rhos[peak_idx[0]] 
# theta_peak = thetas[peak_idx[1]]

# print(peak_idx)


# a = np.cos(theta_peak)
# b = np.sin(theta_peak)
# x0 = a * rho_peak
# y0 = b * rho_peak
# x1 = int(x0 + 1000 * (-b))
# y1 = int(y0 + 1000 * (a))
# x2 = int(x0 - 1000 * (-b))
# y2 = int(y0 - 1000 * (a))

# cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

# print("Detected line: rho =", rho_peak, ", theta =", np.rad2deg(theta_peak))


# แสดงผล
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Canny Edges")
plt.imshow(edges, cmap='gray')

plt.figure(figsize=(10,6)) 
plt.imshow(accumulator, cmap='hot', extent=[0,180, rhos[-1], rhos[0]]) 
plt.title("Hough Space (Accumulator)") 
plt.xlabel("Theta (degrees)") 
plt.ylabel("Rho") 
plt.colorbar() 
plt.show()
