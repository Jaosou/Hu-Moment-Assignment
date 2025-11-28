import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

x_bar = 0.0
y_bar = 0.0
mass = 0

def cal_moments():
    global x_bar, y_bar, mass
    for i in range(rows):
        for j in range(cols):
            if gray[i][j] <= threshold_start or gray[i][j] >= threshold_end:
                bw[i][j] = 0
            else:
                bw[i][j] = 255
                mass += 1
                x_bar += j   # j คือแนวนอน (x)
                y_bar += i   # i คือแนวตั้ง (y)

    if mass > 0:
        x_bar /= mass
        y_bar /= mass

def cal_moments_raw(p,q):
    global x_bar, y_bar, mass
    mu_result = 0
    for i in range(rows):
        for j in range(cols):
            f = 1 if bw[i][j] > 0 else 0
            mu_result += (((j - x_bar)**p * (i - y_bar)**q) * f)

    return mu_result

def nomalize_central(p,q):
    global mass
    narmalize = cal_moments_raw(p,q) / (mass ** (1 + (p+q)/2))
    return narmalize

def cal_hu():
    m1 = nomalize_central(2,0) + nomalize_central(0,2)
    m2 = (nomalize_central(2,0) - nomalize_central(0,2))**2 + 4*(nomalize_central(1,1)**2)
    m3 = (nomalize_central(3,0) - 3*nomalize_central(1,2))**2 + (3*nomalize_central(2,1) - nomalize_central(0,3))**2
    m4 = (nomalize_central(3,0) + nomalize_central(1,2))**2 + (nomalize_central(2,1) + nomalize_central(0,3))**2
    m5 = (nomalize_central(3,0) - 3*nomalize_central(1,2)) * (nomalize_central(3,0) + nomalize_central(1,2)) * ((nomalize_central(3,0) + nomalize_central(1,2))**2 - 3*(nomalize_central(2,1) + nomalize_central(0,3))**2) + (3*nomalize_central(2,1) - nomalize_central(0,3)) * (nomalize_central(2,1) + nomalize_central(0,3)) * (3*(nomalize_central(3,0) + nomalize_central(1,2))**2 - (nomalize_central(2,1) + nomalize_central(0,3))**2)
    m6 = (nomalize_central(2,0) - nomalize_central(0,2)) * ((nomalize_central(3,0) + nomalize_central(1,2))**2 - (nomalize_central(2,1) + nomalize_central(0,3))**2) + 4*nomalize_central(1,1) * (nomalize_central(3,0) + nomalize_central(1,2)) * (nomalize_central(2,1) + nomalize_central(0,3))
    m7 = (3*nomalize_central(2,1) - nomalize_central(0,3)) * (nomalize_central(3,0) + nomalize_central(1,2)) * ((nomalize_central(3,0) + nomalize_central(1,2))**2 - 3*(nomalize_central(2,1) + nomalize_central(0,3))**2) - (nomalize_central(3,0) - 3*nomalize_central(1,2)) * (nomalize_central(2,1) + nomalize_central(0,3)) * (3*(nomalize_central(3,0) + nomalize_central(1,2))**2 - (nomalize_central(2,1) + nomalize_central(0,3))**2)

# ---------------- Dataset path ----------------
dataset_path = "/root/.cache/kagglehub/datasets/khalidboussaroual/2d-geometric-shapes-17-shapes/versions/4/2D_Geometric_Shapes_Dataset"
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
label_to_class = {i: cls for i, cls in enumerate(classes)}

plt.figure(figsize=(15, 8))

for idx, cls in enumerate(classes):
    cls_path = os.path.join(dataset_path, cls)
    files = os.listdir(cls_path)
    if len(files) == 0:
        continue

    img_path = os.path.join(cls_path, files[0])  # รูปแรกของแต่ละ class
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img is None:
        continue

    # หา Hu Moments
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        continue
    cnt = max(contours, key=cv2.contourArea)
    moments = cv2.moments(cnt)
    huMoments = cv2.HuMoments(moments).flatten()
    huMoments_log = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)

    # Plot
    plt.subplot(4, 5, idx+1)  # ปรับ grid ตามจำนวน class
    plt.imshow(thresh, cmap='gray')
    plt.title(f"{cls}\nHu: {np.round(huMoments_log, 2)}", fontsize=8)
    # plt.title(f"{cls}\nHu: {huMoments}", fontsize=8)
    plt.axis('off')

plt.tight_layout()
plt.show()
