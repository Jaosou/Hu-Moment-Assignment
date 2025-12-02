import numpy as np
import matplotlib.pyplot as plt
import cv2

x_bar = 0.0
y_bar = 0.0
mass = 0

gray = cv2.imread('image/object.jpg', cv2.IMREAD_GRAYSCALE)
rows, cols = gray.shape
bw = np.empty_like(gray)

threshold_start = int(input("Enter threshold start (0-255): "))
threshold_end = int(input("Enter threshold end (0-255): ")) 

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


cal_moments()
