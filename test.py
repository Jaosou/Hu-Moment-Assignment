import cv2
import numpy as np

# สร้างภาพ binary (วงกลม)
img = np.zeros((200, 200), dtype=np.uint8)
cv2.circle(img, (100, 100), 40, 255, -1)

def show_moments(binary_img, label="original"):
    M = cv2.moments(binary_img)
    hu = cv2.HuMoments(M).flatten()
    print(f"\n--- {label} ---")
    print("Raw M00 (area):", M['m00'])
    print("Centroid (cx, cy):", (M['m10']/M['m00'], M['m01']/M['m00']))
    print("Central mu20:", M['mu20'])
    print("Central mu02:", M['mu02'])
    print("Hu moments:", np.round(hu, 6))

# original
show_moments(img, "Original")

# หมุน 45 องศา
M_rot = cv2.getRotationMatrix2D((100,100), 45, 1)
rotated = cv2.warpAffine(img, M_rot, (200,200))

show_moments(rotated, "Rotated 45°")

# หมุน 90 องศา
M_rot90 = cv2.getRotationMatrix2D((100,100), 90, 1)
rotated90 = cv2.warpAffine(img, M_rot90, (200,200))

show_moments(rotated90, "Rotated 90°")
