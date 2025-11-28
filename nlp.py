import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- Dataset path ----------------
dataset_path = "/root/.cache/kagglehub/datasets/khalidboussaroual/2d-geometric-shapes-17-shapes/versions/4/2D_Geometric_Shapes_Dataset"
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
label_to_class = {i: cls for i, cls in enumerate(classes)}
print("Classes found:", label_to_class)

# ---------------- Prepare data ----------------
X = []
y = []
images = []  # เก็บภาพดั้งเดิมเพื่อ plot

MAX_FILES_PER_CLASS = 15000

for label_idx, cls in enumerate(classes):
    cls_path = os.path.join(dataset_path, cls)
    files = os.listdir(cls_path)

    # shuffle files เพื่อเลือกแบบสุ่ม
    np.random.shuffle(files)
    
    # จำกัดจำนวนไฟล์ต่อคลาส
    files = files[:MAX_FILES_PER_CLASS]

    for f in tqdm(files, desc=f"Processing {cls}"):
        img_path = os.path.join(cls_path, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Threshold และหา contours
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        thresh = cv2.bitwise_not(thresh)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cnt = max(contours, key=cv2.contourArea)

        # Hu Moments
        moments = cv2.moments(cnt)
        huMoments = cv2.HuMoments(moments).flatten()
        huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)

        X.append(huMoments)
        y.append(label_idx)
        images.append(img)

X = np.array(X)
y = np.array(y)

# ---------------- Scale Hu Moments ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- Train/Test split (stratified) ----------------
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X_scaled, y, images, test_size=0.3, random_state=42, stratify=y
)

# ---------------- RandomForest Classifier ----------------
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = clf.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# ---------------- Plot 10 รูป พร้อม class ----------------
plt.figure(figsize=(15,6))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(img_test[i], cmap='gray')
    plt.title(f"Pred: {label_to_class[y_pred[i]]}\nTrue: {label_to_class[y_test[i]]}")
    plt.axis('off')
plt.tight_layout()
plt.show()
