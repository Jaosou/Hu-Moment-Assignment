import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- Dataset path ----------------
dataset_path = "/root/.cache/kagglehub/datasets/khalidboussaroual/2d-geometric-shapes-17-shapes/versions/4/2D_Geometric_Shapes_Dataset"
classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
label_to_class = {i: cls for i, cls in enumerate(classes)}

# ---------------- Prepare data ----------------
X = []
y = []
images = []  # เก็บภาพดั้งเดิมเพื่อ plot

for label_idx, cls in enumerate(classes):
    cls_path = os.path.join(dataset_path, cls)
    files = os.listdir(cls_path)
    for f in tqdm(files, desc=f"Processing {cls}"):
        img_path = os.path.join(cls_path, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            continue
        cnt = max(contours, key=cv2.contourArea)

        moments = cv2.moments(cnt)
        huMoments = cv2.HuMoments(moments).flatten()
        huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)

        X.append(huMoments)
        y.append(label_idx)
        images.append(img)  # เก็บภาพดั้งเดิม

X = np.array(X)
y = np.array(y)

# ---------------- Train/Test split ----------------
X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X, y, images, test_size=0.3, random_state=42
)

# ---------------- ANN Classifier ----------------
clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
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
