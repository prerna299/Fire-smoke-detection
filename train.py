import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# Store image data
data = []

# Store labels
labels = []

# Categories
categories = ['fire', 'smoke']

print("Loading images...")

for category in categories:

    path = os.path.join('dataset', category)

    label = categories.index(category)

    for img in os.listdir(path):

        img_path = os.path.join(path, img)

        try:
            image = cv2.imread(img_path)

            image = cv2.resize(image, (64, 64))

            data.append(image.flatten())

            labels.append(label)

        except:
            pass

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

print("Dataset loaded successfully!")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training models...")

# KNN Model
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)

knn_acc = accuracy_score(y_test, knn_pred)

# Decision Tree Model
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

# Random Forest Model
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

# Results
print("\nRESULTS")

print("KNN Accuracy:", knn_acc)

print("Decision Tree Accuracy:", dt_acc)

print("Random Forest Accuracy:", rf_acc)

# Select best model
best_model = rf

# Save model
joblib.dump(best_model, 'fire_smoke_detector.pkl')

print("\nModel saved successfully!")