import os
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

data = []
labels = []

classes = {
    "fire":0,
    "normal":1,
    "smoke":2
}

print("Loading Images...")

for folder in classes:

    path = os.path.join("dataset", folder)

    for img_name in os.listdir(path):

        try:

            img_path = os.path.join(path, img_name)

            img = cv2.imread(img_path)

            img = cv2.resize(img,(64,64))

            img = img.flatten()

            data.append(img)

            labels.append(classes[folder])

        except:
            pass

X = np.array(data)
y = np.array(labels)

print("Dataset Loaded")

X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ------------------
# KNN
# ------------------

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)

knn_pred = knn.predict(X_test)

print("\nKNN")

print("Accuracy:",
      accuracy_score(y_test,knn_pred))

print("Precision:",
      precision_score(y_test,knn_pred,
                      average="weighted"))

print("Recall:",
      recall_score(y_test,knn_pred,
                   average="weighted"))

# ------------------
# Decision Tree
# ------------------

dt = DecisionTreeClassifier()

dt.fit(X_train,y_train)

dt_pred = dt.predict(X_test)

print("\nDecision Tree")

print("Accuracy:",
      accuracy_score(y_test,dt_pred))

print("Precision:",
      precision_score(y_test,dt_pred,
                      average="weighted"))

print("Recall:",
      recall_score(y_test,dt_pred,
                   average="weighted"))

# ------------------
# Random Forest
# ------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train,y_train)

import joblib

joblib.dump(rf, "fire_smoke_detector.pkl")

print("Random Forest Model Saved Successfully!")

rf_pred = rf.predict(X_test)

print("\nRandom Forest")

print("Accuracy:",
      accuracy_score(y_test,rf_pred))

print("Precision:",
      precision_score(y_test,rf_pred,
                      average="weighted"))

print("Recall:",
      recall_score(y_test,rf_pred,
                   average="weighted"))