import cv2
import numpy as np
import pandas as pd
from statistics import mode
from handtracking import HandDetector 

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def knn_predict(test_point, train_data, k=5):
    distances = []
    for _, train_row in train_data.iterrows():
        dist = euclidean_distance(test_point, train_row[feature_cols])
        distances.append((dist, train_row['label']))
    
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
    neighbor_labels = [label for (_, label) in neighbors]
    predicted_label = mode(neighbor_labels)
    return predicted_label


signs = ['A', 'B', 'C','D','E']  
train_data = pd.concat([pd.read_csv(f'data_num/{sign}_joints.csv') for sign in signs], ignore_index=True)


for col in train_data.columns:
    if col not in ['id', 'label']:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')


feature_cols = [col for col in train_data.columns if col not in ['id', 'label']]


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, complexity=0)  

while True:
    success, img = cap.read()
    if not success:
        break

    img, bboxes = detector.findHands(img)
    lmList = []  

    if bboxes:

        lmList = detector.findPosition(img, draw=False)

        if lmList:
            test_point = []
            for joint in lmList:
                test_point.extend(joint[1:])  


            if len(test_point) == len(feature_cols):

                predicted_label = knn_predict(np.array(test_point), train_data, k=5)
                print(f"Predicted Label: {predicted_label}")

                cv2.putText(img, f"Sign: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator", img)
    if cv2.waitKey(1) == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()