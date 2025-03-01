import numpy as np
import pandas as pd
from statistics import mode


data_files = {
    'A': r'data_num\A_joints.csv',
    'B': r'data_num\B_joints.csv',
    'C': r'data_num\C_joints.csv',
    'D': r'data_num\D_joints.csv',
}


datasets = {letter: pd.read_csv(file) for letter, file in data_files.items()}

for letter, df in datasets.items():
    df['label'] = letter


train = pd.concat(datasets.values(), axis=0, ignore_index=True)

for col in train.columns:
    if col not in ['id', 'label']:  # Avoid label and id
        train[col] = pd.to_numeric(train[col], errors='coerce')


train = train.dropna()

feature_cols = [col for col in train.columns if col not in ['id', 'label']]

train_data = train[train['id'] <= 70]  
test_data = train[train['id'] > 70]    


def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1.iloc[i] - point2.iloc[i]) ** 2
    return np.sqrt(distance)

def knn_predict(data, train_data, k):
    predictions = []
    for _, row in data.iterrows():
        distances = []
        for _, train_row in train_data.iterrows():
            dist = euclidean_distance(row[feature_cols], train_row[feature_cols])
            distances.append((dist, train_row['label']))
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        neighbor_labels = [label for (_, label) in neighbors]
        predicted_label = mode(neighbor_labels)
        predictions.append(predicted_label)
    return predictions


k = 9

train_predictions = knn_predict(train_data, train_data, k)
train_data['predicted_label'] = train_predictions

test_predictions = knn_predict(test_data, train_data, k)
test_data['predicted_label'] = test_predictions

train_accuracy = (train_data['predicted_label'] == train_data['label']).mean()
test_accuracy = (test_data['predicted_label'] == test_data['label']).mean()
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

def create_confusion_matrix(true_labels, predicted_labels, labels):
    num_classes = len(labels)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        true_index = label_to_index[true_label]
        predicted_index = label_to_index[predicted_label]
        cm[true_index][predicted_index] += 1
    
    return cm

labels = sorted(train_data['label'].unique())

train_cm = create_confusion_matrix(train_data['label'], train_data['predicted_label'], labels)
test_cm = create_confusion_matrix(test_data['label'], test_data['predicted_label'], labels)

print("\nTraining Confusion Matrix:")
print("Rows: Actual Labels, Columns: Predicted Labels")
print(pd.DataFrame(train_cm, index=labels, columns=labels).to_string())

print("\nTesting Confusion Matrix:")
print("Rows: Actual Labels, Columns: Predicted Labels")
print(pd.DataFrame(test_cm, index=labels, columns=labels).to_string())