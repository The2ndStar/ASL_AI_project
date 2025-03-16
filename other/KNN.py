import numpy as np
import pandas as pd
from statistics import mode

# Load training data for each sign
signs = ['A', 'B', 'C']
train_data = pd.concat([pd.read_csv(f'data_num/{sign}_joints.csv') for sign in signs], ignore_index=True)

# Convert numeric columns to float (excluding 'id' and 'label')
for col in train_data.columns:
    if col not in ['id', 'label']:
        train_data[col] = pd.to_numeric(train_data[col], errors='coerce')

# Define feature columns (all columns except 'id' and 'label')
feature_cols = [col for col in train_data.columns if col not in ['id', 'label']]

# Split data into training and testing sets
train_set = train_data[train_data['id'] <= 70]  # Training data
test_set = train_data[train_data['id'] > 70]    # Testing data

# Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1[feature_cols] - point2[feature_cols]) ** 2))

# KNN prediction function
def knn_predict(test_data, train_data, k=5):
    predictions = []
    for _, test_row in test_data.iterrows():
        distances = []
        for _, train_row in train_data.iterrows():
            dist = euclidean_distance(test_row, train_row)
            distances.append((dist, train_row['label']))
        
        # Sort distances and get k nearest neighbors
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:k]
        
        # Predict the label using majority vote
        neighbor_labels = [label for (_, label) in neighbors]
        predicted_label = mode(neighbor_labels)
        predictions.append(predicted_label)
    
    return predictions

# Make predictions
k = 5
predictions = knn_predict(test_set, train_set, k)

# Print predictions
print("Predictions:", predictions)