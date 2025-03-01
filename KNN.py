import numpy as np
import pandas as pd
from statistics import mode


Atrain = pd.read_csv(r'data_num\A_joints.csv')
Btrain = pd.read_csv(r'data_num\B_joints.csv')
Ctrain = pd.read_csv(r'data_num\C_joints.csv')
Dtrain = pd.read_csv(r'data_num\D_joints.csv')



train = pd.concat([Atrain, Btrain, Ctrain,Dtrain], axis=0, ignore_index=True)


for col in train.columns:
    if col not in ['id', 'label']:  # Avoid label and id
        train[col] = pd.to_numeric(train[col], errors='coerce')


feature_cols = [col for col in train.columns if col not in ['id', 'label']]


train_data = train[train['id'] <= 70]  
test_data = train[train['id'] > 70]    

def euclidean_distance(point1, point2):
    distance = 0
    for i in range(len(point1)):
        distance += (point1.iloc[i] - point2.iloc[i]) ** 2
    return np.sqrt(distance) 


k = 5
predictions = []

for _, test_row in test_data.iterrows():
    distances = []
    
    
    for _, train_row in train_data.iterrows():
        dist = euclidean_distance(test_row[feature_cols], train_row[feature_cols])
        distances.append((dist, train_row['label']))
    

    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]
    
 
    neighbor_labels = [label for (_, label) in neighbors]
    
  
    predicted_label = mode(neighbor_labels)
    predictions.append(predicted_label)


test_data['predicted_label'] = predictions


accuracy = (test_data['predicted_label'] == test_data['label']).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")


all_results = pd.concat([ test_data], axis=0)

print("\nAll Results:")
print(all_results[['id', 'label', 'predicted_label']].to_string(index=False))