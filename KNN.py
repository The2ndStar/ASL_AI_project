import numpy as np
import pandas as pd
from statistics import mode



Atrain = pd.read_csv(r'data_num\A_joints.csv')
Btrain = pd.read_csv(r'data_num\B_joints.csv')
Ctrain = pd.read_csv(r'data_num\C_joints.csv')
Dtrain = pd.read_csv(r'data_num\D_joints.csv')
Etrain = pd.read_csv(r'data_num\E_joints.csv')
Ftrain = pd.read_csv(r'data_num\F_joints.csv')
Gtrain = pd.read_csv(r'data_num\G_joints.csv')
Htrain = pd.read_csv(r'data_num\H_joints.csv')
Itrain = pd.read_csv(r'data_num\I_joints.csv')
Jtrain = pd.read_csv(r'data_num\J_joints.csv')
Ktrain = pd.read_csv(r'data_num\K_joints.csv')
Ltrain = pd.read_csv(r'data_num\L_joints.csv')
Mtrain = pd.read_csv(r'data_num\M_joints.csv')
Ntrain = pd.read_csv(r'data_num\N_joints.csv')
Otrain = pd.read_csv(r'data_num\O_joints.csv')
Ptrain = pd.read_csv(r'data_num\P_joints.csv')
Qtrain = pd.read_csv(r'data_num\Q_joints.csv')
Rtrain = pd.read_csv(r'data_num\R_joints.csv')
Strain = pd.read_csv(r'data_num\S_joints.csv')
Ttrain = pd.read_csv(r'data_num\T_joints.csv')
Utrain = pd.read_csv(r'data_num\U_joints.csv')
Vtrain = pd.read_csv(r'data_num\V_joints.csv')
Qtrain = pd.read_csv(r'data_num\Q_joints.csv')
Wtrain = pd.read_csv(r'data_num\W_joints.csv')
Xtrain = pd.read_csv(r'data_num\X_joints.csv')
Ytrain = pd.read_csv(r'data_num\Y_joints.csv')
Ztrain = pd.read_csv(r'data_num\Z_joints.csv')

train = pd.concat([Atrain, Btrain, Ctrain,Dtrain,Etrain,Ftrain,Gtrain,Htrain], axis=0, ignore_index=True)


for col in train.columns:
    if col not in ['id', 'label']: 
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
distances_list = []

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





accuracy = (test_data['predicted_label'] == test_data['label']).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")


all_results = pd.concat([ test_data], axis=0)

print("\nAll Results:")
print(all_results[['id', 'label', 'predicted_label']].to_string(index=False))
