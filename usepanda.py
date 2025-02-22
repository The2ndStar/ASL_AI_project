import pandas as pd
df = pd.read_csv(r'C:\Desktop\AI Project\ASL_project\data_num\A_joints.csv')
print(df) 
print(df.shape)  
print(df.info()) 

print(df.head(10))
pd.set_option('display.max_columns',44)
pd.set_option('display.max_rows',100)
schema_df = pd.read_csv(r'C:\Desktop\AI Project\ASL_project\data_num\A_joints.csv')
print(schema_df)