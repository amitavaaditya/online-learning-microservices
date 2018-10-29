import requests
import pandas as pd
import model_utils
from sklearn.model_selection import train_test_split

# API endpoint URL
URL = 'http://127.0.0.1:5000/predict'
# Test data filename
DATASET_NAME = 'validation.csv'

# Read test data into DataFrame
df = pd.read_csv('validation.csv', sep=';', decimal=',', na_values=['NA'])

# Split into training (for online training) and validation sets
online_train, online_test = train_test_split(df,
                                             test_size=0.1,
                                             random_state=0,
                                             stratify=df['classLabel'])

# Iterate over every row in training set (for online training) and request for
# predictions
for index in range(df.shape[0] * 9 // 10):
    data = online_train.iloc[index].to_json()
    response = requests.post(URL, json=data).json()
    print('Index {}'.format(index))
    print(response)

model_utils.evaluate(online_test)