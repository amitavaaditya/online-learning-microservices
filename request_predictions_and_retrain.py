import requests
import model_utils

# API endpoint URLs
PREDICT_URL = 'http://127.0.0.1:5000/predict'
RETRAIN_URL = 'http://127.0.0.1:5001/retrain'

# Test data filename
DATASET_NAME = 'validation.csv'


def test_predictions(df):
    """
    Method to test prediction API
    :param df: DataFrame to generate predictions for
    :return: None
    """
    json = df.to_json()
    response = requests.post(RETRAIN_URL, json=json).json()
    print(response)


def test_online_training(df):
    """
    Method to test online training API
    :param df: DataFrame for incremental training
    :return: None
    """
    for index in range(df.shape[0]):
        json = df.iloc[[index]].to_json()
        response = requests.post(RETRAIN_URL, json=json).json()
        print('Index {}'.format(index))
        print(response)


if __name__ == '__main__':
    df = model_utils.read_data(DATASET_NAME)
    # Due to tensorflow memory management related to allocating entire gpu
    # memory to single process, only one of the below methods can be tested
    # at a time. The other must be commented out.
    test_predictions(df)
    test_online_training(df)
