import os
import pandas as pd
from sklearn.metrics import classification_report
import tensorflow as tf


LABEL_COL = 'classLabel'
MODEL_DIR = 'model'
MODEL_NAME = 'model.ckpt'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def read_data(filename):
    """
    Method to read the dataset as a pandas DataFrame
    :param filename: filename of the dataset
    :return: DataFrame of the dataset
    """
    return pd.read_csv(filename, sep=';', decimal=',', na_values=['NA'])


def preprocess_data(df):
    """
    Method responsible to perform basic preprocessing required prior to
    training/evaluation/prediction
    :param df: unprocessed DataFrame
    :return: DataFrame after preprocessing
    """
    df['v19'] = df['v19'].apply(str)
    median_values = {
        'v2': 28.67,
        'v3': 0.000425,
        'v8': 1.75,
        'v11': 2.0,
        'v14': 120.0,
        'v15': 113.0,
        'v17': 1200000.0
    }
    iqr_values = {
        'v2': 17.83,
        'v3': 0.0008125000000000001,
        'v8': 4.5,
        'v11': 6.0,
        'v14': 280.0,
        'v15': 1059.75,
        'v17': 2800000.0
    }
    categories_dict = {'v1': ['a', 'missing', 'b'],
                       'v4': ['y', 'u', 'missing', 'l'],
                       'v5': ['p', 'g', 'gg', 'missing'],
                       'v6': ['k', 'x', 'ff', 'cc', 'r', 'j', 'm', 'W', 'aa',
                              'missing', 'q', 'd', 'e', 'i', 'c'],
                       'v7': ['ff', 'j', 'h', 'n', 'missing', 'dd', 'v', 'bb',
                              'z', 'o'], 'v9': ['f', 'missing', 't'],
                       'v10': ['f', 'missing', 't'],
                       'v12': ['f', 'missing', 't'],
                       'v13': ['p', 'g', 'missing', 's'],
                       'v18': ['missing', 'f', 't'],
                       'v19': ['missing', '0', '1'],
                       'classLabel': ['no.', 'missing', 'yes.']}
    df = df.fillna(median_values)
    for column in median_values:
        df[column] = (df[column] - median_values[column]) / iqr_values[column]
    for column in categories_dict:
        df[column].fillna('missing', inplace=True)
        df[column] = pd.Categorical(df[column], ordered=False,
                                    categories=categories_dict[column])
    if LABEL_COL in df:
        df[LABEL_COL] = df[LABEL_COL].map({
            'no.': 0,
            'yes.': 1
        })
    return df


def load_and_prepare_dataset(filename):
    """
    Method to split the DataFrame into training and validation sets
    :param df: preprocessed DataFrame
    :return: tuple of training and validation sets
    """
    return preprocess_data(read_data(filename))


def input_fn(df, label_col=LABEL_COL):
    """
    Input function generator required by Tensorflow Estimator API
    :param df: DataFrame to be used for training/evaluation/prediction
    :return: Input function for Tensorflow Estimator API
    """
    return tf.data.Dataset.from_tensor_slices((dict(df), df[label_col])) \
        .batch(32)


def get_feature_columns(df):
    """
    Method to generate the list of Feature Columns required by Tensorflow
    Estimator API
    :param df: Dataset to fetch columns from
    :return: List of Tensorflow Feature Columns
    """
    numeric_columns = [tf.feature_column.numeric_column(column) for column
                       in ('v2', 'v3', 'v8', 'v11', 'v14', 'v15', 'v17')]
    categorical_columns = [
        tf.feature_column.indicator_column(
            tf.feature_column.categorical_column_with_vocabulary_list(
                column, vocabulary_list=df[column].cat.categories))
        for column in ('v1', 'v4', 'v5', 'v6', 'v7', 'v9', 'v10', 'v12',
                       'v13', 'v18', 'v19')
    ]
    return numeric_columns + categorical_columns


def rebuild_model_with_df(df):
    """
    Method to rebuild the classifier model using existing checkpoints.
    The model configuration needs to match the one saved in MODEL_DIR
    :param feature_columns: List of Tensorflow Feature Columns
    :return: DataFrame to read feature columns from
    """
    feature_columns = get_feature_columns(df)
    return tf.estimator.DNNClassifier(
        hidden_units=[4, 4],
        feature_columns=feature_columns,
        model_dir=MODEL_DIR,
        n_classes=2,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))


def evaluate(df):
    """
    Method to evaluate on the help out validation set
    :param df: DataFrame for validation
    :return: None
    """
    df = preprocess_data(df)
    model = rebuild_model_with_df(df)
    test_input_fn = lambda: input_fn(df)
    y_pred = [pred['class_ids'][0] for pred in list(model.predict(
        test_input_fn))]
    y_true = df[LABEL_COL]
    print(classification_report(y_true, y_pred))


def predict(df):
    """
    Method to generate predictions on new data
    :param df: DataFrame to generate predictions for
    :return: dictionary containing class id, class label and probabilities
    """
    df = preprocess_data(df)
    model = rebuild_model_with_df(df)
    predictions = list(model.predict(lambda: input_fn(df)))
    class_labels = {
        0: 'no.',
        1: 'yes.'
    }
    return [{
        'index': index,
        'probabilities': dict(zip(class_labels.values(),
                                  prediction['probabilities']
                                  .tolist())),
        'class_id': int(prediction['class_ids'][0]),
        'class_label': class_labels[prediction['class_ids'][0]]
    } for index, prediction in enumerate(predictions)]


def incremental_train(df):
    """
    Method for incrementally training the existing model on new data
    :param df: New data to train the model on
    :return: None
    """
    df = preprocess_data(df)
    model = rebuild_model_with_df(df)
    model.train(lambda: input_fn(df))


if __name__ == '__main__':
    test_df = load_and_prepare_dataset('validation.csv')
    evaluate(test_df)
