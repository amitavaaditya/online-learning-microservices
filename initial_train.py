import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf


# Constants needed
LABEL_COL = 'classLabel'
MODEL_DIR = 'model'


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
    df['classLabel'] = df['classLabel'].map({
        'no.': 0,
        'yes.': 1
    })
    return df


def split_data(df):
    """
    Method to split the DataFrame into training and validation sets
    :param df: preprocessed DataFrame
    :return: tuple of training and validation sets
    """
    return train_test_split(df, random_state=0, stratify=df[LABEL_COL])


def input_fn(df):
    """
    Input function generator required by Tensorflow Estimator API
    :param df: DataFrame to be used for training/evaluation/prediction
    :return: Input function for Tensorflow Estimator API
    """
    return tf.data.Dataset.from_tensor_slices((dict(df), df[LABEL_COL])) \
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


def build_model(feature_columns):
    """
    Method to build the classifier model
    :param feature_columns: List of Tensorflow Feature Columns
    :return: Tensorflow Estimator model
    """
    return tf.estimator.DNNClassifier(
        hidden_units=[4, 4],
        feature_columns=feature_columns,
        model_dir=MODEL_DIR,
        n_classes=2,
        optimizer=tf.train.AdamOptimizer(learning_rate=0.001))


def train(model, train_input_fn, epochs):
    """
    Method to invoke the training process
    :param model: Tensorflow Estimator model
    :param train_input_fn: Input function for Tensorflow Estimator API
    :param epochs: number of passes over the data
    :return: None
    """
    for i in range(epochs):
        print('Epoch {}'.format(i + 1))
        model.train(train_input_fn)


def evaluate(model, val_input_fn, val_df):
    """
    Method to invoke the training process
    :param model: Tensorflow Estimator model
    :param val_input_fn: Input function for Tensorflow Estimator API
    :param val_df: DataFrame for validation
    :return: None
    """
    y_pred = [pred['class_ids'][0] for pred in list(model.predict(
        val_input_fn))]
    y_true = val_df[LABEL_COL]
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    train_df, val_df = split_data(preprocess_data(read_data('train.csv')))
    train_input_fn = lambda: input_fn(train_df)
    val_input_fn = lambda: input_fn(val_df)
    feature_columns = get_feature_columns(train_df)
    model = build_model(feature_columns)
    train(model, train_input_fn, epochs=10)
    evaluate(model, val_input_fn, val_df)
