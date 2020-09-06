#  * @Author: abhinav.mazumdar
#  * @Date: 2020-09-02 23:08:21
#  * @Last Modified by:abhinav.mazumdar
#  * @Last Modified time: 2020-09-02 23:08:49


# This model classifies movie (IMDB Dataset)reviews as positive
# or negative ( binary classification)

from keras.datasets import imdb
from keras import models
from keras import layers
import numpy as np

# num words means we will only keep the top 10000 most frequently
# occuring words in the training data. Rare words will be discarded
# This will allow us to work with vector data of manageable size


def get_dataset():
    """
    Import IMDB data set
    """
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        num_words=10000
    )
    return (train_data, train_labels), (test_data, test_labels)


# One hot encoding
def vectorize_sequences(sequences, dimension=10000):
    # Create a all zero matrix of shape : ((len(sequences( number of samples/data point, dimesions)))
    results = np.zeros((len(sequences), dimension))
    # Iterate over the sequence and set specific indices of results[i] to 1
    for index, sequence in enumerate(sequences):
        results[index, sequence] = 1

    return results


def get_prepared_data(train_data, train_labels, test_data, test_labels):
    """[summary]

    Parameters
    ----------
    train_data : [type]
        [description]
    train_labels : [type]
        [description]
    test_data : [type]
        [description]
    test_labels : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    x_train = vectorize_sequences(train_data)
    x_test = vectorize_sequences(test_data)

    y_train = np.asarray(train_labels).astype("float32")
    y_test = np.asarray(test_labels).astype("float32")
    return (x_train, y_train, x_test, y_test)


def create_train_val_set(x_train, y_train):
    """[summary]

    Parameters
    ----------
    x_train : [type]
        [description]
    y_train : [type]
        [description]
    """
    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]
    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]
    return (x_val, partial_x_train, y_val, partial_y_train)


def create_model_definition(x_val, partial_x_train, y_val, partial_y_train):
    """[summary]"""
    print("X" * 1000)
    print(x_val)
    model = models.Sequential()
    model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["accuracy"])

    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
        verbose=2,
    )
    return history


if __name__ == "__main__":
    from numba import cuda

    cuda.select_device(0)
    cuda.close()

    (train_data, train_labels), (test_data, test_labels) = get_dataset()
    x_train, y_train, x_test, y_test = get_prepared_data(
        train_data, train_labels, test_data, test_labels
    )
    x_val, partial_x_train, y_val, partial_y_train = create_train_val_set(
        x_train, y_train
    )
    print("Y" * 100)
    print(x_val)
    history = create_model_definition(x_val, partial_x_train, y_val, partial_y_train)
