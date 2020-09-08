# /*
#  * @Author: abhinav.mazumdar
#  * @Date: 2020-09-07 00:06:53
#  * @Last Modified by:   abhinav.mazumdar
#  * @Last Modified time: 2020-09-07 00:06:53
#  */

import numpy as np
from keras import models, layers
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical


def get_dataset():
    """
    Import IMDB data set
    """
    (train_data, train_labels), (test_data, test_labels) = reuters.load_data(
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


# Dimensions is 46 as there are 46 class labels / output class for classification
def to_one_hot_labels(labels, dimensions=46):
    """[summary]

    Parameters
    ----------
    labels : [type]
        [description]
    dimensions : int, optional
        [description], by default 46

    Returns
    -------
    [type]
        [description]
    """
    results = np.zeros(len(labels), dimensions)
    for i, label in enumerate(labels):
        results[i, label] = 1
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

    # one_hot_train_labels = to_one_hot_labels(train_labels)
    # one_hot_test_labels = to_one_hot_labels(test_labels)
    # Instead of using our own custom one hot  (to_one_hot_labels), we can use the built in keras function to do it
    one_hot_train_labels = to_categorical(train_labels)
    one_hot_test_labels = to_categorical(test_labels)
    return (x_train, one_hot_train_labels, x_test, one_hot_test_labels)


def create_model_definition(x_val, partial_x_train, y_val, partial_y_train):
    """[summary]"""
    model = models.Sequential()

    model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(46, activation="softmax"))

    model.compile(
        optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    history = model.fit(
        partial_x_train,
        partial_y_train,
        epochs=20,
        batch_size=512,
        validation_data=(x_val, y_val),
    )

    return history


def create_train_val_set(x_train, one_hot_train_labels):
    """[summary]

    Parameters
    ----------
    x_train : [type]
        [description]
    y_train : [type]
        [description]
    """
    x_val = x_train[:1000]
    partial_x_train = x_train[1000:]

    y_val = one_hot_train_labels[:1000]
    partial_y_train = one_hot_train_labels[1000:]
    return (x_val, partial_x_train, y_val, partial_y_train)


def plot_loss(history, epochs, plt):
    history_dict = history.history
    loss_values = history_dict.get("loss")
    val_loss_values = history_dict.get("val_loss")

    epochs_to_plot = range(1, epochs + 1)
    plt.plot(epochs_to_plot, loss_values, "bo", label="Training Loss")
    plt.plot(epochs_to_plot, val_loss_values, "b", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()


def plot_accuracy(history, epochs, plt):
    history_dict = history.history
    acc_values = history_dict.get("accuracy")
    val_acc_values = history_dict.get("val_accuracy")

    epochs_to_plot = range(1, epochs + 1)
    plt.plot(epochs_to_plot, acc_values, "bo", label="Training Accuracy")
    plt.plot(epochs_to_plot, val_acc_values, "b", label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()


if __name__ == "__main__":
    from numba import cuda

    cuda.select_device(0)
    cuda.close()

    (train_data, train_labels), (test_data, test_labels) = get_dataset()
    x_train, one_hot_train_labels, x_test, one_hot_test_labels = get_prepared_data(
        train_data, train_labels, test_data, test_labels
    )
    x_val, partial_x_train, y_val, partial_y_train = create_train_val_set(
        x_train, one_hot_train_labels
    )
    history = create_model_definition(x_val, partial_x_train, y_val, partial_y_train)
    # # plot_loss(history, 20, plt)
    # # plot_accuracy(history, 20, plt)
