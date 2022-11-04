import csv
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from settings import *
from preprocessing import *



def weighted_multilabel_loss(pos_weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        return tf.nn.weighted_cross_entropy_with_logits(
            labels, logits, pos_weight
        )
    return weighted_cross_entropy_with_logits


def newmodel(summary=True):
    """
    Creates model
    Input dims (batch, vocab size, input time length)
    Ouptut dims (batch, vocab size, output time length)
    
    Embedding -> LSTM(dropout) -> LSTM -> Dense(dropout) -> Output
    """
    model = Sequential()
    model.add(layers.Embedding(
        input_dim=HYPERPARAMS["vocab size"],
        output_dim=HYPERPARAMS["embedding size"],
        input_length=HYPERPARAMS["input length"])
    )
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(HYPERPARAMS["vocab size"], activation='softmax'))
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=HYPERPARAMS["learning rate"]),
        loss=weighted_multilabel_loss(pos_weight=HYPERPARAMS["pos weight"]),
        metrics=["acc"],
    )
    if summary:
        model.summary()
    return model



def train_model(model, x_train, y_train):
    """
    Trains new model using data from filename
    
    x_train = (n, 84, 192) array
    y_train = (n, 84, 1) one-hot-encoded output of next note
    """
    
    # prepare model save and tensorboard
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_callback = ModelCheckpoint(filepath=f"saves/{now}.ckpt", save_weights_only=True, verbose=0)
    tensorboard_callback = TensorBoard(log_dir=f"logs/{now}", histogram_freq=1)
    
    # start training
    model.fit(
        x_train, y_train,
        batch_size=HYPERPARAMS["batch size"],
        epochs=HYPERPARAMS["epochs"],
        validation_split=HYPERPARAMS["validation ratio"],
        shuffle=True,
        verbose=2,
        callbacks=[model_save_callback, tensorboard_callback]
    )



def loadmodel():
    """
    Loads previously trained model
    """
    model = newmodel(summary=False)
    model.load_weights("saves/saves.ckpt")
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=HYPERPARAMS["learning rate"]),
        loss=weighted_multilabel_loss(pos_weight=HYPERPARAMS["pos weight"]),
        metrics=["acc"],
    )
    return model



if __name__ == '__main__':
    dataset = load_tracks_from_genre(GENRE, TRACK)
    X = np.split(dataset, HYPERPARAMS["input length"])
    #n = X.shape(1)
    Y = X[:, 0, :]
    print(X.shape)
    print(Y.shape)

    #newmodel()
