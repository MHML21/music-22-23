import csv
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

from settings import *
from preprocessing import *



def newmodel(summary=True):
    """
    Creates model
    Input dims (batch, vocab size, input length)
    Ouptut dims (batch, vocab size, output length)
    
    Embedding -> LSTM(dropout) -> LSTM -> Dense(dropout) -> Output
    """
    model = Sequential()
    model.add(layers.InputLayer(
        input_shape=(HYPERPARAMS["vocab size"], HYPERPARAMS["input length"]),
        batch_size=HYPERPARAMS["batch size"]
    ))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(256))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(HYPERPARAMS["vocab size"], activation='sigmoid'))
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=HYPERPARAMS["learning rate"]),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["acc"],
    )
    if summary:
        model.summary()
    return model



def train_model(model, X, Y):
    """
    Trains new model using data from filename
    
    X = (n, vocab size, input length) array
    Y = (n, vocab size, output length) one-hot-encoded output of next note
    """
    
    # prepare model save and tensorboard
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_save_callback = ModelCheckpoint(filepath=f"saves/{now}.ckpt", save_weights_only=True, verbose=0)
    tensorboard_callback = TensorBoard(log_dir=f"logs/{now}", histogram_freq=1)
    
    # start training
    model.fit(
        X, Y,
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
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["acc"],
    )
    return model



def load_data():
    dataset = load_tracks_from_genre(GENRE, TRACK)
    ndata = (dataset.shape[0] // HYPERPARAMS["input length"]) * HYPERPARAMS["input length"] # trim dataset to multiple of input length

    # split traindata into multiple data of size (notes, input length)
    X = np.array(np.split(dataset[:ndata, :], HYPERPARAMS["input length"])) # output dim = (input length, sample count, notes)
    X = np.moveaxis(X, 0, -1)      # move input length to last axis
    Y = X[:, :, :HYPERPARAMS["output length"]]
    
    # last X has no corresponding Y, first Y has no corresponding X
    X = X[:-1]
    Y = np.squeeze(Y[1:])
    
    print(f"Loaded {X.shape[0]} training samples")
    print(f"X shape = {X.shape}")
    print(f"Y shape = {Y.shape}")
    return (X, Y)

    



if __name__ == '__main__':
    X, Y = load_data()
    model = newmodel()
    train_model(model, X, Y)
