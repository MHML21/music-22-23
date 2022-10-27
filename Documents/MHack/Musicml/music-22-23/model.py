import csv
import datetime
import numpy as np
from settings import HYPERPARAMS, VOCAB_SIZE
from tensorflow.keras import Sequential, layers, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard




def newmodel(summary=True):
    """
    Creates model
    Embedding -> LSTM(dropout) -> LSTM -> Dense(dropout) -> Output
    """
    model = Sequential()
    model.add(layers.Embedding(VOCAB_SIZE, HYPERPARAMS["embedding size"], input_length=HYPERPARAMS["input length"]))
    model.add(layers.LSTM(64, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(64))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(VOCAB_SIZE, activation='softmax'))
    model.compile(
        optimizer=optimizers.RMSprop(learning_rate=HYPERPARAMS["learning rate"]),
        loss='categorical_crossentropy',
        metrics=["acc"],
    )
    if summary:
        model.summary()
    return model



def trainmodel(model, filename):
    """
    Trains new model using data from filename
    """
    # prepare training data
    x_train, y_train = loaddata(filename)
    
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
    model = newmodel(False)
    model.load_weights("saves/saves.ckpt")
    return model



def loaddata(filename):
    with open(f'{filename}.csv', 'r', encoding='UTF8') as f:
        reader = csv.reader(f)
        fulldata = np.array([r for r in reader])
        x, y_true = fulldata[:,:-1].astype(int), fulldata[:,-1].astype(int)
        # one hot encode y
        y = np.zeros((y_true.size, VOCAB_SIZE))
        y[np.arange(y_true.size), y_true] = 1
        return x, y



if __name__ == '__main__':
    # creates and trains model
    # tensorboard --logdir logs
    model = newmodel()
    trainmodel(model, "data/cleandata")