import matplotlib as plt
from SRData import load_dataset_from_hdf5
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 
from keras.layers import BatchNormalization, Dense, Flatten, Dropout, Input, LeakyReLU
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam, Adamax
from keras.utils import to_categorical


MODEL_FILE = 'SRModel.h5'
MULTIHEAD_MODEL_FILE = 'SRMultiHeadModel.h5'
SOURCE_DATA_FILE = 'SRData.h5'
HISTORY_FILE = 'ModelHistory'

NORMALIZE = False
STANDARDIZE = True
VALIDATE = True

EPOCHS = 250
BATCH_SIZE = 256
ADAM_LEARNING_RATE = 0.005
VERBOSE = 1
ALPHA = 0.01
REG_VAL = 0.0001


# load, shuffle, one-hot encode, and split the data up for the model
def load_data():
    x_train, y_train = load_dataset_from_hdf5(SOURCE_DATA_FILE)

    if VALIDATE:
        # split the data into training and validation sets, 90/10
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.05) 

        if STANDARDIZE:
            x_train_mean = np.mean(x_train)
            x_train_std = np.std(x_train)
            x_train = (x_train-x_train_mean)/x_train_std
            x_valid = (x_valid-x_train_mean)/x_train_std

            # add mean and standard deviation to parameter file
            params = {}
            with open('params.npy', 'rb') as paramfile:
                params = pickle.load(paramfile)

            params['Std'] = x_train_std
            params['Mean'] = x_train_mean

            with open('params.npy', 'wb') as paramfile:
                pickle.dump(params, paramfile)

        if NORMALIZE:
            x_train_max = np.max(x_train)
            x_train_min = np.min(x_train)
            x_train = (x_train-x_train_min)/(x_train_max-x_train_min)
            x_valid = (x_valid-x_train_min)/(x_train_max-x_train_min)

        # one-hot encode target values
        y_valid = to_categorical(y_valid)

    # one-hot encode target values
    y_train = to_categorical(y_train)

    if VALIDATE:
        return x_train, x_valid, y_train, y_valid
    else:
        return x_train, None, y_train, None


def evaluate_model(x_train, x_valid, y_train, y_valid, load_file):
    # load a saved Keras model
    if load_file:
        model = load_model(MODEL_FILE)
    else:
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        model=Sequential()
        # Input convolutional layer
        model.add(Conv1D(filters=64, kernel_size=8, input_shape=(n_timesteps, n_features), kernel_regularizer=regularizers.l2(REG_VAL), bias_regularizer=regularizers.l2(REG_VAL)))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization())
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.4))
        # Second convolutional layer
        model.add(Conv1D(filters=64, kernel_size=8, kernel_regularizer=regularizers.l2(REG_VAL), bias_regularizer=regularizers.l2(REG_VAL)))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.4))
        # Third convolutional layer
        model.add(Conv1D(filters=64, kernel_size=8, kernel_regularizer=regularizers.l2(REG_VAL), bias_regularizer=regularizers.l2(REG_VAL)))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization())
        # model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.4))
        # Fourth convolutional layer
        model.add(Conv1D(filters=64, kernel_size=8, padding='same', kernel_regularizer=regularizers.l2(REG_VAL), bias_regularizer=regularizers.l2(REG_VAL)))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(BatchNormalization())
        model.add(MaxPooling1D(pool_size=2))
        model.add(Dropout(0.2))

        # Flatten convolution out to attach to dense NN
        model.add(Flatten())
        # Hidden layer
        model.add(Dense(500, kernel_regularizer=regularizers.l2(REG_VAL), bias_regularizer=regularizers.l2(REG_VAL)))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(Dropout(0.25))
        model.add(Dense(500, kernel_regularizer=regularizers.l2(REG_VAL), bias_regularizer=regularizers.l2(REG_VAL)))
        model.add(LeakyReLU(alpha=ALPHA))
        model.add(Dropout(0.25))
        # Softmax output layer so that the the values are turned into probablities to assess
        model.add(Dense(n_outputs, activation='softmax'))
        # opt=Adamax(lr=ADAM_LEARNING_RATE)  # Using if we want to set the initial learning rate value
        model.compile(loss='categorical_crossentropy', optimizer='adamax', metrics=['accuracy'])
        
    if VALIDATE:
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, min_lr=1e-8, cooldown=15, verbose=1, epsilon=1e-3, mode='min')
        earlyStopping = EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='min')
        mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
        model.summary()

        return model.fit(x_train, y_train, validation_data=(x_valid, y_valid), callbacks=[earlyStopping, mcp_save, reduce_lr_loss], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    else:
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-8, cooldown=25, verbose=1, epsilon=1e-4, mode='min')
        earlyStopping = EarlyStopping(monitor='loss', patience=100, verbose=1, mode='min')
        mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='loss', mode='min')
        model.summary()

        return model.fit(x_train, y_train, callbacks=[earlyStopping, mcp_save, reduce_lr_loss], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)


def main():
    x_train, x_valid, y_train, y_valid = load_data()
    history = evaluate_model(x_train, x_valid, y_train, y_valid, False)

    with open(HISTORY_FILE, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return


main()