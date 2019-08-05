#import all dependencies
import pandas as pd 
import numpy as np 
from glob import glob # file handling
import librosa #audio manipulation
from sklearn.utils import shuffle #shuffling of data
import os # interation with the OS
from random import sample # random selection
from tqdm import tqdm 
from scipy import signal #audio processing
from scipy.io import wavfile # reading the wavefile
import matplotlib as plt
from SRData import load_dataset_from_hdf5
from sklearn.model_selection import train_test_split
import pickle

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau 
from keras.layers import BatchNormalization, Dense, Flatten, Dropout, Input
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.layers import LSTM, Dense, Dropout

MODEL_FILE = 'SRModel.h5'
MULTIHEAD_MODEL_FILE = 'SRMultiHeadModel.h5'
HISTORY_FILE = 'SBTrainHistoryDict'
SOURCE_DATA_FILE = 'SRData.h5'

EPOCHS = 100
BATCH_SIZE = 64
ADAM_LEARNING_RATE = 0.00001
VERBOSE = 1

# load, shuffle, one-hot encode, and split the data up for the model
def load_data():
    x_train, y_train = load_dataset_from_hdf5(SOURCE_DATA_FILE)

    # split the data into training and validation sets, 80/10
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2) 

    # one-hot encode target values
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    return x_train, x_valid, y_train, y_valid

def evaluate_model(x_train, x_valid, y_train, y_valid, load_mod, show_plots):
    # load a saved Keras model
    if load_mod:
        model = load_model(MODEL_FILE)
        print('model loaded')
    else:
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        model=Sequential()
        # Input LSTM
        model.add(LSTM(256,input_shape=(n_timesteps, n_features)))
        model.add(BatchNormalization())
        # Dropout layer, randomly removes p percentage of values, helps with overfitting
        model.add(Dropout(0.2))
        model.add(Dense(128))
        model.add(Dropout(0.2))
        model.add(Dense(n_outputs, activation = 'softmax'))
        model.compile(optimizer = 'Adam', loss = 'mean_squared_error', metrics = ['accuracy'])
        
        earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(MULTIHEAD_MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model.fit(x_train, y_train, validation_data=(x_valid, y_valid), callbacks=[earlyStopping, mcp_save, reduce_lr_loss], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

def main():
    x_train, x_valid, y_train, y_valid = load_data()
    history = evaluate_model(x_train,x_valid,y_train,y_valid,False,False)

    with open(HISTORY_FILE,'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    
    return


if __name__ == '__main__':
    main()