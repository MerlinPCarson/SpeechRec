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


MODEL_FILE = 'SRModel.h5'
MULTIHEAD_MODEL_FILE = 'SRMultiHeadModel.h5'
HISTORY_FILE = 'SBTrainHistoryDict'
SOURCE_DATA_FILE = 'SRData.h5'

EPOCHS = 1000
BATCH_SIZE = 64
ADAM_LEARNING_RATE = 0.00001
VERBOSE = 1


# load, shuffle, one-hot encode, and split the data up for the model
def load_data():
    x_train, y_train = load_dataset_from_hdf5(SOURCE_DATA_FILE)

    # split the data into training and validation sets, 90/10
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1) 

    # one-hot encode target values
    y_train = to_categorical(y_train)
    y_valid = to_categorical(y_valid)

    return x_train, x_valid, y_train, y_valid


def evaluate_model(x_train, x_valid, y_train, y_valid, load_mod, show_plots):
    # load a saved Keras model
    if load_mod:
        model = load_model(MODEL_FILE)
    else:
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        model=Sequential()
        # Input convolutional layer
        model.add(Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(n_timesteps, n_features)))
        model.add(BatchNormalization())
        # Dropout layer, randomly removes p percentage of values, helps with overfitting
        model.add(Dropout(0.55))
        # Second convolutional layer
        model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.55))
        # Pooling layer to remove noise
        model.add(MaxPooling1D(pool_size=2))
        # Flatten convolution out to attach to dense NN
        model.add(Flatten())
        # Hidden layer
        model.add(Dense(150, activation='relu'))
        # Softmax output layer so that the the values are turned into probablities to assess
        model.add(Dense(n_outputs, activation='softmax'))

        # opt=Adam(lr=ADAM_LEARNING_RATE)  # Using if we want to set the initial learning rate value
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, min_lr=1e-8, cooldown=25, verbose=1, epsilon=1e-4, mode='min')
    earlyStopping = EarlyStopping(monitor='val_loss', patience=75, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
    model.summary()
	
    return model.fit(x_train, y_train, validation_data=(x_valid, y_valid), callbacks=[earlyStopping, mcp_save, reduce_lr_loss], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)



def evaluate_model_multihead(x_train, x_valid, y_train, y_valid, load_model, show_plots):
    # load a saved Keras model
    if load_model:
        model = load_model(MULTIHEAD_MODEL_FILE)
    else:
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        # head 1
        inputs1 = Input(shape=(n_timesteps,n_features))
        conv1 = Conv1D(filters=32, kernel_size=3, activation='relu')(inputs1)
        batch1 = BatchNormalization()(conv1)
        drop1 = Dropout(0.44)(batch1)
        # conv2 = Conv1D(filters=64, kernel_size=3, activation='relu')(drop1)
        # batch2 = BatchNormalization()(conv2)
        # drop2 = Dropout(0.7)(batch2)
        pool2 = MaxPooling1D(pool_size=2)(drop1)
        # pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat1 = Flatten()(pool2)
        # flat1 = Flatten()(pool2)
        # head 2
        inputs2 = Input(shape=(n_timesteps,n_features))
        conv3 = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs2)
        batch3 = BatchNormalization()(conv3)
        drop3 = Dropout(0.44)(batch3)
        # conv4 = Conv1D(filters=64, kernel_size=5, activation='relu')(drop3)
        # batch4 = BatchNormalization()(conv4)
        # drop4 = Dropout(0.7)(batch4)
        pool4 = MaxPooling1D(pool_size=2)(drop3)
        # pool4 = MaxPooling1D(pool_size=2)(drop4)
        flat2 = Flatten()(pool4)
        # flat2 = Flatten()(pool4)
        # head 3
        inputs3 = Input(shape=(n_timesteps,n_features))
        conv5 = Conv1D(filters=32, kernel_size=7, activation='relu')(inputs3)
        batch5 = BatchNormalization()(conv5)
        drop5 = Dropout(0.44)(batch5)
        # conv6 = Conv1D(filters=64, kernel_size=7, activation='relu')(drop5)
        # batch6 = BatchNormalization()(conv6)
        # drop6 = Dropout(0.7)(batch6)
        pool6 = MaxPooling1D(pool_size=2)(drop5)
        # pool6 = MaxPooling1D(pool_size=2)(drop6)
        flat3 = Flatten()(pool6)
        # flat3 = Flatten()(pool6)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(200, activation='relu')(merged)
        outputs = Dense(n_outputs, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        earlyStopping = EarlyStopping(monitor='val_loss', patience=100, verbose=0, mode='min')
        mcp_save = ModelCheckpoint(MULTIHEAD_MODEL_FILE, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=20, verbose=1, epsilon=1e-4, mode='min')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
    return model.fit([x_train,x_train,x_train], y_train, validation_data=([x_valid, x_valid, x_valid], y_valid), callbacks=[earlyStopping, mcp_save, reduce_lr_loss], epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)


def main():
    x_train, x_valid, y_train, y_valid = load_data()
    history = evaluate_model(x_train, x_valid, y_train, y_valid, False, False)
    # history = evaluate_model_multihead(x_train, x_valid, y_train, y_valid, False, False)

    with open(HISTORY_FILE, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

    return



main()