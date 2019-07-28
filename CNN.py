import matplotlib as plt

from SRData import load_dataset_from_hdf5
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Keras packages model
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Input
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.utils import to_categorical


MODEL_FILE = 'Model.h5'
SOURCE_DATA_FILE = 'SRData.h5'

EPOCHS = 1000
BATCH_SIZE = 33
ADAM_LEARNING_RATE = 0.0001

VERBOSE = 1

# load, shuffle, one-hot encode, and split the data up for the model
def load_data():
    x_train, y_train = load_dataset_from_hdf5(SOURCE_DATA_FILE)

    # split the data into training and test sets, 80/20
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2) 

    # one-hot encode target values
    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test)

    return x_train, x_test, y_train, y_test


def evaluate_model(x_train, x_test, y_train, y_test, load_model, save_model):
    # load a saved Keras model
    if load_model:
        model = load_model(MODEL_FILE)
    else:
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        model=Sequential()
        # Input convolutional layer
        model.add(Conv1D(filters=32, kernel_size=7, activation='relu', input_shape=(n_timesteps, n_features)))
        # # Second convolutional layer
        model.add(Conv1D(filters=32, kernel_size=7, activation='relu'))

        # Dropout layer, randomly removes p percentage of values, helps with overfitting
        model.add(Dropout(0.7))
        # Pooling layer to remove noise
        model.add(MaxPooling1D(pool_size=2))
        # Flatten convolution out to attach to dense NN
        model.add(Flatten())
        # Hidden layer
        model.add(Dense(200, activation='relu'))
        # Softmax output layer so that the the values are turned into probablities to assess
        model.add(Dense(n_outputs, activation='softmax'))

        opt=Adam(lr=ADAM_LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    if save_model:
        model.save(MODEL_FILE)
	
    return loss, accuracy


def evaluate_model_multihead(x_train, x_test, y_train, y_test, load_model, save_model):
# load a saved Keras model
    if load_model:
        model = load_model(MODEL_FILE)
    else:
        n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
        # head 1
        inputs1 = Input(shape=(n_timesteps,n_features))
        conv1 = Conv1D(filters=32, kernel_size=5, activation='relu')(inputs1)
        drop1 = Dropout(0.7)(conv1)
        pool1 = MaxPooling1D(pool_size=2)(drop1)
        flat1 = Flatten()(pool1)
        # head 2
        inputs2 = Input(shape=(n_timesteps,n_features))
        conv2 = Conv1D(filters=32, kernel_size=7, activation='relu')(inputs2)
        drop2 = Dropout(0.7)(conv2)
        pool2 = MaxPooling1D(pool_size=2)(drop2)
        flat2 = Flatten()(pool2)
        # head 3
        inputs3 = Input(shape=(n_timesteps,n_features))
        conv3 = Conv1D(filters=32, kernel_size=9, activation='relu')(inputs3)
        drop3 = Dropout(0.7)(conv3)
        pool3 = MaxPooling1D(pool_size=2)(drop3)
        flat3 = Flatten()(pool3)
        # merge
        merged = concatenate([flat1, flat2, flat3])
        # interpretation
        dense1 = Dense(100, activation='relu')(merged)
        outputs = Dense(n_outputs, activation='softmax')(dense1)
        model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)

        opt=Adam(lr=ADAM_LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        model.fit([x_train, x_train, x_train], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=VERBOSE)

    if save_model:
        model.save(MODEL_FILE)
	
    return loss, accuracy 


def main():
    
    x_train, x_test, y_train, y_test = load_data()
    _, acc = evaluate_model(x_train, x_test, y_train, y_test, False, False)
    _, acc = evaluate_model_multihead(x_train, x_test, y_train, y_test, False, False)

    print(f'Accuracy of model: {acc}')

    return


main()