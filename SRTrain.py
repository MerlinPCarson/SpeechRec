import pickle
from sklearn.model_selection import train_test_split
# Keras packages model
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, GRU, Convolution1D, MaxPooling1D, Dropout, BatchNormalization, Flatten, concatenate 
from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
from keras import regularizers
from SRData import load_dataset_from_hdf5


# Model definitions
def model_dense(batchSize, numNeurons, timesteps, num_features, outputs):
    regVal = 0.0
    x_in = Input(batch_shape=(None, timesteps, num_features))
    x = Dense(numNeurons, activation='relu', kernel_regularizer=regularizers.l2(regVal), bias_regularizer=regularizers.l2(regVal))(x_in)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(numNeurons, activation='relu', kernel_regularizer=regularizers.l2(regVal/2), bias_regularizer=regularizers.l2(regVal/2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(numNeurons, activation='relu', kernel_regularizer=regularizers.l2(regVal/3), bias_regularizer=regularizers.l2(regVal/3))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)   
    x = Dense(numNeurons, activation='relu', kernel_regularizer=regularizers.l2(regVal/4), bias_regularizer=regularizers.l2(regVal/4))(x)
    x = Dense(outputs, activation='softmax', kernel_regularizer=regularizers.l2(regVal/5), bias_regularizer=regularizers.l2(regVal/5))(x)
    model = Model(inputs=[x_in], outputs=[x])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_conv(batchSize, numNeurons, outputs, freqBins):
    regVal = 0.0001
    x_in = Input(batch_shape=(None, 4, freqBins))
    x = Convolution1D(128, 2, activation='relu')(x_in)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Convolution1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Convolution1D(32, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
#    x = MaxPooling1D(pool_size=2)(x)
    x = Flatten()(x)
    x = Dense(numNeurons, activation='relu')(x)
    x = Dense(outputs, activation='softmax', kernel_regularizer=regularizers.l2(regVal), bias_regularizer=regularizers.l2(regVal))(x)
    model = Model(inputs=[x_in], outputs=[x])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def model_rnn(batchSize, numNeurons, outputs, freqBins):
    regVal = 0.0001
    x_in = Input(batch_shape=(None, 4, freqBins))
    x = Dense(numNeurons, activation='relu', kernel_regularizer=regularizers.l2(regVal), bias_regularizer=regularizers.l2(regVal))(x_in)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = GRU(numNeurons, activation='relu', return_sequences=True, stateful=False, kernel_regularizer=regularizers.l2(regVal), bias_regularizer=regularizers.l2(regVal))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = GRU(numNeurons, activation='relu', return_sequences=False, stateful=False, kernel_regularizer=regularizers.l2(regVal), bias_regularizer=regularizers.l2(regVal))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
#    x = Flatten()(x) 
    x = Dense(numNeurons, activation='relu')(x)
    x = Dense(outputs, activation='softmax', kernel_regularizer=regularizers.l2(regVal), bias_regularizer=regularizers.l2(regVal))(x)
    model = Model(inputs=[x_in], outputs=[x])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def main():
    datasetfile = 'SRData.h5'
    words = ['yes', 'no']
    num_neurons = 256
    batch_size = 32
    epochs = 100

    # Loads dataset from disk
    x_train, y_train = load_dataset_from_hdf5(datasetfile)
    
    print(f'Training vector: {x_train.shape} --> (Num Examples, Num Time Steps, Features)')
    print(f'Target vector: {y_train.shape} --> (Num Examples, Word Number)' )

    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15, shuffle=True )

    SpeechRecog = model_dense(batch_size, num_neurons, x_train.shape[1], x_train.shape[2], len(words))
    SpeechRecog.summary()

    modelFile = 'SpeechRecog.h5'
    bestModelCheckpoint = ModelCheckpoint(modelFile, save_best_only=True)
    
    model_history = SpeechRecog.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=batch_size, epochs=epochs, callbacks=[bestModelCheckpoint])

    with open(modelFile.replace('.h5', '.npy'), "wb") as outfile:
        pickle.dump(model_history.history, outfile)

if __name__ == '__main__':
    main()
