from keras.models import load_model
from librosa.core import resample
from SRDataGenerator import DataGenerator
import sys
import numpy as np

def main():
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'other']
    wav_file = sys.argv[1]
    print(f'Using file {wav_file}') 
    data_generator = DataGenerator('data', [], [], 8000, 0.97, .024, 512, 40, 10)
    test_data = data_generator.wav_to_vector(wav_file, showgraphs=False)
    #print(test_data.shape)
    model = load_model('SpeechRecog.h5')
    model.summary()
    preds = model.predict(np.expand_dims(test_data, axis=0))
    print(f'Prediction is {words[np.argmax(preds)]}') 

if __name__ == '__main__':
    main()
