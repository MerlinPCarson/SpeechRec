from keras.models import load_model
from librosa.core import resample
from playsound import playsound
from random import randint
from gtts import gTTS
from SRDataGenerator import DataGenerator
from SRRec import rec_and_clip
import sys
import numpy as np

phrases = ['I predict you said ', 'Did you say ']
unknown = "I don't know that word"

def write_phrase(pred, filename):
    if pred == 'other':
        phrase = unknown 
    else:
        phrase = phrases[randint(0,len(phrases)-1)] + pred
   
    tts = gTTS(phrase)
    tts.save('pred.mp3')

    
 
def main():
    words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go', 'other']
    wav_file = 'input.wav' #sys.argv[1]
    pred_filename = 'pred.mp3'
    rec_and_clip()
    print(f'Using file {wav_file}') 

    data_generator = DataGenerator('data', [], [], 8000, 0.97, .024, 512, 40, 10)
    test_data = data_generator.wav_to_vector(wav_file, showgraphs=False)
    #print(test_data.shape)

    model = load_model('SpeechRecog.h5')
    model.summary()
    preds = model.predict(np.expand_dims(test_data, axis=0))

    pred = words[np.argmax(preds)]
    print(f'Prediction is {pred}') 
    write_phrase(pred, pred_filename)

    playsound(pred_filename)

if __name__ == '__main__':
    main()
