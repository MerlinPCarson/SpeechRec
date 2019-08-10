import sounddevice as sd
import numpy as np
import time
import pickle
from playsound import playsound
from random import randint
from gtts import gTTS
from tqdm import tqdm
from keras.models import load_model
from SRDataGenerator import DataGenerator
import subprocess


class Predictor():

    def  __init__(self, modelfile):
        # setup predictor params
        self.pred_filename = 'pred.mp3'
        self.recordtime = 3 # in seconds
        self.model_file = modelfile 
        self.param_file = 'params.npy'
        self.phrases = ['I predict you said ', 'Did you say ', 'I think you said ', 'Sounded like ', 'I believe you spoke ']
        self.unknown = ["I don't know that word", "I didn't understand you", "Please try again", "I didn't hear you, can you please speak up"]

        # load parameters from file
        with open(self.param_file, 'rb') as paramfile:
            params = pickle.load(paramfile)

        # setup parameters from data generation and training
        self.words = params['Words'] 
        self.samplerate = params['SampleRate'] 
        self.preemphasis = params['Preemphasis'] 
        self.framesize = params['Framesize'] 
        self.windowsize = params['Windowsize'] 
        self.nummels = params['NumMels'] 
        self.nummfccs = params['NumMFCCS'] 
        self.std = params['Std']
        self.mean = params['Mean']

        # create predictor and model object
        self.data_generator = DataGenerator(None, [], [], self.samplerate, self.preemphasis, self.framesize, self.windowsize, self.nummels, self.nummfccs)
        self.model = load_model(self.model_file)
        self.model.summary()


    def write_phrase(self, pred):
        if pred == 'other':
            phrase = self.unknown[randint(0,len(self.unknown)-1)]
        else:
            phrase = self.phrases[randint(0,len(self.phrases)-1)] + pred
       
        tts = gTTS(phrase)
        tts.save(self.pred_filename)

   
    def known_words(self):
        
        print('\nWords that I know: ', end="")
        for word in self.words:
            if word != 'other':
                print(word, end=" ")

        print()

    def record_and_predict(self):
    
        # display known words   
        self.known_words()

        # record audio and create input vector for model 
        samples = self.rec_and_clip()
        test_data = self.data_generator.samples_to_vector(samples, showgraphs=False)
   
        # standardize input vector
        test_data = (test_data - self.mean) / self.std 

        preds = self.model.predict(np.expand_dims(test_data, axis=0))
    
        pred = self.words[np.argmax(preds)]
        print(f'\nPrediction is {pred}\n') 
        self.write_phrase(pred)
    
        # playsound(self.pred_filename)
        rtnVal = subprocess.call(['afplay', self.pred_filename])

 
    def rec_and_clip(self):
    
        print('\nrecording begins in... ')
        for count in reversed(range(1, 3+1)):
            time.sleep(1)
            print(count)
            
        print('\nRECORDING\n') 
        audio = sd.rec(int(self.recordtime * self.samplerate), samplerate=self.samplerate, channels=1)
        for timer in tqdm(range(1, self.recordtime*1000)):
            time.sleep(0.001)
         
        step = 1000
        framesize = self.samplerate 

        mags = []
        for frame in range(0, len(audio)-framesize, step):
            mags.append(np.sum(np.abs(audio[frame:frame+framesize])))
    
        maxenergyframe = np.argmax(mags)
        framestart = maxenergyframe*step

        return audio[framestart:framestart+framesize]
