import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from tqdm import tqdm

def rec_and_clip():
    sr = 8000
    seconds = 3

    print('recording begins in... ')
    for count in reversed(range(1, 3+1)):
        time.sleep(1)
        print(count)
        
    print('RECORDING') 
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1)
    for timer in tqdm(range(1, seconds*1000)):
        time.sleep(0.001)
     
    #sd.wait()
    print('recording done')

    step = 1000
    framesize = sr
    mags = []
    for frame in range(0, len(audio)-framesize, step):
        mags.append(np.sum(np.abs(audio[frame:frame+framesize])))

    #sf.write('input-full.wav', audio, samplerate=sr)

    maxenergyframe = np.argmax(mags)
    framestart = maxenergyframe*step

    sf.write('input.wav', audio[framestart:framestart+framesize], samplerate=sr)
    return 
