
# coding: utf-8

# In[52]:


# Install a pip package in the current Jupyter kernel
import sys
get_ipython().system('{sys.executable} -m pip install wavefile --user')


# In[1]:


import os, sys
import soundfile as sf
import scipy.io.wavfile
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy import signal
from librosa import power_to_db
from librosa.feature import mfcc, melspectrogram
from librosa.core import resample
from librosa.display import specshow
import noisereduce as nr
import wavefile
#import sounddevice as sd
#import librosa
import numpy as np
import glob
import matplotlib.pyplot as plt
from IPython.display import Audio
import thinkdsp
import subprocess
from subprocess import Popen, PIPE
#from sklearn.model_selection import train_test_split
#import pickle
# Keras packages model
#from keras.models import Sequential, Model, load_model
#from keras.layers import Input, Reshape, Dense, GRU, Convolution1D, Convolution2D, MaxPooling1D, MaxPooling2D, Dropout, BatchNormalization, Flatten, concatenate 
#from keras.callbacks import Callback, ModelCheckpoint, CSVLogger, EarlyStopping
#from keras import regularizers

SAMPLERATE = 8000
DATAPATH = 'data/test'


# In[100]:


# Download raw data from source
# and uncompress to DATAPATH

os.makedirs(DATAPATH, exist_ok=True)

url = 'http://download.tensorflow.org/data/'
file_name = 'speech_commands_v0.01.tar.gz'

if not os.path.exists(DATAPATH):
    os.makdirs(DATAPATH)
        
if not os.path.exists(os.path.join(DATAPATH, file_name)):
    
    print('Downloading', file_name)
    rtnVal = subprocess.call(['wget', os.path.join(url,file_name), '-P', DATAPATH])
    assert rtnVal == 0, 'downloaded failed!'
    print(file_name, 'downloaded successfully')
    
    print('Uncompressing', os.path.basename(file_name))
    rtnVal = subprocess.call(['tar', '-C', DATAPATH, '-zxvf', os.path.join(DATAPATH, file_name)])
    assert rtnVal == 0, 'file failed to uncompress!'
    print(file_name, 'uncompressed successfully')

# save data at correct SAMPLERATE
for wav_file in glob.iglob(os.path.join(DATAPATH, 'bed', '*.wav')):
    print('File:', wav_file)
    samples, sample_rate = sf.read(wav_file)
    
    if sample_rate != SAMPLERATE:
        print('Resampling file ', wav_file, 'from', sample_rate, 'to', SAMPLERATE)
        samples = resample(samples, sample_rate, SAMPLERATE)
        sf.write(wav_file, samples, SAMPLERATE)    
        
print('Data downloaded and resampled to', SAMPLERATE, 'Hz')


# In[13]:


wave_file = 'data/no/9db2bfe9_nohash_0.wav'
wave_file = 'data/yes/9b6c08ba_nohash_2.wav'
wave_file = 'data/test/bed/01b4757a_nohash_0.wav'

samples, sample_rate = sf.read(wave_file)

# if sample_rate != SAMPLERATE:
#     print(sample_rate)
#     samples = resample(samples, sample_rate, SAMPLERATE)
    
# pre_emphasis
pre_emphasis = 0.97
emphasized_samples = signal.lfilter(  [1],[1, pre_emphasis], samples )



noise = emphasized_samples[:1000] + emphasized_samples[-1000:]
samples_nr = nr.reduce_noise(audio_clip=emphasized_samples, noise_clip=noise, verbose=True)

plt.figure()
plt.plot(emphasized_samples)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid(True)

plt.figure()
plt.plot(samples_nr)
plt.xlabel('time')
plt.ylabel('amplitude')
plt.grid(True)

# get MFCCs
print(len(samples), sample_rate)
S = melspectrogram(y=samples, sr=sample_rate)
mfccs = mfcc(y=samples, sr= sample_rate, n_mfcc=128)
print(mfccs.shape)
plt.figure(figsize=(10, 4))
specshow(mfccs, x_axis='time', sr=sample_rate)
plt.colorbar()
plt.title('MFCC')
plt.xlabel('time')
plt.ylabel('coeffecients')
plt.tight_layout()
plt.show()

#Audio(samples, rate=SAMPLERATE)
Audio(samples_nr, rate=SAMPLERATE)


# In[101]:


wave_file = 'data/test/bed/00176480_nohash_0.wav'
#wave_file = 'data/no/9db2bfe9_nohash_0.wav'
#wave_file = 'data/yes/9b6c08ba_nohash_2.wav'
samples, sample_rate = sf.read(wave_file)

if sample_rate != SAMPLERATE:
    samples = resample(samples, sample_rate, SAMPLERATE)

# test pre-empahsis filter (view and hear)
plt.figure()
plt.plot(samples)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)
Audio(samples, rate=SAMPLERATE)

pre_emphasis = 0.97  # coeffecient for pre-emphasis filter, boost high end for cleaner FFT
emphasized_samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])
# pre_emphasis
proc_samples = signal.lfilter(  [1],[1, pre_emphasis], samples )
# post_emphasis
#proc_samples = signal.lfilter( [1, pre_emphasis], [1], samples )
print(samples)
print(proc_samples)

# test pre-empahsis filter (view and hear)
plt.figure()
plt.plot(emphasized_samples)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)


# get MFCCs
S = melspectrogram(y=samples, sr=sample_rate)
print(S.shape)
mfccs = mfcc(emphasized_samples, n_mfcc=40)
print(mfccs.shape)
plt.figure(figsize=(10, 4))
specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.xlabel('time')
plt.ylabel('coeffecients')
plt.tight_layout()
plt.show()

Audio(proc_samples, rate=SAMPLERATE)


# In[102]:


wave_file = 'data/yes/9b6c08ba_nohash_2.wav'
wave_file = 'data/test/bed/01b4757a_nohash_0.wav'
samples, sample_rate = sf.read(wave_file)

if sample_rate != SAMPLERATE:
    samples = resample(samples, sample_rate, SAMPLERATE)

# test pre-empahsis filter (view and hear)
plt.figure()
plt.plot(samples)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)
Audio(samples, rate=SAMPLERATE)

pre_emphasis = 0.97  # coeffecient for pre-emphasis filter, boost high end for cleaner FFT
emphasized_samples = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

# test pre-empahsis filter (view and hear)
plt.figure()
plt.plot(emphasized_samples)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)


# get MFCCs
S = melspectrogram(y=emphasized_samples, sr=sample_rate, n_mels=128, fmax=8000)
mfccs = mfcc(S=power_to_db(S), n_mfcc=20)
plt.figure(figsize=(10, 4))
specshow(mfccs, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.xlabel('time')
plt.ylabel('Coeffecients')
plt.tight_layout()
plt.show()

Audio(emphasized_samples, rate=SAMPLERATE)


# In[9]:


#Examine a wave's spectrum
high_cutoff = 3400
low_cutoff = 300

#waveFile = 'data/trainingNums/05_620a81c69b-VAD.wav'
wave_file = 'data/test/bed/00176480_nohash_0.wav'
samples, sample_rate = sf.read(wave_file)
print('Sample Rate:', sample_rate)
print('Num Samples:', len(samples))

if sample_rate != SAMPLERATE:
    samples = resample(samples, sample_rate, SAMPLERATE)
    
plt.figure()
sample = thinkdsp.read_wave(wave_file)
spectrum = sample.make_spectrum()
spectrum.low_pass(cutoff=high_cutoff, factor=0.01)
spectrum.high_pass(cutoff=low_cutoff, factor=0.01)

spectrum.plot()
plt.figure()
plt.plot(samples)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)

Audio(samples, rate=SAMPLERATE)




# In[57]:


wave_file = 'data/bed/9d050657_nohash_1.wav'
wave_file = 'data/six/cc6ee39b_nohash_4.wav'
samples, sample_rate = sf.read(wave_file)

if sample_rate != SAMPLERATE:
    samples = resample(samples, sample_rate, SAMPLERATE)
    
#sample_rate, samples = scipy.io.wavfile.read(wave_file)  # File assumed to be in the same directory

# test pre-empahsis filter (view and hear)
plt.figure()
plt.plot(samples)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)
Audio(samples, rate=SAMPLERATE)

pre_emphasis = 0.97  # coeffecient for pre-emphasis filter, boost high end for cleaner FFT
emphasized_signal = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

# test pre-empahsis filter (view and hear)
plt.figure()
plt.plot(emphasized_signal)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)
Audio(emphasized_signal, rate=SAMPLERATE)

# set up frames for FFTs 
frame_size = 0.024
frame_stride = frame_size/2

signal_length = len(emphasized_signal)
frame_length = int(round(frame_size * SAMPLERATE))
frame_step = int(round(frame_stride * SAMPLERATE))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)/frame_step)))

pad_signal_length = num_frames * frame_step + frame_length
z = np.zeros((pad_signal_length - signal_length))
pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy=False)]
print(frames.shape)
# windowing
frames *= np.hamming(frame_length)

# Convert to FFTs and power spectrum
NFFT = 512
mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
#print(pow_frames.shape)
freqs = np.arange(0, len(mag_frames[0]), dtype='float64') * ((SAMPLERATE/2)/len(mag_frames[0]))

# show FFT mag
plt.figure()
plt.plot(freqs, mag_frames[0])
plt.xlabel('frequency')
plt.ylabel('magnitude')
#plt.yscale('log')
plt.grid(True)

# show power spectrum
plt.figure()
plt.imshow(pow_frames.T, aspect='auto')
plt.xlabel('frequency')
plt.ylabel('power')
#plt.yscale('log')
plt.grid(True)

# mel filter banks
nfilt = 40
low_freq_mel = 0
high_freq_mel = (2595 * np.log10(1 + (SAMPLERATE / 2) / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / SAMPLERATE)
#print(mel_points)
#print(hz_points)
#print(bin.shape)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
print(fbank.shape)

print(bin)
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
#print(fbank.shape)
filter_banks = np.dot(pow_frames, fbank.T)
#print(filter_banks.shape)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
filter_banks = 20 * np.log10(filter_banks)  # dB

# mean normalization for mel filter banks
#filter_banks -= (np.mean(filter_banks, axis=0) + 1e-8)
print(filter_banks.shape)
# show banks
plt.figure()
xdata = np.array((range(NFFT//2+1)), dtype='float')
xdata *= SAMPLERATE/NFFT
for i in range(fbank.shape[0]):
    plt.plot(xdata, fbank[i,:])
#plt.plot(fbank)
plt.xlabel('frequency')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)

# show mel filter banks
plt.figure()
plt.imshow(filter_banks.T, aspect='auto')
plt.xlabel('time')
plt.ylabel('frequency')
#plt.yscale('log')
plt.grid(True)

# get MFCCs
num_ceps = 12
mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)] # Keep 2-13

# sinusoidal liftering1 to the MFCCs to de-emphasize higher MFCCs
(nframes, ncoeff) = mfcc.shape
n = np.arange(ncoeff)
cep_lifter = 22
lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
mfcc *= lift

# mean normalization for MFCCs
mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

# show MFCCs
plt.figure()
plt.imshow(mfcc.T, aspect='auto')
plt.xlabel('time')
plt.ylabel('Coeffecients')
#plt.yscale('log')
plt.grid(True)


# In[30]:


print(SAMPLERATE)


# In[3]:


wave_file = 'data/bed/9d050657_nohash_1.wav'
samples, sample_rate = sf.read(wave_file)

if sample_rate != SAMPLERATE:
    samples = resample(samples, sample_rate, SAMPLERATE)

pre_emphasis = 0.97  # coeffecient for pre-emphasis filter, boost high end for cleaner FFT
emphasized_signal = np.append(samples[0], samples[1:] - pre_emphasis * samples[:-1])

Audio(emphasized_signal, rate=SAMPLERATE)


# In[ ]:


frame_size = 0.025
frame_stride = 0.01

signal_length = len(emphasized_signal)
frame_length = int(round(frame_size * SAMPLERATE))
frame_step = int(round(frame_stride * SAMPLERATE))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)/frame_step)))


# In[84]:


# load a wave, plot and play
fileName = 'data/filteredNums/12_cc5336bbc0.wav' 
#fileName = 'data/trainingNums/06_1507906720.wav'
fileName = 'data/tst_numbers_test2/cb2b6b25ca.wav'
#fileName = 'data/trainingNums2/12_a398641269.wav'
#fileName = 'data/trainingNums4/11_817144b701.wav'
#fileName = 'data/numbers/11_5589b5726b.wav'

data, sample_rate = sf.read(fileName)
print(sample_rate)
print(len(data))
duration = len(data)//5
#print(duration)
#print(max(data), min(data))
# test clip from max energy out
#maxEnergy = np.argmax(data)
#data = data[maxEnergy-(SAMPLERATE//2):maxEnergy+(SAMPLERATE//2)]
#print('new num samples', len(data))
frameSize = 6000
#data = highest_energy_frame(data, frameSize)
#print(len(data))

plt.figure()
plt.plot(data)
plt.xlabel('time')
plt.ylabel('amplitude')
#plt.yscale('log')
plt.grid(True)

plt.figure()
sample = thinkdsp.read_wave(fileName)
spectrum = sample.make_spectrum()
spectrum.plot()
#mags = abs(np.real(np.fft.rfft(data[:duration])))
#print(mags.shape)
#plt.figure()
#plt.plot(mags)
#plt.xlabel('freq')
#plt.ylabel('amplitude')
#plt.yscale('log')
#plt.grid(True)

#clippedSamples = clip_audio(data)
Audio(data, rate=sample_rate)

