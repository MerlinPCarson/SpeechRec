import os
import glob
import numpy as np
import soundfile as sf
from scipy.fftpack import dct
from tqdm import tqdm

class DataGenerator():

    def __init__(self, datadir, words, samplerate, preemphasis, framesize, windowsize, num_melfilters, num_mfccs):
        self.datadir = datadir
        self.samplerate = samplerate
        self.preemphasis = preemphasis
        self.words = words
        self.framesize = framesize
        self.windowsize = windowsize
        self.num_melfilters = num_melfilters
        self.num_mfccs = num_mfccs


    def pre_emphasis(self, samples):
        return np.append(samples[0], samples[1:] - self.preemphasis * samples[:-1])


    def time_to_freq(self, samples):
        # setup frames sizes for FFTs 
        frame_stride = self.framesize/2
        signal_length = len(samples)
        frame_length = int(round(self.framesize * self.samplerate))
        frame_step = int(round(frame_stride * self.samplerate))
        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)/frame_step)))
       
        # setup array for FFTs 
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        pad_signal = np.append(samples, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
      
        # setup windows 
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_length)

        # convert samples to frequency domain 
        mag_frames = np.absolute(np.fft.rfft(frames, self.windowsize))  # Magnitude of the FFT

        # get frequency centers for each bin
        freqs = np.arange(0, len(mag_frames[0]), dtype='float64') * ((self.samplerate/2)/len(mag_frames[0]))

        return mag_frames, freqs


    def mag_to_power(self, mag_frames):
        pow_frames = ((1.0 / self.windowsize) * ((mag_frames) ** 2))  # Power Spectrum
        return pow_frames


    def power_to_mel(self, pow_frames):
        # mel filter banks
        low_freq_mel = 0
        high_freq_mel = (2595 * np.log10(1 + (self.samplerate / 2) / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, self.num_melfilters + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bins = np.floor((self.windowsize + 1) * hz_points / self.samplerate)
        
        fbank = np.zeros((self.num_melfilters, int(np.floor(self.windowsize / 2 + 1))))
        
        for m in range(1, self.num_melfilters + 1):
            f_m_minus = int(bins[m - 1])   # left
            f_m = int(bins[m])             # center
            f_m_plus = int(bins[m + 1])    # right
        
            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m])
        
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        return filter_banks


    def mels_to_mfccs(self, filter_banks):
        mfccs = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (self.num_mfccs + 1)] # Keep 2 - num_mfccs + 1

        # sinusoidal lifting to the MFCCs to de-emphasize higher MFCCs
        (nframes, ncoeff) = mfccs.shape
        n = np.arange(ncoeff)
        cep_lifter = 22
        lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
        mfccs *= lift

        return mfccs
        

    def convert_wavs_to_dataset(self):

        x_train_vec = None
        y_train_vec = []

        # only check samplerate for files in word list
        #wav_files = [f for word in words for f in glob.glob(os.path.join('data', word,'*.wav'))] 
        # load all wave files in words directory, enumerate so we can set target vector to the number of element in words list
        wav_files = [(num,f) for num, word in enumerate(self.words) for f in glob.glob(os.path.join('data', word,'*.wav'))]

        print('Generating dataset')
        # process all wave files for specified words
        for wav_file in tqdm(wav_files):
            samples, sr = sf.read(wav_file[1])

            # check length of wav file 
            if len(samples) != self.samplerate:
                samples = np.append(samples, np.zeros(self.samplerate-len(samples)))    # pad shorter wav files with 0s

            # verify file is correct samplerate
            if sr != self.samplerate:
                samples = resample(samples, sr, self.samplerate)

            # shift audio to higher frequencies
            samples = self.pre_emphasis(samples)

            magnitudes, freqs = self.time_to_freq(samples) 

            power_spectrum = self.mag_to_power(magnitudes)

            mel_spectrum = self.power_to_mel(power_spectrum)

            mfccs = self.mels_to_mfccs(mel_spectrum)    

            # setup empty array on first iteration, since we don't know the dimensions before hand
            if(x_train_vec is None):
                x_train_vec = np.empty((0,mfccs.shape[0], mfccs.shape[1]))

            #print(x_train_vec.shape, mfccs.shape, len(samples), wav_file[1])
            x_train_vec = np.vstack((x_train_vec, mfccs.reshape(1,mfccs.shape[0],mfccs.shape[1])))
            y_train_vec.append(wav_file[0])    # target is first element in tuple from enumeration


        #print(x_train_vec.shape)
        #print(len(y_train_vec), y_train_vec)

        return x_train_vec, np.array(y_train_vec, dtype='uint8').reshape(len(y_train_vec), 1)
