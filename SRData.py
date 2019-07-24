import os, sys, time
import glob
import h5py
import argparse
import subprocess
import numpy as np
import soundfile as sf
from tqdm import tqdm
from librosa.core import resample
from SRDataGenerator import DataGenerator

def verify_words(words, datadir):

    for word in words:
        if os.path.isdir(os.path.join(datadir, word)):
            print(f'Using word "{word}"')
        else:
            print(f'Word "{word}" not found in dataset')
            sys.exit('Failed to build dataset')

    print('All words verified in dataset')


# Download raw data from source and uncompress in data directory
def download_data(datafile, datadir):

    file_name = os.path.basename(datafile)
    print('Downloading', file_name)
    rtnVal = subprocess.call(['wget', datafile, '-P', datadir])
    assert rtnVal == 0, 'downloaded failed!'
    print(file_name, 'downloaded successfully')
    
    print('Uncompressing', file_name)
    rtnVal = subprocess.call(['tar', '-C', datadir, '-zxvf', os.path.join(datadir, file_name)])
    assert rtnVal == 0, 'file failed to uncompress!'
    print(file_name, 'uncompressed successfully')


# save data at correct samplerate 
def set_samplerate(datadir, words, samplerate):

    # only check samplerate for files in word list
    wav_files = [f for word in words for f in glob.glob(os.path.join('data', word,'*.wav'))]

    print(f'Verifying data is at {samplerate} Hz')
    for wav_file in tqdm(wav_files):
        #print('Checking Samplerate for file:', wav_file)
        samples, sr = sf.read(wav_file)
        
        if sr != samplerate:
            #print('Resampling file ', wav_file, 'from', sr, 'to', samplerate)
            samples = resample(samples, sr, samplerate)
            sf.write(wav_file, samples, samplerate)    
        
    print(f'All Data sampled at {samplerate} Hz')
   
 
def setup_data(datafile, datadir, words, other_words, samplerate):
 
    file_name = os.path.basename(datafile)
    
    if not os.path.exists(datadir):
        os.makedirs(datadir)

    if not os.path.exists(os.path.join(datadir, file_name)):
        download_data(datafile, datadir)

    verify_words(words, datadir)

    set_samplerate(datadir, words, samplerate) 
    set_samplerate(datadir, other_words, samplerate) 

   
def save_dataset_to_hdf5(datasetfile, x_train_vec, y_train_vec):

    print(f'Saving dataset to {datasetfile}')
    with h5py.File(datasetfile, 'w') as hf:
        hf.create_dataset('x_train', data = x_train_vec, chunks=True)
        hf.create_dataset('y_train', data = y_train_vec, chunks=True)


def load_dataset_from_hdf5(datasetfile):

    print(f'Loading dataset from {datasetfile}')
    with h5py.File(datasetfile, 'r') as hf:
        x_train = hf['x_train']
        y_train = hf['y_train']

        return np.array(x_train, dtype='float32'), np.array(y_train, dtype='uint8')
        
       
def get_other_words(words, datadir):
    word_dirs = [os.path.basename(dir) for dir in glob.glob('data/*') if os.path.isdir(dir)] 
    other_words = set(word_dirs) - set(words)
    return list(other_words)

def main():
    print(f"Speech Recognition Data Generation starting at {time.ctime()}")
    script_dir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument("-dd", "--datadir", help="root directory of data", default="data")
    parser.add_argument("-ds", "--datasetfile", help="HDF5 file for dataset", default="SRData.h5")
    parser.add_argument("-w", "--words", help="specifies which words to include in data set: <--wordss <word1, word2, ...>", nargs='+', default=['yes','no'])
    parser.add_argument("-sr", "--samplerate", help="audio sample rate", type=int, default=8000)
    parser.add_argument("-pe", "--preemphasis", help="preemphasis coeffecient", type=int, default=.97)
    parser.add_argument("-fs", "--framesize", help="framesize, audio context in FFT window", type=int, default=.024)
    parser.add_argument("-ws", "--windowsize", help="windowsize, num samples for FFT", type=int, default=512)
    parser.add_argument("-mf", "--nummelfilters", help="number of melfilters", type=int, default=40)
    parser.add_argument("-mfcc", "--nummfccs", help="number of MFCCs to keep", type=int, default=10)
    parser.add_argument("-sg", "--showgraphs", help="show graphs for each feature transformation", action="store_true")
    parser.add_argument("-df", "--datafile", help="URL for dataset", default="http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz")
    arg = parser.parse_args()

    datadir = os.path.join(script_dir, arg.datadir)
    print(f'Using directory {datadir} as data location')

    datafile = arg.datafile
    datasetfile = arg.datasetfile
    words = arg.words
    print(f' Using words {words}')
    # any words not specified should be used as training 'other' category
    other_words = get_other_words(words, datadir)
    print(f' Using words {other_words} as unknown words')
    samplerate = arg.samplerate
    preemphasis = arg.preemphasis
    framesize = arg.framesize
    windowsize = arg.windowsize
    num_melfilters = arg.nummelfilters
    num_mfccs = arg.nummfccs
    showgraphs = arg.showgraphs

    # Gets data from source and converts to specified samplerate
    #setup_data(datafile, datadir, words, other_words, samplerate)

    # Generates dataset from wav files containing specified words 
    data_generator = DataGenerator(datadir, words, other_words, samplerate, preemphasis, framesize, windowsize, num_melfilters, num_mfccs)
    x_train_vec, y_train_vec = data_generator.convert_wavs_to_dataset(showgraphs)

    #print(x_train_vec.shape, y_train_vec.shape)

    # Saves dataset to disk
    save_dataset_to_hdf5(datasetfile, x_train_vec, y_train_vec)

    # Loads dataset from disk
    x_train, y_train = load_dataset_from_hdf5(datasetfile)

    #print(x_train.shape, y_train.shape)
    # Verifies consistance of dataset generated and saved
    assert x_train_vec.shape == x_train.shape and y_train_vec.shape == y_train.shape, "Mismatch between generated data and data saved to disk!"    

    print(f'Dataset Generation Complete at {time.ctime()}')

if __name__ == '__main__':
    main()
