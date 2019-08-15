[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/AudioRNN/blob/master/LICENSE.txt)
# Tensorflow speech recognition challenge

  - Purpose
  
  >><p>Inquire and explore steps in the data scientist pipeline including: data wrangling, data cleaning, and predictive modeling with AI. We   will use the Speech Recognition data set, found on Kaggle.com, in order to build an algorithm that recognizes simple speech commands. </p>

  - Objective
  
  >><p>Learn Command Words: Yes, No, Up, Down, Left, Right, On, Off, Stop, Go</p>

# Dataset Generation
  Downloads Tensorflow Speech Recognition Dataset, extracts MFCCs as features, and saves vectors to .h5 file
  
 - Usage

   <p>python3 SRData.py -o </p>

 - Requirements
    - h5py
    - pickle
    - numpy
    - soundfile
    - tqdm
    - librosa
    - matplotlib
    - scipy
  <br></br>
  
  ## Feature Extraction
 - Raw Samples
   ![alt_text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/AudioWave.png)
 - Frequency Domain
   ![alt_text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/Spectrum.png)
 - Power Spectrum
   ![alt_text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/PowerSpectrum.png)
 - Mel Filters
   ![alt_text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/MelFilters.png)
 - Mel Filter Banks
   ![alt_text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/MelSpectra.png)
 - Mel-frequency cepstral coefficients (MFCCs)
   ![alt_text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/MFCCcoefficients.png)
   
   
# Training
  Trains the model
  
 - Usage

   <p>python3 SRnn.py</p>

 - Requirements
    - h5py
    - pickle
    - numpy
    - sklearn
    - keras
    - matplotlib

  <br></br>
 - Convolutional Model
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/CNNmodel.png "CNN Model")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/AccuracyCNN.png "Training and Validation Accuracy")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/LossCNN.png "Training and Validation Loss")
  
 - Recurrent Model (LSTM)
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/RNNmodel.png "RNN Model")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/AccuracyRNN.png "Training and Validation Accuracy")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/LossRNN.png "Training and Validation Loss")

# Evaluation
  Plots a trained model's history (accuracy/loss)
  
 - Usage

   <p>python3 SRPlots.py &ltmodel_history_file&gt</p>

 - Requirements
    - pickle
    - matplotlib
  <br></br>
  
 - Convolutional Model
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/AccuracyCNN.png "Training and Validation Accuracy")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/LossCNN.png "Training and Validation Loss")
  
 - Recurrent Model (LSTM)
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/AccuracyRNN.png "Training and Validation Accuracy")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/LossRNN.png "Training and Validation Loss")
  
# Demo
  Runs test demo that allows a user to record a word, extracts the features (MFCCs), uses model to predict the word, uses Google Text To   Speech to play back the model's prediction
  
 - Usage

   <p>python3 SRTest.py &ltmodel_file&gt</p>

 - Requirements
    - sounddevice
    - pickle
    - numpy
    - playsound
    - keras
    - gtts
    - tqdm
    - scipy
    <br></br>
    
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/Recording.png "Demo Program")
  
# Presentation
  - [Research Slides](https://docs.google.com/presentation/d/1Y0GeGzcjNZaEwUKqFNz6XXiSq94oDpOGp4fgnUxdDTg/edit?usp=sharing)
  - [Project Slides](https://docs.google.com/presentation/d/1yGEzSyJ9kr97wDnjVuW-BfNnlgVCIJRGiUEYGF7TBjI/edit?usp=sharing)
    
# Resources
  - [Tensorflow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
  - [Tensorflow Command Word Dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
  - [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
  - [Keras Conv1D: Working with 1D Convolutional Neural Networks in Keras](https://missinglink.ai/guides/deep-learning-frameworks/keras-conv1d-working-1d-convolutional-neural-networks-keras/)
  - [Time Series Classification with CNNs](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)
  - [A Beginner's Guide to LSTMs and Recurrent Neural Networks](https://skymind.com/wiki/lstm)
  
  \* This repo is under MIT License, use as you please.
