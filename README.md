[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/mpc6/AudioRNN/blob/master/LICENSE.txt)
# Tensorflow speech recognition challenge

  - Purpose
  
  Inquire and explore steps in the data scientist pipeline including: data wrangling, data cleaning, and predictive modeling with AI. We   will use the Speech Recognition data set, found on Kaggle.com, in order to build an algorithm that recognizes simple speech commands. 

  - Objective
  
  Learn Command Words:
    Yes, No, Up, Down, Left, Right, On, Off, Stop, Go

# Data
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
 - Convolutional Model
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/CNNmodel.png "CNN Model")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/AccuracyCNN.png "Training and Validation Accuracy")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/LossCNN.png "Training and Validation Loss")
  
 - Recurrent Model (LSTM)
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/RNNmodel.png "RNN Model")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/AccuracyRNN.png "Training and Validation Accuracy")
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/LossRNN.png "Training and Validation Loss")
  
# Evaluation
  ![alt text](https://github.com/mpc6/SpeechRec/blob/master/DataVisualizationImages/Recording.png "Demo Program")
  
# Presentation
  - [Project Slides](https://docs.google.com/presentation/d/1yGEzSyJ9kr97wDnjVuW-BfNnlgVCIJRGiUEYGF7TBjI/edit?usp=sharing)
    
# Resources
  - [Tensorflow Speech Recognition Challenge](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge)
  - [Tensorflow Command Word Dataset](http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz)
  - [Speech Processing for Machine Learning: Filter banks, Mel-Frequency Cepstral Coefficients (MFCCs) and What's In-Between](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
  - [Keras Conv1D: Working with 1D Convolutional Neural Networks in Keras](https://missinglink.ai/guides/deep-learning-frameworks/keras-conv1d-working-1d-convolutional-neural-networks-keras/)
  - [Time Series Classification with CNNs](https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/)
  - [A Beginner's Guide to LSTMs and Recurrent Neural Networks](https://skymind.com/wiki/lstm)
  
  \* This repo is under MIT License, use as you please.
