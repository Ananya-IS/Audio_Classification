# Audio Signal Classification with CNN
## Introduction
This project aims to classify animal sounds using a Convolutional Neural Network (CNN). The model extracts features from audio signals, particularly using Mel-Frequency Cepstral Coefficients (MFCC), and classifies them into different animal categories.
## Overview
Environmental sound classification is a growing research area with numerous real-world applications, such as multimedia indexing, assistance for the deaf, smart home security, automotive safety, and industrial predictive maintenance. This project focuses on classifying sounds from five different animals: Dog, Cat, Cow, Hen, and Frog.
## Installation
### Requirements
Python 3.8+
TensorFlow
Keras
Librosa
NumPy
Scikit-learn
Sounddevice (for real-time monitoring)
### Dataset
The dataset consists of 200 sound files (40 for each animal). The sound excerpts are in .wav format and are sampled at 44.1 kHz with a 16-bit depth. Each file is labeled with its corresponding animal category.

## Usage
Preprocessing
Load the dataset and preprocess it using Librosa. Extract MFCCs for each audio file.

Running the Model
To train the model:

python train_model.py

To classify an audio file:

python classify.py --input audiofile.wav

Real-Time Monitoring
To use real-time audio classification:

python realtime_classify.py

## Model Architecture
The CNN model has four convolutional layers followed by max-pooling layers. The final layer is a dense output layer with a softmax activation function. The model was trained on the extracted MFCCs with the following key configurations:

Conv2D layers with filters of size 16, 32, 64, and 128
Kernel size: 2x2
Activation function: ReLU
Dropout: 20%
GlobalAveragePooling2D before the final dense layer
Output layer with softmax activation

## Results
The classifier performs well on the test dataset, accurately classifying animal sounds. However, misclassifications can occur between similar classes, such as Cat and Dog, especially during real-time monitoring due to noise or unclear sounds.

### Contributing
Contributions are welcome! Please submit any issues or pull requests to help improve this project.
