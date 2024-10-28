# Kingfisher Bird Chirp Detection with CNN

This project implements a Convolutional Neural Network (CNN) to detect Kingfisher bird chirps from long, noisy audio recordings. Spectrogram intervals from these recordings are transformed into images, which the CNN processes to identify bird chirps with high accuracy.

## Table of Contents
- [Overview](#overview)
- [Data Preparation](#data-preparation)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [Results](#results)

## Overview
Detecting Kingfisher chirps in environmental audio is challenging due to background noise and long audio duration. This CNN model leverages image processing techniques to analyze spectrograms for efficient and reliable detection.

## Data Preparation
- **Dataset**: Contains labeled spectrogram images generated from audio clips.
- **Preprocessing**: Convert audio intervals to spectrograms, normalize, and prepare for CNN input.

## Model Architecture
The model is a CNN, tuned for processing spectrogram images. Key layers include:
- **Convolutional Layers**: Extract features from spectrograms.
- **Pooling Layers**: Reduce spatial dimensions.
- **Fully Connected Layers**: Aggregate features for final chirp classification.

## Training Process
The model is trained on labeled spectrogram images with K-fold cross-validation to ensure robustness and prevent overfitting.

## Results
Achieves high accuracy on test data, effectively distinguishing Kingfisher chirps from noise.
