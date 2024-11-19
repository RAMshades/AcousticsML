<img src="https://github.com/RAMshades/AcousticsML/blob/main/Acoustics_ML.png" style="float: left;" alt="Machine Learning ini Acoustics" width="200" /> 
Logo | created by ChatGPT

# Tutorial: Machine Learning for Acoustics
### By: Ryan A. McCarthy, Neil Zhang, Samuel Verburg, and Peter Gerstoft

Acoustic data provide scientific and engineering insights in fields ranging from biology and communications to ocean and Earth science. We survey the recent advances and transformative potential of machine learning (ML), including deep learning, in the field of acoustics. ML is a broad family of techniques, which are often based in statistics, for automatically detecting and utilizing patterns in data. We have ML examples from ocean acoustics, room acoustics, and personalized spatial audio. For room acoustics, we take room impulse responses (RIR) generation as an example application. For personalized spatial audio, we take head-realted transfer function (HRTF) upsampling as examples. This tutorial covers a wide range of machine learning approaches for acoustic applications including supervised, unsupervised, and deep learning. Although this notebook doesn't cover all the topics, it provides an initial look into applying machine learning for acoustics research.

# Installation 
## Environment Installation
In order to follow along the examples, you will need to download Anaconda and Python. We have provided a brief outline on how to install Anaconda in [Installation Guide](Python_Installation_instructions.pdf). Once Anaconda has been installed, we will need to create a new environment from the one provided in the .yml file. This can be done in the conda terminal as: 
```python
conda env create -f environment.yml
```

or you can manually create an environment and install certain packages. As an example:

```python
conda create -n audioenv
conda activate audioenv
conda install -c conda-forge librosa scikit-learn  
```

## Additional Installation
### Pytorch
In addition to packages described in the .yml file, you can install PyTorch, Torchvision, TorchAudio for GPU. This can be done by going to [This Website](https://pytorch.org/get-started/locally/) to find the correct package to install. 

### PyRoomAcoustics
For a few of the notebooks, you will also need to install the pyroomacoustics package seen [Here](https://github.com/LCAV/pyroomacoustics). This can be installed using the following line:
```python
pip install pyroomacoustics
pip install python-sofa
```
### Kaggle Datasets
A few of the datasets used in the notebooks require downloading data from the platform [Kaggle](https://www.kaggle.com/). If you do not have an account, please register for an account. Once logged in go to your profile icon in the top right, select settings, and scroll down to API. Please select create a new token and a file "kaggle.json" will download. Place this file within this downloaded directory that contains the Jupyter notebooks above. This API key will grant access for the opendatasets package to download the data (seen in the Jupyter notebooks). Data downloaded through opendatasets can be downloaded once into your directory and will not duplicate a download unless forced. 


# Chapters
The chapters for this repository are ordered as follows: 1) an introduction to signal processing for acoustics; 2) an initial look into feature extraction and selecting features for machine learning models; 3) unsupervised machine learning approaches; 4) supervised machine learning approaches; 5) deep learning models examples; 6) explainable AI and feature importance.

## Chapter 1 - Signal Processing toward Machine Learning
### 1.1 [Introduction to Signal Processing](1_1_Introduction_Signal_Processing.ipynb)
Brief overview of signal processing and techniques that are useful for processing acoustic data.

## Chapter 2 - Feature Extraction and Selection
### 2.1 [Feature Extraction](2_1_FeatureExtraction.ipynb)
Descriptions of features and an introduction to feature extraction approaches for machine learning.

### 2.2 [Feature Selection](2_2_FeatureSelection.ipynb)
Feature selection aims to improve complexity of models, reduce training time, or improve performance of machine learning models. This notebook talks through how to perform feature selection through an example of picking out major vs minor chords.

## Chapter 3 - Unsupervised Machine Learning
### 3.1 [Unsupervised Approaches](3_1_Unsupervised_Learning--Long_Timeseries.ipynb)
Given a long time series, how can we quickly segment frames of a time series to find similarities in the acoustic sound.

### 3.2 [Principal Component Analysis](3_2_PCA--Creating_Sound.ipynb)
Principal component analysis is discussed and demonstrated to construct new guitar sounds through the frequency domain.

### 3.3 [Dictionary Learning](3_3_Dictionary_Learning.ipynb)
Dictionary Learning for the sparse representation of room impulse responses and bandwidth extension. 

### 3.4 [Autoencoder|Variational Autoencoder](3_4_AE_VAE--Anomalous_Sound_Detection.ipynb)
Autoencoder and VAE for machine sound anomaly detection.

## Chapter 4 - Supervised Machine Learning
### 4.1 [Linear Regression](4_1_Linear_Regression--Predict_Reverberation_time.ipynb)
Linear regression for the use case of predicting the room reverberation time.

### 4.2 [Decision Tree and Random Forest](4_2_DT_RF--Number_Identification.ipynb)
Classify AudioMNIST dataset through decision trees and random forests to distinguish numbers from 0 to 9.

## Chapter 5 - Deep Learning

### 5.1 [Neural Network and Convolutional Neural Networks](5_1_LR_NN_CNN--Audio_Classification.ipynb)
Targeting audio classification problem, we introduce the classical logistic regression approach and basics of deep learning with a simple neural network and a convolutional neural network.

### 5.2 [Generative Adversarial Networks (GAN)](5_2_Generative_Adversarial_Network--Room_Impulse_Response_Generation.ipynb)
Generative Adversarial Network for generating room impulse responses (RIRs).

### 5.3 [Implicit Neural Representation](5_3_Implicit_Neural_Representation--HRTF_Representation_Learning_and_Interpolation.ipynb)
Implicit Neural Representation for representing personalized Head-Related Transfer Function (HRTF). 

### 5.4 [Forward Propagation Physics Informed Neural Network](5_4_PINNs_forward.ipynb)
Solve the wave equation using Physics Informed Neural Networks (forward problem). 

### 5.5 [Inverse Propagation Physics Informed Neural Network](5_5_PINNs_inverse.ipynb)
Estimate the wave speed using Physics Informed Neural Networks (inverse problem).

## Chapter 6 - Explainable AI
### 6.1 [Unsupervised Models](6_1_ExplainableAI-Unsupervised.ipynb)
Identify number of clusters using a variety of techniques and interpret the clustering of the feature space.

### 6.2 [Supervised Models](6_2_ExplainableAI-Supervised.ipynb)
Determine features importance for supervised models and evaluate performance to determine where pitfalls may occur.

### 6.3 [Deep Learning](6_3_ExplainableAI-DeepLearning.ipynb)
Visualize key distinctions in model activations for given inputs and evaluate how deep learning model's interpret observe and interpret data.

## Reference
[PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master)

OpenAI. (2024). ChatGPT - Image Generator by Naif J Alotaibi (Nov. 12 2024)[Large language model]. https://chat.openai.com/chat 

