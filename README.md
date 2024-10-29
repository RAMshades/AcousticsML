# AcousticsML
Acoustic data provide scientific and engineering insights in fields ranging from biology and communications to ocean and Earth science. We survey the recent advances and transformative potential of machine learning (ML), including deep learning, in the field of acoustics. ML is a broad family of techniques, which are often based in statistics, for automatically detecting and utilizing patterns in data. We have ML examples from ocean acoustics, room acoustics, and personalized spatial audio for personalized head-related transfer functions modeling in gaming. For room acoustics, we take room impulse responses (RIR) generation as an example application. For personalized spatial audio, we take head-realted transfer function (HRTF) upsampling as examples. This tutorial covers a wide range of machine learning approaches for acoustic applications including supervised, unsupervised, and deep learning. Although this notebook doesn't cover all the topics, it provides an initial look into applying machine learning for acoustics research.

# Installation 
In order to follow along the examples, you will need to download Anaconda and Python. We have provided a brief outline on how to install Anaconda in [Installation Guide](Python_Installation_instructions.pdf). Once Anaconda has been installed, we will need to create a new environment. This can be done in the conda terminal as: 
```python
conda create -n audioenv
```

go ahead and press y when the dialog pops up. Once this is done you will need to go into your environment using:
```python
conda activate audioenv
```

We need to install a few packages in the environment to use the notebooks. This can be done by using the requirements file in the repository. 

```python
conda install --yes --file requirements. txt
```

# Chapters
The chapters for this repository are ordered as follows: 1) an introduction to signal processing for acoustics; 2) an initial look into feature extraction and selecting features for machine learning models; 3) 

# Chapter 1 - Signal Processing toward Machine Learning
## [Introduction to Signal Processing](Introduction_Signal_Processing.ipynb)

# Chapter 2 - Feature Extraction and Selection
## Feature Extraction
[Feature Extraction](FeatureExtraction.ipynb)

## Feature Selection
[Feature Selection](FeatureSelection.ipynb)

# Chapter 2 - Basic Machine Learning
## Basic Machine Learning 

## 

# Chapter 3 - Deep Learning
## Physics Informed Neural Network
[Forward Propagation NN](PINNs_forward.ipynb)

## Implicit Neural Representation (Neural Field)
[Implicit Neural Representation](Implicit Neural Representation -- HRTF representation learning and interpolation.ipynb)

## Generative Adversarial Network

# chapter 4
## Explainable Artificial intelligence (AI)
[Explainable AI](Explainable AI.ipynb)

## Feature Selection
[Feature Selection](FeatureSelection.ipynb)



# Background

How to install/requirements - Ryan

# Tutorial Breakdown

~~Signal Processing - Ryan~~

~~ Feature Extraction - Ryan ~~

# Basics
Here we can put Regression, Classification, etc.

~~Linear Regression - Neil~~

~~Decision Tree/Random Forest - Ryan~~

~~Logistic Regression/Neural Network/CNN - Neil~~

SVM - Neil

# Unsupervised Learning
Here we can put KNNs, PCA, Hierarchial, autoencoders, etc.

K-Nearest Neighbor/Gaussian Mixture Model - Ryan

Autoencoders/Variational Autoencoder - Neil

PCA/EOF - Ryan

Dictionary Learning - Samuel

# Deep Learning
Here we can put PINNs, NN, GANs, etc.

PINN - Samuel

GAN - Neil

Implicit Neural Representation (Neural Field) - Neil

## Dataset Webpages
https://fishsounds.net/how-to-cite.js

https://www.kaggle.com/datasets?search=audio


## Reference
[PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master)

