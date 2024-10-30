# AcousticsML
Acoustic data provide scientific and engineering insights in fields ranging from biology and communications to ocean and Earth science. We survey the recent advances and transformative potential of machine learning (ML), including deep learning, in the field of acoustics. ML is a broad family of techniques, which are often based in statistics, for automatically detecting and utilizing patterns in data. We have ML examples from ocean acoustics, room acoustics, and personalized spatial audio. For room acoustics, we take room impulse responses (RIR) generation as an example application. For personalized spatial audio, we take head-realted transfer function (HRTF) upsampling as examples. This tutorial covers a wide range of machine learning approaches for acoustic applications including supervised, unsupervised, and deep learning. Although this notebook doesn't cover all the topics, it provides an initial look into applying machine learning for acoustics research.

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

IF the above packages don't insall, it may be due to PyTorch. To fix this, delete PyTorch, Torchvision, TorchAudio from the .yml and go to [This Website](https://pytorch.org/get-started/locally/) to find the correct package to install. Additionally, you will need the pyroomacoustics package which can be installed through the line below:

```python
pip install pyroomacoustics
```

# Chapters
The chapters for this repository are ordered as follows: 1) an introduction to signal processing for acoustics; 2) an initial look into feature extraction and selecting features for machine learning models; 3) unsupervised machine learning approaches; 4) basic machine learning 

# Chapter 1 - Signal Processing toward Machine Learning
## [Introduction to Signal Processing](Introduction_Signal_Processing.ipynb)

# Chapter 2 - Feature Extraction and Selection
## [Feature Extraction](FeatureExtraction.ipynb)

## [Feature Selection](FeatureSelection.ipynb)

# Chapter 3 - Unsupervised Learning
## [Unsupervised Approaches](<Unsupervised Learning -- Long Timeseries.ipynb>)

## [Principal Component Analysis](<PCA -- Creating Sound.ipynb>)

## [Dictionary Learning](dictionary_learning.ipynb)

# Chapter 4 - Supervised Machine Learning
## [Linear Regression](<Linear regression -- Predict the reverberation time.ipynb>)

## [Decision Tree and Random Forest](<DT_RF -- Number Identification .ipynb>)

# Chapter 5 - Deep Learning
## [Neural Representation (HRTF)](<Implicit Neural Representation -- HRTF representation learning and interpolation.ipynb>)

## [Neural Network and Convolutional Neural Networks](<LR|NN|CNN -- Audio Classification.ipynb>)

## [Forward Propagation Physics Informed Neural Network](PINNs_forward.ipynb)

# Chapter 6 - Explainable AI
## [Interpreting Models](<Explainable AI.ipynb>)



## Dataset Webpages
https://fishsounds.net/how-to-cite.js

https://www.kaggle.com/datasets?search=audio


## Reference
[PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master)

