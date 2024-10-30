# AcousticsML
Acoustic data provide scientific and engineering insights in fields ranging from biology and communications to ocean and Earth science. We survey the recent advances and transformative potential of machine learning (ML), including deep learning, in the field of acoustics. ML is a broad family of techniques, which are often based in statistics, for automatically detecting and utilizing patterns in data. We have ML examples from ocean acoustics, room acoustics, and personalized spatial audio. For room acoustics, we take room impulse responses (RIR) generation as an example application. For personalized spatial audio, we take head-realted transfer function (HRTF) upsampling as examples. This tutorial covers a wide range of machine learning approaches for acoustic applications including supervised, unsupervised, and deep learning. Although this notebook doesn't cover all the topics, it provides an initial look into applying machine learning for acoustics research.

# Installation 
In order to follow along the examples, you will need to download Anaconda and Python. We have provided a brief outline on how to install Anaconda in [Installation Guide](Python_Installation_instructions.pdf). Once Anaconda has been installed, we will need to create a new environment from the one provided in the .yml file. This can be done in the conda terminal as: 
```python
conda env create -f environment.yml
```

or you can manually create an environment and install certain packages. As an example:

```python
conda create -n audioenv
conda activate audioenv
conda install -c conda-forge scikit-learn  
```

## Additional Installation
### Pytorch
In addition to packages described in the .yml file, you will also need to install PyTorch, Torchvision, TorchAudio. This can be done by going to [This Website](https://pytorch.org/get-started/locally/) to find the correct package to install. 

### PyRoomAcoustics
For a few of the notebooks, you will also need to install the pyroomacoustics package seen [Here](https://github.com/LCAV/pyroomacoustics). This can be installed using the following line:
```python
pip install pyroomacoustics
```

# Chapters
The chapters for this repository are ordered as follows: 1) an introduction to signal processing for acoustics; 2) an initial look into feature extraction and selecting features for machine learning models; 3) unsupervised machine learning approaches; 4) supervised machine learning approaches; 5) deep learning models examples; 6) explainable AI and feature importance.

# Chapter 1 - Signal Processing toward Machine Learning
## [Introduction to Signal Processing](Introduction_Signal_Processing.ipynb)

# Chapter 2 - Feature Extraction and Selection
## [Feature Extraction](FeatureExtraction.ipynb)

## [Feature Selection](FeatureSelection.ipynb)

# Chapter 3 - Unsupervised Machine Learning
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


## Reference
[PyTorch Tutorial](https://github.com/yunjey/pytorch-tutorial/tree/master)

