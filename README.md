# Unsupervised Lensing
A PyTorch-based tool for Unsupervised Deep Learning applications in strong lensing cosmology

This is a Google Summer of Code (GSoC) 2020 project

### Project Description

Gravitational lensing has been a cornerstone in many cosmology experiments, and studies since it was discussed in Einstein’s calculations back in 1936 and discovered in 1979, and one area of particular interest is the study of substructure in strong lensing images and differentiating weakly interacting massive particle (WIMP) dark matter. While statistical and supervised machine learning algorithms have been implemented for this task, the potential of unsupervised deep learning algorithms is yet to be explored and could prove to be crucial in the analysis of LSST data. The primary aim of this GSoC 2020 project is to design a python-based framework for implementing unsupervised deep learning architectures to study strong lensing images. 

## Models

Currently supported models:

* Adversarial Autoencoder
* Convolutional Variational Autoencoder
* Deep Convolutional Autoencoder
* Restricted Boltzmann Machine

## Installation

```shell
pip install unsupervised-lensing
```

## Data

The models expect the data to be in the form of a five-dimensional NumPy array of shape [number_of_batches, batch_size, number_of_channels, height, width]. You can use the data preprocessing module in unsupervised_lensing.utils to prepare your dataset.

## Examples

[Example Notebook](https://github.com/DeepLense-Unsupervised/unsupervised-lensing/blob/master/Example_Notebook.ipynb)

### Training

```python
from unsupervised_lensing.models import Adversarial_AE
from unsupervised_lensing.models.AAE_Nets import *
from unsupervised_lensing.utils import loss_plotter as plt

# Model Training
out = Adversarial_AE.train(data_path='./Data/no_sub_train.npy',   # Path to training dataset
                             epochs=100,
                             checkpoint_path='./Weights',         # Path to store model weights
                             pretrain=True)                       # Set True for transfer learning

# Plot the training loss
plt.plot_loss(out)
```

### Inference

```python
from unsupervised_lensing.models import Adversarial_AE
from unsupervised_lensing.models.AAE_Nets import *
from unsupervised_lensing.utils import loss_plotter as plt
from unsupervised_lensing.utils.EMD_Lensing import EMD

# Model Validation
recon_loss = Adversarial_AE.evaluate(data_path='./Data/no_sub_test.npy',   # Path to validation dataset
                                       checkpoint_path='./Weights',        # Path to model weights
                                       out_path='./Results')               # Path to store reconstructed samples

# Plot the reconstruction loss
plt.plot_dist(recon_loss)

# Calculate Wasserstein distance
print(EMD(data_path='./Data/no_sub_test.npy', recon_path='./Results/Recon_samples.npy'))
```

