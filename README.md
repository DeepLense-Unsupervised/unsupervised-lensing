# Unsupervised Lensing
A PyTorch-based tool for Unsupervised Deep Learning applications in strong lensing cosmology

## Models

Currently supported models:

* Adversarial Autoencoder
* Convolutional Variational Autoencoder
* Deep Convolutional Autoencoder
* Restricted Boltzmann Machine

## Data

The models expect the data to be in the form of a five-dimensional NumPy array of shape [number_of_batches, batch_size, number_of_channels, height, width]. You can use the data preprocessing module in unsupervised_lensing.utils to prepare your dataset.

## Examples

### Training

```python
from unsupervised_lensing.models import Adversarial_AE
from unsupervised_lensing.models.AAE_Nets import *
from unsupervised_lensing.utils import loss_plotter as plt

# Model Training
out = Adversarial_AE.train(data_path='./Data/no_sub_train.npy',
                             epochs=100,
                             checkpoint_path='./Weights',
                             pretrain=True)

# Plot the training loss
plt.plot_loss(out)

```

