# Unsupervised Lensing
A PyTorch-based tool for Unsupervised Deep Learning applications in strong lensing cosmology

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

# Model Validation
recon_loss = Adversarial_AE.evaluate(data_path='./Data/no_sub_test.npy',   # Path to validation dataset
                                       checkpoint_path='./Weights',        # Path to model weights
                                       out_path='./Results')               # Path to store reconstructed samples

# Plot the reconstruction loss
plt.plot_dist(recon_loss)
```

