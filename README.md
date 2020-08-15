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

