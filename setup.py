from setuptools import find_packages, setup
setup(
  name = 'unsupervised_lensing',
  version = '0.1.2',
  license='MIT',
  description = 'A PyTorch-based tool for Unsupervised Deep Learning applications in strong lensing cosmology',
  author = 'K Pranath Reddy',
  author_email = 'pranath.mail@gmail.com',
  url = 'https://github.com/DeepLense-Unsupervised/unsupervised-lensing',
  packages=find_packages(include=['unsupervised_lensing',
  'unsupervised_lensing.models',
  'unsupervised_lensing.utils',
  ]),
  keywords = ['Gravitational Lensing', 'Unsupervised Deep Learning', 'Dark Matter'],
  install_requires=[            
          'numpy==1.18.5',
          'cython==0.29.21',
          'scipy==1.4.1',
          'POT==0.7.0',
          'tqdm==4.47.0',
          'matplotlib==3.2.2',
          'seaborn==0.10.1',
          'torch==1.6.0',
          'torchvision==0.7.0',
          'googledrivedownloader==0.4.0',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: MIT License',
  ],
)
