from unsupervised_lensing.models import Convolutional_AE
from unsupervised_lensing.models.Convolutional_AE import DCA
from unsupervised_lensing.utils import loss_plotter as plt

out = Convolutional_AE.train(data_path='./Data/no_sub_train.npy',
                  epochs=10,
                  checkpoint_path='./Weights',
                  pretrain=True)
                  
plt.plot_loss(out)
                  






