from unsupervised_lensing.models import Adversarial_AE
from unsupervised_lensing.models.Adversarial_AE import Encoder,Decoder,Discriminator
from unsupervised_lensing.utils import loss_plotter as plt

# Model Training
out = Adversarial_AE.train(data_path='./Data/no_sub_train.npy',
                             epochs=10,
                             checkpoint_path='./Weights',
                             pretrain=True)
                  
plt.plot_loss(out)

# Model Validation
recon_loss = Adversarial_AE.evaluate(data_path='./Data/no_sub_test.npy',
                                       checkpoint_path='./Weights',
                                       out_path='./Results',
                                       pretrain=False)
                  
plt.plot_dist(recon_loss)





