from unsupervised_lensing.models import RBM_Model
from unsupervised_lensing.models.RBM_Model import RBM
from unsupervised_lensing.utils import loss_plotter as plt

# Model Training
out = RBM_Model.train(data_path='./Data/no_sub_train.npy',
                             epochs=10,
                             checkpoint_path='./Weights',
                             pretrain=True)
                  
plt.plot_loss(out)

# Model Validation
recon_loss = RBM_Model.evaluate(data_path='./Data/no_sub_test.npy',
                                       checkpoint_path='./Weights',
                                       out_path='./Results',
                                       pretrain=False)
                  
plt.plot_dist(recon_loss)





