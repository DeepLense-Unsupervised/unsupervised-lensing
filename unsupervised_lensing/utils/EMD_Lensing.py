import numpy as np
import ot
from scipy.spatial import distance

def EMD(data_path='./Data/no_sub_test.npy', recon_path='./Results/Recon_samples.npy'):

    '''
    
    Args:
    ______
    
    data_path: str
       Path to your input NumPy array of shape [number_of_batches, batch_size, number_of_channels, height, width]
    
    recon_path: str
        Path to store output reconstructed lenses
    '''

    data = np.load(data_path, allow_pickle='True')
    recon_data = np.load(recon_path, allow_pickle='True')

    data_temp = []
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            data_temp.append(data[i][j])
    data = np.asarray(data_temp)

    print(data.shape)
    print(recon_data.shape)

    EMD = 0
    for i in range(data.shape[0]):
        
        data_1D = np.sum(data[i][0], axis=0)
        recon_1D = np.sum(recon_data[i][0], axis=0)
        cost_matrix = distance.cdist(data_1D.reshape(-1,1), recon_1D.reshape(-1,1), 'euclidean')
        EMD += ot.emd2([],[],cost_matrix)
        
    return EMD/data.shape[0]

