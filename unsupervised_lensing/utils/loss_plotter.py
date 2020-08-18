import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()

def plot_loss(x):

    '''
    
    Args:
    ______
    
    x: []
       array containing loss values
    '''
    
    plt.plot(x)
    plt.xlabel('epochs')
    plt.ylabel('MSE Loss')
    plt.title('Loss Convergence Plot')
    plt.show()
    
def plot_dist(x, bin_width=0.000005):

    '''
    
    Args:
    ______
    
    x: []
       array containing loss values
       
    bin_width: float
    '''
    
    n = math.ceil((x.max() - x.min())/bin_width)
    sns.distplot(x,kde=0,norm_hist=0, color="g", bins = n)
    plt.xlabel('MSELoss')
    plt.ylabel('Population')
    plt.title('Distribution of Reconstruction Loss for test samples')
    plt.show()






