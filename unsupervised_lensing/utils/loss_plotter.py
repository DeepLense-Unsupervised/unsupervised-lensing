import matplotlib.pyplot as plt
import seaborn as sns
import math
sns.set()

def plot_loss(x):
    plt.plot(x)
    plt.xlabel('epochs')
    plt.ylabel('MSE Loss')
    plt.title('Loss Convergence Plot')
    plt.show()
    
def plot_dist(x, hist_width=0.000005):
    n = math.ceil((x.max() - x.min())/hist_width)
    sns.distplot(x,kde=0,norm_hist=0, color="g", bins = n)
    plt.xlabel('MSELoss')
    plt.ylabel('Population')
    plt.title('Distribution of Reconstruction Loss for test samples')
    plt.show()






