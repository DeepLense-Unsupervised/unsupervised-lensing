import matplotlib.pyplot as plt

def plot_loss(x):
    plt.plot(x)
    plt.xlabel('epochs')
    plt.ylabel('MSE Loss')
    plt.title('Loss Convergence Plot')
    plt.show()






