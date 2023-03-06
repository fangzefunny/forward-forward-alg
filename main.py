'''
Reference: https://github.com/ljj7975/forward-forward-algorithm/blob/main/supervised_ffa_mnist.ipynb
'''
import os 
import numpy as np

import torch 
import torch.nn as nn 
from torch.optim import Adam

from tqdm import tqdm

import seaborn as sns 
import matplotlib.pyplot as plt 

from utils.get_data import get_mnist
from utils.viz import viz 
viz.get_style()

pth = os.path.dirname(os.path.abspath(__file__))

# -------  Basic  ------- #

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def add_label_to_img(x_batch, y_batch, nClass):
    '''Combine label and the images
    '''
    x_aug   = x_batch.clone()
    max_val = x_batch.max()
    x_aug[:, :nClass] = torch.eye(nClass)[:, y_batch].T * max_val
    return x_aug

def show_img(x_batch, labels=None, preds=None):
    fig, axs = plt.subplots(3, 3, figsize=(10, 10), 
                    sharex=True, sharey=True)
    for i in range(9):
        x = x_batch[i, :]
        ax = axs[i//3, i%3]
        ax.imshow(x.reshape(28, 28), cmap='Greys')
        title_str = ''
        if labels is not None:
            title_str += f'Label: {labels[i]}'
        if preds is not None:
            title_str += f', Pred: {preds[i]}'
        ax.set_title(title_str)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()

# ------  Architecture ----- #

class FFlinear(nn.Linear):
    '''A forward-forward layer
    '''

    def __init__(self, in_dim, out_dim, bias=True, device=None, lr=0.03):
        super().__init__(in_dim, out_dim, bias, device)
        # threshold for classify inputs as  
        # postive and negative data 
        self.theta = 2.
        self.optim = Adam(self.parameters(), lr=lr)

    def forward(self, x):
        '''Forward the data

        y = relu(w @ x + self.b)
        '''
        return torch.relu(x@self.weight.T
                        +self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg, maxEpoch=1000):
        '''Train the network

        Sec. 2: the aim of the learning is to make
        the goodness be well above the threshold for 
        real data and well below it for negative data

            G(pos) = σ(∑j yj^2 - θ)
            G(neg) = σ(θ - ∑j yj^2)
        '''
        for _ in tqdm(range(maxEpoch)):
            pos2 = self.forward(x_pos).square().sum(1)
            neg2 = self.forward(x_neg).square().sum(1)
            # push the pos (neg) samples to values
            # larger (smaller) than theta
            # this is a LOCAL objective function 
            G_pos = torch.sigmoid(pos2 - self.theta)
            G_neg = torch.sigmoid(self.theta - neg2)
            # take negative for the gradient descent
            loss = -(G_pos + G_neg).mean()
            # LOCAL derivative to train the weight 
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
        
        x_pos = self.forward(x_pos).detach()
        x_neg = self.forward(x_neg).detach() 
        return x_pos, x_neg 


class FFnet(nn.Module):
    '''Multi-layer forward-forward network
    '''

    def __init__(self, dims, nClass):
        super().__init__()
        self.nClass = nClass
        self.layers = [FFlinear(dims[d], dims[d+1], device=device) 
                            for d in range(len(dims) - 1)]

    def train(self, x_pos, x_neg, maxEpoch=1000):
        '''Train the multi-layer network

        Train the network using forward-forward algorithm
        layer by layer. Before entering the next layer, 
        the output is normalized to eliminate the effect of
        vector length.
        '''
        for i, layer in enumerate(self.layers):
            print(f'Training layer {i}')
            # normalize the output
            x_pos = x_pos / x_pos.norm(dim=1)
            x_neg = x_neg / x_neg.norm(dim=1)
            # train the network
            x_pos, x_neg = layer.train(x_pos, x_neg, maxEpoch=maxEpoch)

    def predict(self, x_batch):
        '''Predict the label for images
        '''
        goodness_per_label = [] 
        for label in range(self.nClass):
            h = add_label_to_img(x_batch, y_batch=label, 
                                    nClass=self.nClass)
            goodness = 0
            for layer in self.layers:
                h = layer.forward(h)
                goodness += h.square().mean(1)
            goodness_per_label += [goodness.unsqueeze(1)]
        goodness_per_label = torch.hstack(goodness_per_label)
        return goodness_per_label.argmax(-1)            

if __name__ == '__main__':

    # some variables 
    torch.manual_seed(1234)
    nClass = 10 

    # get data, use "if_download=True" for the first time 
    # using it
    trainLoader, testLoader = get_mnist(if_download=False)

    # create positive data, and negative data 
    x_batch, y_batch = next(iter(trainLoader))
    # positive data 
    x_pos = add_label_to_img(x_batch.view(-1, 28*28), y_batch, nClass=nClass)
    show_img(x_pos, labels=y_batch)
    plt.savefig(f'{pth}/figures/show_x_pos', dpi=300)
    plt.close()
    # negative data 
    ind = torch.randperm(x_batch.shape[0])
    y_neg = y_batch[ind]
    x_neg = add_label_to_img(x_batch.view(-1, 28*28), y_neg, nClass=nClass)
    show_img(x_neg, labels=y_neg)
    plt.savefig(f'{pth}/figures/show_x_neg', dpi=300)
    plt.close()

    # instantiate a model and train 
    model = FFnet(dims=[784, 500, 500], nClass=10)
    model.train(x_pos, x_neg)
    acc_train = model.predict(x_batch.view(-1, 28*28)
                    ).eq(y_batch).float().mean().item()
    print(f'Train Acc.: {acc_train:.4f}')

    # predict the testing data 
    x_test, y_test = next(iter(testLoader))
    y_pred = model.predict(x_test.view(-1, 28*28))
    acc_test = y_pred.eq(y_test).float().mean().item()
    print(f'Test Acc.: {acc_test:.4f}')
    show_img(x_test, labels=y_test, preds=y_pred)
    plt.savefig(f'{pth}/figures/show_x_test', dpi=300)
    plt.close()


    print(1)

    






