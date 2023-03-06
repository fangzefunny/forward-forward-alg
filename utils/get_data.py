import torch 
from torchvision import datasets, transforms

batch_size = 128
seed       = 2023

def get_mnist(if_download=False):
    '''Get mnist dataloaders

    Args:
        if_download: download the data or not

    Outputs:
        train_loader: load training data
        test_loader:  load testing data
    '''
    
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', 
            train=True, 
            download=if_download,
            transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((.1307,), (.3081,))
                ])),
        batch_size=batch_size,
        shuffle=True
    )

    test_loader  = torch.utils.data.DataLoader(
        datasets.MNIST('../data/MNIST', 
            train=False, 
            transform=transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize((.1307,), (.3081,))
                ])),
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, test_loader
