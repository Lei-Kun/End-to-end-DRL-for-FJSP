import numpy as np
from torch.utils.data import Dataset
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import DataParallel
def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high,seed=None):
    if seed != None:
        np.random.seed(seed)

    num_mch = np.random.randint(low=1,high=n_m,size=(n_j*n_m))
    time = np.random.randint(low=low,high=high,size=(n_j*n_m,n_m))

    for i in range(n_j*n_m):
        time[i][0:num_mch[i]] = 0
    time = time.reshape(n_j,n_m,n_m)
    for i in range(n_j):
        time[i] = permute_rows(time[i])
    return time


class FJSPDataset(Dataset):

    def __init__(self,n_j, n_m, low, high,num_samples=1000000,seed=None,  offset=0, distribution=None):
        super(FJSPDataset, self).__init__()
        self.data_set = []
        if seed != None:
            np.random.seed(seed)
        num_mch = np.random.randint(low=1, high=n_m, size=(num_samples,n_j * n_m))
        time = np.random.randint(low=low, high=high, size=(num_samples, n_j* n_m, n_m ))
        for i in range(num_samples):
            for j in range(n_j*n_m):
                time[i][j][0:num_mch[i,j]] = 0
        time = time.reshape(num_samples,n_j,n_m,n_m)
        for i in range(num_samples):
            for j in range(n_j):
                time[i][j] = permute_rows(time[i][j])
        self.data = np.array(time)

        self.size = len(self.data)
    def getdata(self):
        return self.data

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]



def override(fn):
    """
    override decorator
    """
    return fn
