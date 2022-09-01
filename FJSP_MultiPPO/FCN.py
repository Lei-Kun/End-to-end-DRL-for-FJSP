import  torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from Params import configs
from uniform_instance import uni_instance_gen,FJSPDataset
class Pooling(nn.Module):
    def __init__(self,kernel_row=1,kernel_col=1):
        super(Pooling, self).__init__()
        self.kernel_row = kernel_row
        self.kernel_col = kernel_col
        self.relu = nn.ReLU()
        self.cnn1 = nn.Conv2d(3,1,(self.kernel_row,self.kernel_col))
        self.linear = nn.Linear(15,3)
    def forward(self,input):
        #input (batch_size,num_task,feature)
        print(input.shape)
        Output = self.linear(input)

        return Output

pool = Pooling()
data=[]
train_dataset = FJSPDataset(3, configs.n_m, configs.low, configs.high, configs.num_ins)

data_loader = DataLoader(train_dataset, batch_size=configs.batch_size)
for batch_idx, batch in enumerate(data_loader):

    data = torch.from_numpy(np.array(batch)).float()
    b = pool(data)
    print(b)