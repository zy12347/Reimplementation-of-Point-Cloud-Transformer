import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataloader,Dataset
import numpy as np
import matplotlib.pyplot as plt

class PCTransCls(nn.Module):
    def __init__(self,in_channel):
        super(PCTransCls,self).__init__()

        self.sg = SampleGroup(in_channel)
        self.sa1 = SelfAttention(128)
        self.sa2 = SelfAttention(128)
        self.sa3 = SelfAttention(128)
        self.sa4 = SelfAttention(128)

    def forward(self):
        pass

class SelfAttention(nn.Module):
    def __init(self,in_channel):
        super(SelfAttention,self).__init__()

    def forward(self):
        pass

class OffSetAttention(nn.Module):
    def __init(self):
        super(OffSetAttention,self).__init__()

    def forward(self):
        pass

class SampleGroup(nn.Module):
    def __init__(self,in_channel):
        super(SampleGroup,self).__init__()

    def forward(self):
        pass

def test():
    pass

def train():
    pass

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    pass

if __name__=='__main__':
    main()
