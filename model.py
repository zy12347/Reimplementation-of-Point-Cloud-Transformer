import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataloader,Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PCTransCls(nn.Module):
    def __init__(self,in_channel=1000,out_channel=6,task='classification'):
        super(PCTransCls,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.task = task
        self.sg = SampleGroup(in_channel)
        self.sa1 = SelfAttention(128)
        self.sa2 = SelfAttention(128)
        self.sa3 = SelfAttention(128)
        self.sa4 = SelfAttention(128)

        self.lbr1 = nn.Sequential(
                    nn.Linear(in_features=512,out_features=1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                )
        # for classification
        self.lbrd1 = nn.Sequential(
                    nn.Linear(in_features=2048,out_features=256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout()
                )
        
        self.lbrd2 = nn.Sequential(
                    nn.Linear(in_features=256,out_features=256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout()
                )
        
        self.l1 = nn.Linear(in_features=256,out_features=out_channel)
        
        # for segmentation
        self.lbrd3 = nn.Sequential(
                    nn.Linear(in_features=3072,out_features=256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout()
                    )
        
        self.lbr2 = nn.Sequential(
                    nn.Linear(in_features=256,out_features=256),
                    nn.BatchNorm1d(256),
                    nn.ReLU()
                )
        
        
    def forward(self,x):
        x1 = self.sa1(x)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = torch.concatenate((x1,x2,x3,x4),0)
        x6 = self.lbr1(x5)#point feature  
        
        if self.task=='classification':
            x7 = torch.max(x6,dim=2)
            x8 = torch.mean(x6,dim=2)  
            x9 = torch.concatenate((x7,x8),1)
            x10 = self.lbrd1(x9)
            x11 = self.lbrd2(x10)
        
        if self.task=='segmentation':
            x7 = x6.repeat(1,1,self.in_channel)
            x8 = torch.concatenate((x7,x6),1)
            x9 = self.lbrd3(x8)
            x11 = self.lbr2(x9) #for the alignment with classification part
        
        output = self.l1(x11)
        return output

class ModelNetDataset(Dataset):
    def __init__(self,path,train=True):
        self.path = path
        if train:
            self.data_path = self.path+'/train_data.npy'
            self.label_path = self.path+'train_labels.npy'
        else:
            self.data_path = self.path+'/test_data.npy'
            self.label_path = self.path+'test_labels.npy'

        self.data = torch.from_numpy(np.load(self.data_path))
        self.label = torch.from_numpy(np.load(self.label_path)).to(torch.long)
        
    def __len__(self):
        return self.data.size()[0]
    
    def __getitem__(self, index):
        return self.data[index],self.label[index]

def get_data_loader(path,batch_size,train=True):
    dataset = ModelNetDataset(path,train)
    dataloader = Dataloader(dataset,batch_size=batch_size,shuffle=train,num_workers=1)
    
class SelfAttention(nn.Module):
    def __init(self,in_channel):
        super(SelfAttention,self).__init__()

    def forward(self,x):
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


def test(test_dataloader,model,epoch,device,task):
    model.eval()
    correct_obj = 0
    num_obj = 0
    for batch in test_dataloader:
        point_clouds,label = batch
    pass

def train(train_dataloader,model,opt,epoch,device,Task):
    model.train()
    step = epoch*len(train_dataloader)
    epoch_loss = 0
    
    for i,batch in enumerate(train_dataloader):
        point_clouds,labels = batch
        point_clouds.to(device)
        labels = labels.to(device)
        output = model.forward(point_clouds)
        
    pass

def main():
    NUM_CLASSES = 6
    NUM_TRAIN_POINTS = 2000
    TASK = 'classification'
    model_save_path = os.path.join(os. getcwd(), 'model/best_model.pt')
    learning_rate = 0.001
    path = 'data/cls'
    batch_size =32 
    num_epochs = 200
    model = PCTransCls(NUM_TRAIN_POINTS,NUM_CLASSES,TASK)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    opt = optim.Adam(model.parameters(),learning_rate)
    
    train_dataloader = get_data_loader(path,batch_size,True)
    test_dataloader = get_data_loader(path,batch_size,False)
    
    model = model.to(device)
    
    best_accuracy = -1
    
    for epoch in tqdm(range(num_epochs)):
        train_epoch_loss = train(train_dataloader,model,opt,epoch,device,TASK) 
        test_accuracy = test()
        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, test_accuracy))
        if test_accuracy>best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(),model_save_path)
            

if __name__=='__main__':
    main()
