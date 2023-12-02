import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PCTransCls(nn.Module):
    def __init__(self,in_channel=3,out_channel=6,points_num=1024,task='classification'):
        super(PCTransCls,self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.task = task
        
        self.embed = Embedding(in_channel)
        self.sg = SampleGroup(in_channel)
        self.sa1 = SelfAttention(128)
        self.sa2 = SelfAttention(128)
        self.sa3 = SelfAttention(128)
        self.sa4 = SelfAttention(128)
        
        self.lbr1 = nn.Sequential(
                    nn.Conv1d(in_channels=512,out_channels=1024,kernel_size=1),
                    nn.BatchNorm1d(1024),
                    nn.ReLU()
                )
        
        #self.m = nn.MaxPool1d(1024,1)
        #self.avg = nn.MaxPool1d(1024,1)
        # for classification
        self.lbrd1 = nn.Sequential(
                    nn.Conv1d(in_channels=2048,out_channels=256,kernel_size=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout()
                )
        
        self.lbrd2 = nn.Sequential(
                    nn.Conv1d(in_channels=256,out_channels=256,kernel_size=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout()
                )
        
        self.l1 = nn.Conv1d(in_channels=256,out_channels=out_channel,kernel_size=1)
        
        # for segmentation
        self.lbrd3 = nn.Sequential(
                    nn.Conv1d(in_channels=3072,out_channels=256,kernel_size=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout()
                    )
        
        self.lbr2 = nn.Sequential(
                    nn.Conv1d(in_channels=256,out_channels=256,kernel_size=1),
                    nn.BatchNorm1d(256),
                    nn.ReLU()
                )
        
        
    def forward(self,x):
        x0 = self.embed(x)
        x1 = self.sa1(x0)
        x2 = self.sa2(x1)
        x3 = self.sa3(x2)
        x4 = self.sa4(x3)
        x5 = torch.concatenate((x1,x2,x3,x4),1)
        #print("x5",x5.shape)
        x6 = self.lbr1(x5)#point feature  
        
        if self.task=='classification':
            #print("x6",x6.shape)
            x7 = torch.max(x6,-1)[0]
            #print("x6",x6.shape)
            x8 = torch.mean(x6,-1)
            #print("x7 shape",x7.shape)
            #print("x8 shape",x8.shape)
            x9 = torch.concatenate((x7,x8),1)
            x9 = x9.view(x9.shape[0],x9.shape[1],1)
            #print(x9.shape)
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
            self.label_path = self.path+'/train_labels.npy'
        else:
            self.data_path = self.path+'/test_data.npy'
            self.label_path = self.path+'/test_labels.npy'

        self.data = torch.from_numpy(np.load(self.data_path)).to(torch.long)
        self.label = torch.from_numpy(np.load(self.label_path)).to(torch.long)
        
    def __len__(self):
        return self.data.size()[0]
    
    def __getitem__(self, index):
        return self.data[index],self.label[index]

def get_data_loader(path,batch_size,train=True):
    dataset = ModelNetDataset(path,train)
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=train,num_workers=1)
    return dataloader
    
class SelfAttention(nn.Module):
    def __init(self,in_channel):
        super(SelfAttention,self).__init__()
        self.in_channel = in_channel
        self.q_conv = nn.Conv1d(in_channel, in_channel // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channel, in_channel // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channel, in_channel, 1)

        self.q_conv.weight = self.k_conv.weight 
        self.trans_conv = nn.Conv1d(in_channel, in_channel, 1)
        self.after_norm = nn.BatchNorm1d(in_channel)
        self.activate = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)


    def forward(self,x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)                       
        x_v = self.v_conv(x)                  

        div = math.sqrt(self.in_channel // 4)
        energy = torch.bmm(x_q, x_k) / div  

        attention = self.softmax(energy)                      

        x_s = torch.bmm(x_v, attention) 
        x_s = self.activate(self.after_norm(self.trans_conv(x_s)))
        
        x = x + x_s
        return 

class OffSetAttention(nn.Module):
    def __init(self,in_channel):
        super(OffSetAttention,self).__init__()
        self.in_channel = in_channel
        self.q_conv = nn.Conv1d(in_channel, in_channel // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channel, in_channel // 4, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channel, in_channel, 1)

        self.q_conv.weight = self.k_conv.weight 
        self.trans_conv = nn.Conv1d(in_channel, in_channel, 1)
        self.after_norm = nn.BatchNorm1d(in_channel)
        self.activate = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        x_q = self.q_conv(x).permute(0, 2, 1)
        x_k = self.k_conv(x)    
        x_v = self.v_conv(x)

        energy = torch.bmm(x_q, x_k)
        attention_raw = self.softmax(energy)
        sum_atten_raw = 1e-9 + attention.sum(dim=1, keepdims=True)
        attention = attention_raw / sum_atten_raw 

        x_a = torch.bmm(x_v, attention)
        x_d = x - x_a
        x_r = self.activate(self.after_norm(self.trans_conv(x_d)))
        x = x + x_r

        return x

class SampleGroup(nn.Module):
    def __init__(self,in_channel):
        super(SampleGroup,self).__init__()

    def forward(self):
        pass

class Embedding(nn.Module):
    def __init__(self,in_channel):
        super(Embedding,self).__init__()
        self.lbr0 = nn.Sequential(
                    nn.Conv1d(in_channels=in_channel,out_channels=128,kernel_size=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU()
                )
        
        self.lbr1 = nn.Sequential(
                    nn.Conv1d(in_channels=128,out_channels=128,kernel_size=1),
                    nn.BatchNorm1d(128),
                    nn.ReLU()
                )
        
    def forward(self,x):
        x1 = self.lbr0(x)
        x2 = self.lbr1(x1)
        return x2

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
        point_clouds = torch.permute(point_clouds,(0,2,1))
        point_clouds = point_clouds.to(device).float()
        labels = labels.to(device)
        output = model(point_clouds)
        
        # loss = loss_fn(y_pred, y_train_torch)
        # print(t, loss.item())
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()`
    return 0
        

def main():
    NUM_CLASSES = 6
    NUM_FEATURES =3
    NUM_TRAIN_POINTS = 1024
    
    TASK = 'classification'
    model_save_path = os.path.join(os. getcwd(), 'model/best_model.pt')
    learning_rate = 0.001
    path = 'data/cls'
    batch_size =16
    num_epochs = 200
    model = PCTransCls(NUM_FEATURES,NUM_CLASSES,NUM_TRAIN_POINTS,TASK)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    opt = optim.Adam(model.parameters(),learning_rate)
    
    train_dataloader = get_data_loader(path,batch_size,True)
    #test_dataloader = get_data_loader(path,batch_size,False)
    
    model = model.to(device)
    
    best_accuracy = -1
    
    for epoch in tqdm(range(num_epochs)):
        train_epoch_loss = train(train_dataloader,model,opt,epoch,device,TASK) 
        test_accuracy = 1#test()
        print ("epoch: {}   train loss: {:.4f}   test accuracy: {:.4f}".format(epoch, train_epoch_loss, test_accuracy))
        #if test_accuracy>best_accuracy:
        #    best_accuracy = test_accuracy
        #    torch.save(model.state_dict(),model_save_path)
            

if __name__=='__main__':
    main()
