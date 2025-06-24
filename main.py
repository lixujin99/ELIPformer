
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from turtle import forward
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import os
import glob
import time
import math
import random
import argparse
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torch.autograd import Function
import collections
import random
from tqdm import tqdm
from torch import nn
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from scipy.fftpack import fft, fftshift, ifft
from scipy.fftpack import fftfreq
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import scipy
import clip
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat,savemat
import pandas as pd
from torchvision import transforms
from PIL import Image as IMG
from torchtoolbox.transform import Cutout
from sklearn.metrics import roc_auc_score, f1_score
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = IMG.BICUBIC
from config import config
from model import ELIPformer



dist.init_process_group(backend="nccl")

local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)



# In[75]:


# In[77]:

class MyDataset(Dataset):
    def __init__(self, data_eeg, label_eeg, label_index, eeg_img, transform0, transform1):
                                                              
        self.data_eeg = data_eeg
        self.label_eeg = label_eeg
        self.label_index = label_index
        self.img = eeg_img
        self.transforms0 = transform0
        self.transforms1 = transform1

    def __getitem__(self, index):
        eeg = self.data_eeg[index]
        eeg_label = self.label_eeg[index]
        if eeg_label==0:
            img = self.img[self.label_index[index].long()]
            img = self.transforms0(img)
        else:
            img = self.img[self.label_index[index].long()]
            img = self.transforms1(img)

        return eeg, eeg_label, img

    def __len__(self):
        return self.data_eeg.size(0)
    

def acc10(output, labels):
    preds = output.max(1)[1].type_as(labels)
    al = labels.shape[0]
    TP = 0   
    TN = 0   
    num_1 = 0
    for i in range(al):
        if ((preds[i]==1)and(labels[i]==1)):
            TP += 1
        if ((preds[i]==0)and(labels[i]==0)):
            TN += 1
        if (labels[i]==1):
            num_1 += 1
    return num_1,TP,TN



name = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']
name_test = ['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20']

path1 = ' ' #data location
path2 = '.npz'
path_img = path1 + 'images.mat'
path_index = path1 + 'index_car.mat'

text_label = clip.tokenize([ "nontarget","target"]).to(device)
text_train = clip.tokenize(["nontarget background","car"]).to(device)
text_test = clip.tokenize(["nontarget background","plane"]).to(device)

epochs = config.epoch 

BNmatrix = np.zeros(len(name_test))
PRmatrix = np.zeros((len(name_test),2))

_, preprocess = clip.load('ViT-B/32', device)
################################## load data #############################################

transform_non = transforms.Compose([transforms.RandomResizedCrop(500, scale=(0.3, 0.6), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                    transforms.RandomAffine(180, translate=None, scale=(0.8,1), shear=45, fill=0),
                                    transforms.Resize(224, interpolation=BICUBIC),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                        (0.26862954, 0.26130258, 0.27577711))])#0.5,0.8
transform_tar = transforms.Compose([transforms.RandomResizedCrop(500, scale=(0.2, 0.4), ratio=(0.75, 1.3333333333333333), interpolation=2),
                                    transforms.RandomAffine(180, translate=None, scale=(0.8,1), shear=45, fill=0),
                                    transforms.Resize(224, interpolation=BICUBIC),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                        (0.26862954, 0.26130258, 0.27577711))])
transform_test = transforms.Compose([transforms.Resize([224, 224]),                                   
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])])

Image = loadmat(path_img)
Image_array = Image['image']
Image = []
for i in range(Image_array.shape[0]):
    Image.append(IMG.fromarray(Image_array[i]))


for id_name in range(1):

    name1 = name
    model = ELIPformer(device,config.N,config.num_class)

    model = model.to(device)
    if (torch.cuda.device_count() > 1)and(dist.get_rank() == 0):
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    if (torch.cuda.device_count() > 1):
        model = DDP(model,device_ids=[local_rank],output_device=local_rank,find_unused_parameters=True)
    
    criterion = nn.CrossEntropyLoss().to(device)
    criterion_dis = nn.CosineEmbeddingLoss(margin=config.margin, reduction='mean')
    criterion_tri = nn.TripletMarginLoss(margin=config.margin, p=2.0, eps=1e-06, swap=False, size_average=None, reduce=None, reduction='mean')
    criterion_L2 = nn.MSELoss()
    
    t_total = time.time()
    
    for i in range(len(name1)):
        path = path1+name1[i]+path2
        mat = np.load(path)
        data1c = mat['Data1']
        data1c = data1c[:,np.newaxis,:,:]
        label1c = mat['label1']
        Data0 = data1c[label1c==0,:]
        Data1 = data1c[label1c==1,:]

        img_index = loadmat(path_index)
        if name1[i]=='ll2':
            img_index = loadmat(path1 + 'index_car_ll2.mat')
        img_index0 = np.squeeze(img_index['nontarget'])
        img_index1 = np.squeeze(img_index['target'])

        img_index = np.concatenate((img_index1,img_index0+1+np.max(img_index1)))

        img_index0 = img_index[label1c==0]
        img_index1 = img_index[label1c==1]
        rd_index = np.random.permutation(Data0.shape[0])
        data_0_downsampled = Data0[rd_index[:Data1.shape[0]],:]
        img_index0_downsampled = img_index0[rd_index[:Data1.shape[0]]]
        train_data_p = np.concatenate((Data1,data_0_downsampled),axis=0)
        img_index_p = np.concatenate((img_index1,img_index0_downsampled))
        train_label_p = np.concatenate((np.ones(Data1.shape[0]),np.zeros(data_0_downsampled.shape[0])),axis=0)
        if (i == 0):
            train_datac = train_data_p
            train_labelc = train_label_p
            img_indexc = img_index_p
        else:
            train_datac = np.append(train_datac, train_data_p, axis = 0)
            train_labelc = np.append(train_labelc, train_label_p)
            img_indexc = np.append(img_indexc, img_index_p)
        if (dist.get_rank() == 0):
            print(train_datac.shape)
        
            
    datas = train_datac 
    label = np.expand_dims(train_labelc, 1)
    img_indexc = np.expand_dims(img_indexc,1)
    label = np.concatenate((label,img_indexc),axis=1)
    
    a = np.random.permutation(datas.shape[0])
    datas = datas[a]  
    label = label[a]
        

    num_val = int(datas.shape[0]*0.2)
    val_data = torch.from_numpy(datas[datas.shape[0]-num_val:])
    val_label = torch.from_numpy(label[datas.shape[0]-num_val:])
    datas = datas[:datas.shape[0]-num_val]
    label = label[:label.shape[0]-num_val]
    train_data = torch.from_numpy(datas)
    train_label = torch.from_numpy(label)

    EEG_dataset1 = MyDataset(train_data,train_label[:,0],train_label[:,1],Image,transform_non,transform_tar)
    EEG_dataset2 = MyDataset(val_data,val_label[:,0],val_label[:,1],Image,preprocess,preprocess)

    train_sampler = torch.utils.data.distributed.DistributedSampler(EEG_dataset1)
    nw = min([os.cpu_count(), config.batchsize if config.batchsize > 1 else 0, 8]) 
    nw = 4*torch.cuda.device_count()
    trainloader = torch.utils.data.DataLoader(EEG_dataset1, batch_size=64,sampler=train_sampler, num_workers=nw)
    valloader = torch.utils.data.DataLoader(EEG_dataset2, batch_size=64, shuffle=True, num_workers=nw)

    
    ############################################################################ Pre-training #########################################################################################
    step = 1e-3
    val_max = 0
    stepp_new = 0
    
    for i in range(30):
        trainloader.sampler.set_epoch(i)
        dist.barrier()
        t = time.time()
        if (i%10 == 0 and i>0):
            step = step*0.8

        optimizer = optim.Adam(model.parameters(), lr=step, weight_decay=0.01)
        train_l_sum, train_acc_sum, n, acc1_sum, acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii1, data in enumerate(trainloader, 0):
            inputs, labels, vit_input= data
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            vit_input = vit_input.to(device)
                
            _, _, outputs_label, dis_feature, tri_feature, tri_feature0 = model(inputs,vit_input,labels,text_train,text_label)
            
            loss = criterion(outputs_label, labels.long())
            
            loss.backward()
            optimizer.step()
            
            train_l_sum += loss.cpu().item()
            train_acc_sum += (outputs_label.argmax(dim=1) == labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(outputs_label, labels)
            acc1_sum += num_acc1
            acc0_sum += num_acc0
            sum_1 += num_1
            n += labels.shape[0]

        sum_0 = n - sum_1
        train_l_sum = train_l_sum / (ii1+1)
        BN = train_acc_sum / n
        acc1 = acc1_sum/sum_1
        acc0 = acc0_sum/sum_0
        
        #Validation
        val_l_sum, val_acc_sum, n, val_acc1_sum, val_acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii2, data in enumerate(valloader, 0):
            val_inputs, val_labels, val_img_inputs = data
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.to(device)
            val_img_inputs = val_img_inputs.to(device)

            _, _, val_output, _, _, _ = model(val_inputs,val_img_inputs,None,text_train,text_label)
            loss_val = criterion(val_output, val_labels.long()) 

            val_l_sum += loss_val.cpu().item()
            val_acc_sum += (val_output.argmax(dim=1) == val_labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(val_output, val_labels)
            val_acc1_sum += num_acc1
            val_acc0_sum += num_acc0
            sum_1 += num_1
            n += val_labels.shape[0]

        sum_0 = n - sum_1
        val_l_sum = val_l_sum / (ii2+1)
        val_BN = val_acc_sum / n
        val_acc1 = val_acc1_sum/sum_1
        val_acc0 = val_acc0_sum/sum_0
            
        if (dist.get_rank() == 0):
            print('Epoch: {:04d}'.format(i+1),
                  'loss_train: {:.4f}'.format(train_l_sum),
                   "acc1= {:.4f}".format(acc1),
                    "acc0= {:.4f}".format(acc0),
                    "BN= {:.4f}".format(BN),
                    'loss_val: {:.4f}'.format(val_l_sum),
                    "val_BN= {:.4f}".format(val_BN),
                    "val_acc1= {:.4f}".format(val_acc1),
                    "val_acc0= {:.4f}".format(val_acc0),
                    "time: {:.4f}s".format(time.time() - t))
        

        if (val_BN>val_max):
            val_max = val_BN
            stepp_new = 0
            if (dist.get_rank() == 0):
                torch.save(model.state_dict(), config.savepre)
            
        stepp_new = stepp_new + 1

    dist.barrier()
    
    if (dist.get_rank() == 0):
        print('Finished Pre-training')
        model.load_state_dict(torch.load(config.savepre,map_location=device))

    
    ################################################################# Training #############################################################################
    trainloader = torch.utils.data.DataLoader(EEG_dataset1, batch_size=config.batchsize,sampler=train_sampler, num_workers=nw)
    valloader = torch.utils.data.DataLoader(EEG_dataset2, batch_size=64, shuffle=True, num_workers=nw)

    step = config.lr
    val_max = 0
    stepp_new = 0
    
    for i in range(epochs):
        trainloader.sampler.set_epoch(i)
        dist.barrier()
        t = time.time()
        if (i%20 == 0 and i>0):
            step = step*0.8

        special_layers = nn.ModuleList([model.module.embedding_t, model.module.model])

        special_layers_params = list(map(id, special_layers.parameters()))

        base_params = filter(lambda p: id(p) not in special_layers_params, model.parameters())
        optimizer = optim.Adam([{'params': base_params},
                                {'params': special_layers.parameters(), 'lr': step}], lr=step, weight_decay=0.01)

        train_l_sum, train_acc_sum, n, acc1_sum, acc0_sum, sum_1 = 0.0, 0.0, 0, 0, 0, 0
        for ii1, data in enumerate(trainloader, 0):
            inputs, labels, vit_input= data
            labels = labels.to(device)
            optimizer.zero_grad()
            inputs = inputs.to(device)
            vit_input = vit_input.to(device)
                
            outputs_label, outputs_img, outputs_eeg, dis_feature, tri_feature, tri_feature0 = model(inputs,vit_input,labels,text_train,text_label)
            
            loss = criterion(outputs_label, labels.long()) + criterion(outputs_eeg, labels.long()) + config.lam_tri1*criterion_tri(tri_feature[0],tri_feature[1],tri_feature[2]) + config.lam_tri0*criterion_tri(tri_feature0[0],tri_feature0[1],tri_feature0[2]) 
            
            loss.backward()
            optimizer.step()
            
            train_l_sum += loss.cpu().item()
            train_acc_sum += (outputs_label.argmax(dim=1) == labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(outputs_label, labels)
            acc1_sum += num_acc1
            acc0_sum += num_acc0
            sum_1 += num_1
            n += labels.shape[0]

        sum_0 = n - sum_1
        train_l_sum = train_l_sum / (ii1+1)
        BN = train_acc_sum / n
        acc1 = acc1_sum/sum_1
        acc0 = acc0_sum/sum_0
        
        #Validation
        val_l_sum, val_acc_sum, n, val_acc1_sum, val_acc0_sum, sum_1, val_acc_sum_EEG, val_acc_sum_img = 0.0, 0.0, 0, 0, 0, 0, 0, 0
        for ii2, data in enumerate(valloader, 0):
            val_inputs, val_labels, val_img_inputs = data
            val_labels = val_labels.to(device)
            val_inputs = val_inputs.to(device)
            val_img_inputs = val_img_inputs.to(device)

            val_output, val_output_img, val_output_EEG, _, _, _ = model(val_inputs,val_img_inputs,None,text_train,text_label)
            loss_val = criterion(val_output, val_labels.long()) 

            val_l_sum += loss_val.cpu().item()
            val_acc_sum += (val_output.argmax(dim=1) == val_labels).sum().cpu().item()
            val_acc_sum_EEG += (val_output_EEG.argmax(dim=1) == val_labels).sum().cpu().item()
            val_acc_sum_img += (val_output_img.argmax(dim=1) == val_labels).sum().cpu().item()
            num_1,num_acc1,num_acc0 = acc10(val_output, val_labels)
            val_acc1_sum += num_acc1
            val_acc0_sum += num_acc0
            sum_1 += num_1
            n += val_labels.shape[0]

        sum_0 = n - sum_1
        val_l_sum = val_l_sum / (ii2+1)
        val_BN = val_acc_sum / n
        val_acc1 = val_acc1_sum/sum_1
        val_acc0 = val_acc0_sum/sum_0

        val_BN_EEG = val_acc_sum_EEG / n
        val_BN_img = val_acc_sum_img / n
            
        if (dist.get_rank() == 0):
            print('Epoch: {:04d}'.format(i+1),
                  'loss_train: {:.4f}'.format(train_l_sum),
                   "acc1= {:.4f}".format(acc1),
                    "acc0= {:.4f}".format(acc0),
                    "BN= {:.4f}".format(BN),
                    'loss_val: {:.4f}'.format(val_l_sum),
                    "val_BN= {:.4f}".format(val_BN),
                    "val_acc1= {:.4f}".format(val_acc1),
                    "val_acc0= {:.4f}".format(val_acc0),
                    "val_BN_EEG= {:.4f}".format(val_BN_EEG),
                    "val_BN_image= {:.4f}".format(val_BN_img),
                    "time: {:.4f}s".format(time.time() - t))
        

        if (val_BN>val_max):
            val_max = val_BN
            stepp_new = 0
            if (dist.get_rank() == 0):
                torch.save(model.state_dict(), config.save)
            
        stepp_new = stepp_new + 1
        if (stepp_new==config.patience):
            break
    dist.barrier()
    
    if (dist.get_rank() == 0):
        print('Finished Training')
        model.load_state_dict(torch.load(config.save,map_location=device))


############################################################### Test ############################################################################
# Testing plane

BNmatrix = np.zeros(len(name_test))
AUCmatrix = np.zeros(len(name_test))
F1matrix = np.zeros(len(name_test))
PRmatrix = np.zeros((len(name_test),2))
Result = np.zeros((5,2))

path1 = ' ' #Test data location
path_img_test = path1 + 'images.npz'
Image = np.load(path_img_test)
Image_array = Image['image']
Image_test = []
for i in range(Image_array.shape[0]):
    Image_test.append(IMG.fromarray(Image_array[i]))

if (dist.get_rank() == 0):
    print('Testing')

for id_name in range(len(name_test)):
    path = path1+name_test[id_name]+path2
    path_index_test = path1 + 'index_plane.mat'
    mat = np.load(path)
    datas = mat['Data1']
    datas = datas[:,np.newaxis,:,:]
    label = mat['label1']
    img_index = loadmat(path_index_test)
    img_index = np.concatenate((np.squeeze(img_index['target']),np.squeeze(img_index['nontarget'])+1+np.max(img_index['target'])))

    label = np.expand_dims(label,1)
    img_index = np.expand_dims(img_index,1)
    label = np.concatenate((label,img_index),axis=1)

        
    test_data = torch.from_numpy(datas)
    test_label = torch.from_numpy(label)
    EEG_dataset3 = MyDataset(test_data,test_label[:,0],test_label[:,1],Image_test,preprocess,preprocess)
    testloader = torch.utils.data.DataLoader(EEG_dataset3, batch_size=128, shuffle=False)
        
    TP_all = 0
    TN_all = 0
    FP_all = 0
    FN_all = 0
        
    Label = None
    pre = None
    pre_prob = None
    for j, data in enumerate(testloader, 0):
        inputs, labels, img_inputs = data
        labels = labels.to(device)
        inputs = inputs.to(device)
        img_inputs = img_inputs.to(device)
        with torch.no_grad(): 
            outputs_label, _, _, _, _, _ = model(inputs,img_inputs,None,text_test,text_label)
        preds = outputs_label.max(1)[1].type_as(labels)

        Label = labels if Label is None else torch.cat((Label,labels))
        pre = preds if pre is None else torch.cat((pre,preds))
        pre_prob = F.softmax(outputs_label,dim=-1) if pre_prob is None else torch.cat((pre_prob,F.softmax(outputs_label,dim=-1)))

        al = labels.shape[0]
        TP = 0  
        FP = 0.001  
        TN = 0   
        FN = 0.001 
        for i in range(al):
            if ((preds[i]==1)and(labels[i]==1)):
                TP += 1
            if ((preds[i]==1)and(labels[i]==0)):
                FP += 1
            if ((preds[i]==0)and(labels[i]==1)):
                FN += 1
            if ((preds[i]==0)and(labels[i]==0)):
                TN +=1
        correct = preds.eq(labels).double()
        correct = correct.sum()
        acc_test = correct / len(labels)
        TP_all = TP+TP_all
        TN_all = TN+TN_all
        FP_all = FP+FP_all
        FN_all = FN+FN_all
        
    acc1 = TP_all/(TP_all+FN_all)
    acc0 = TN_all/(TN_all+FP_all)
    BN_all = (acc1+acc0)/2
    BNmatrix[id_name] = BN_all
    AUCmatrix[id_name] = roc_auc_score(Label.cpu().numpy(),pre_prob[:,1].cpu().numpy())
    F1matrix[id_name] = f1_score(Label.cpu().numpy(),pre.cpu().numpy())
    PRmatrix[id_name,0] = acc1
    PRmatrix[id_name,1] = acc0
    if (dist.get_rank() == 0):
        print(name_test[id_name]," Test set results:","acc1= {:.4f}".format(acc1),"acc0= {:.4f}".format(acc0),"BN= {:.4f}".format(BN_all),"F1-score= {:.4f}".format(F1matrix[id_name]),"AUC= {:.4f}".format(AUCmatrix[id_name]))
    dist.barrier()


BNmatrix = BNmatrix*100
acc = np.mean(BNmatrix)
var = np.var(BNmatrix)
auc_acc = np.mean(AUCmatrix)
auc_std = np.sqrt(np.var(AUCmatrix))
f1_acc = np.mean(F1matrix)
f1_std = np.sqrt(np.var(F1matrix))
std = np.sqrt(var)
std = std

Result[0,0] = acc
Result[0,1] = std
PRmatrix1 = PRmatrix
PRmatrix1[:,1] = 1 - PRmatrix1[:,1]
Result[1:3,0] = np.mean(PRmatrix1*100, axis=0)
Result[1:3,1] = np.sqrt(np.var(PRmatrix1*100, axis=0))
Result[3,0] = f1_acc
Result[3,1] = f1_std
Result[4,0] = auc_acc
Result[4,1] = auc_std

if (dist.get_rank() == 0):
    print(F1matrix)
    print(f1_acc, "+-", f1_std)
    print(AUCmatrix)
    print(auc_acc, "+-", auc_std)
    print(BNmatrix)
    print(PRmatrix*100)
    print(acc, "+-", std)
    print(np.mean(PRmatrix*100, axis=0))
    

#python -m torch.distributed.launch --master_port 29502 --nproc_per_node= EIformer/run.py