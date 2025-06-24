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


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, max_len, d_model=config.d_model, dropout=config.p):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],  requires_grad=False)
        return self.dropout(x)

class PatchEmbedding(nn.Module):
    def __init__(self, device, in_channels: int = 1, patch_sizeh: int = config.patchsizeh, patch_sizew: int = config.patchsizew, emb_size: int = config.d_model, img_size1: int = config.C, img_size2:int = config.T):
        self.patch_sizeh = patch_sizeh
        self.patch_sizew = patch_sizew
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=(self.patch_sizeh,self.patch_sizew), stride=(self.patch_sizeh,self.patch_sizew),padding=(0,0)),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )
        self.cls_token = nn.Parameter(torch.randn(1,1, emb_size))

        self.positions = nn.Parameter(torch.randn(((img_size1*img_size2) // (self.patch_sizew*self.patch_sizeh)), emb_size))
        self.nonpara = PositionalEncoding(((img_size1*img_size2) // (self.patch_sizew*self.patch_sizeh))).to(device)

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x += self.positions
        return x

class Mutihead_Attention(nn.Module):
    def __init__(self, device,d_model,dim_k,dim_v,n_heads):
        super(Mutihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)
        self.device = device

    def generate_mask(self,dim,score):
        thre = torch.mean(score,dim=-1).to(self.device) 
        thre = torch.unsqueeze(thre, 3)
        vec = torch.ones((1,dim)).to(self.device)
        thre = torch.matmul(thre,vec) 
        cha = score - thre
        one_vec = torch.ones_like(cha).to(self.device)
        zero_vec = torch.zeros_like(cha).to(self.device)
        mask = torch.where(cha > 0, zero_vec, one_vec).to(self.device)
        return mask==1

    def forward(self,x,y,requires_mask=True):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(y).reshape(-1,y.shape[0],y.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1,y.shape[0],y.shape[1],self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v
        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        if requires_mask:
            mask = self.generate_mask(attention_score.size()[3],attention_score)
            attention_score = attention_score.masked_fill(mask==True,value=float("-inf")) 
        attention_score = F.softmax(attention_score,dim=-1)
        output = torch.matmul(attention_score,V).reshape(x.shape[0],x.shape[1],-1)

        output = self.o(output)
        return output

class Mutihead_Attention_K(nn.Module):
    def __init__(self, device,d_model,dim_k,dim_v,n_heads):
        super(Mutihead_Attention_K, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model,dim_k)
        self.k = nn.Linear(d_model,dim_k)
        self.v = nn.Linear(d_model,dim_v)

        self.o = nn.Linear(dim_v,d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

        self.fc_w = nn.Linear(dim_v*2,2)
        self.device = device

    def generate_mask(self,dim,score):
        thre = torch.mean(score,dim=-1).to(self.device) 
        thre = torch.unsqueeze(thre, 3)
        vec = torch.ones((1,dim)).to(self.device)
        thre = torch.matmul(thre,vec) 
        cha = score - thre
        one_vec = torch.ones_like(cha).to(self.device)
        zero_vec = torch.zeros_like(cha).to(self.device)

        mask = torch.where(cha > 0, zero_vec, one_vec).to(self.device)
        return mask==1

    def generate_maxmask(self,dim,score):
        thre, _ = torch.max(score,dim=-2) 
        thre = torch.unsqueeze(thre, 2).to(self.device)
        vec = torch.ones((dim,1)).to(self.device)
        thre = torch.matmul(vec,thre)  
        cha = score - thre
        one_vec = torch.ones_like(cha).to(self.device)
        zero_vec = torch.zeros_like(cha).to(self.device)

        mask = torch.where(cha < 0, zero_vec, one_vec).to(self.device)
        mask1 = torch.where(thre < 0, zero_vec, one_vec).to(self.device)
        mask = torch.mul(mask, mask1)
        return mask==0

    def weight_sum(self,anchor,F1,F2):
        all_token2 = torch.cat((F1,F2),dim=-1)
        w = F.softmax(self.fc_w(all_token2),dim=-1)        
        w1 = w[:,:,0:1]
        w2 = w[:,:,1:]

        return torch.mul(w1,F1) + torch.mul(w2,F2)

    
    def forward(self,x,y,requires_mask=False):
        assert self.dim_k % self.n_heads == 0 and self.dim_v % self.n_heads == 0
        # size of x : [batch_size * seq_len * batch_size]
  
        Q = self.q(x).reshape(-1,x.shape[0],x.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        K = self.k(y).reshape(-1,y.shape[0],y.shape[1],self.dim_k // self.n_heads) # n_heads * batch_size * seq_len * dim_k
        V = self.v(y).reshape(-1,y.shape[0],y.shape[1],self.dim_v // self.n_heads) # n_heads * batch_size * seq_len * dim_v
   
        attention_score = torch.matmul(Q,K.permute(0,1,3,2)) * self.norm_fact
        attention_score1 = attention_score

        attention_score = F.softmax(attention_score,dim=-2)

        if requires_mask:
            mask = self.generate_mask(attention_score1.size()[3],attention_score1)
            attention_score1 = attention_score1.masked_fill(mask==True,value=float("-inf")) 

        attention_score = F.normalize(attention_score,p=1,dim=-1)
        attention_score1 = F.softmax(attention_score1,dim=-1)
        output = torch.matmul(attention_score,V).reshape(x.shape[0],x.shape[1],-1)
        output1 = torch.matmul(attention_score1,V).reshape(x.shape[0],x.shape[1],-1)

        output = 0.5*(output + output1)  

        output = self.o(output)
        return output

class Feed_Forward1(nn.Module):
    def __init__(self, device,input_dim,hidden_dim):
        super(Feed_Forward1, self).__init__()
        self.L1 = nn.Linear(input_dim,hidden_dim).to(device)
        self.L2 = nn.Linear(hidden_dim,input_dim).to(device)
        self.dropout = nn.Dropout(config.p)
        self.gelu = GELU().to(device)

    def forward(self,x):
        output = self.gelu((self.L1(x)))
        output = self.dropout(output)
        output = self.L2(output)
        return output

class Feed_Forward(nn.Module):  #output
    def __init__(self,device,input_dim=config.d_model,hidden_dim=config.hidden):
        super(Feed_Forward, self).__init__()
        F1 = 16
        self.conv1 = nn.Conv2d(1, F1, (50, 16), bias = False, stride = (50,16))  #Conv2d #F1*4*8
        self.fc = nn.Linear(F1*1*8,config.d_model)
        self.dropout = nn.Dropout(config.p)
        self.gelu = QuickGELU().to(device)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self,x):  
        output = self.gelu(self.conv1(x.unsqueeze(1)))
        output = output.contiguous().view(-1, self.num_flat_features(output))
        return output
        
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features*s
        return num_features

class Add_Norm(nn.Module):
    def __init__(self, device):
        super(Add_Norm, self).__init__()
        self.dropout = nn.Dropout(config.p).to(device)
        self.device = device

    def forward(self,x,sub_layer,**kwargs):
        sub_output = sub_layer(x,**kwargs)
        x = self.dropout(x + sub_output)
        
        layer_norm = nn.LayerNorm(x.size()[1:]).to(self.device)
        out = layer_norm(x)
        return out

class Add_Norm_cross(nn.Module):
    def __init__(self, device):
        super(Add_Norm_cross, self).__init__()
        self.dropout = nn.Dropout(config.p).to(device)
        self.device = device

    def forward(self,x,sub_layer,y,z):
        sub_output = sub_layer(x,y,z)
        x = self.dropout(x + sub_output)
        
        layer_norm = nn.LayerNorm(x.size()[1:]).to(self.device)
        out = layer_norm(x)
        return out

class Encoder(nn.Module):
    def __init__(self, device,dim_seq,dim_fea,n_heads,hidden):
        super(Encoder, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = Mutihead_Attention(device,self.dim_fea,self.dim_k,self.dim_v,self.n_heads).to(device)
        self.feed_forward = Feed_Forward1(device,self.dim_fea,self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self,x):
        output = self.add_norm(x,self.muti_atten,y=x)
        output = self.add_norm(output,self.feed_forward)
        return output
        
class Encoder_last(nn.Module):
    def __init__(self, device,dim_seq,dim_fea,n_heads,hidden):
        super(Encoder_last, self).__init__()
        self.dim_seq = dim_seq
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden

        self.muti_atten = Mutihead_Attention(device,self.dim_fea,self.dim_k,self.dim_v,self.n_heads).to(device)
        self.feed_forward = Feed_Forward(device,self.dim_fea,self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self,x):
        output = self.add_norm(x,self.muti_atten,y=x)
        output = self.feed_forward(output)
        return output
    
class Decoder(nn.Module):
    def __init__(self,device,dim_seq,dim_fea,n_heads,hidden):
        super(Decoder, self).__init__()
        self.dim_seq = dim_seq 
        self.dim_fea = dim_fea
        self.n_heads = n_heads
        self.dim_k = self.dim_fea // self.n_heads
        self.dim_v = self.dim_k
        self.hidden = hidden
        
        self.muti_atten = Mutihead_Attention_K(device,self.dim_fea,self.dim_k,self.dim_v,self.n_heads).to(device)
        self.feed_forward = Feed_Forward1(device,self.dim_fea,self.hidden).to(device)
        self.add_norm = Add_Norm(device).to(device)

    def forward(self,v,q):
        output = self.add_norm(q,self.muti_atten,y=v,requires_mask=True)
        output = self.add_norm(output,self.feed_forward)
        output = output + q
        return output

class Cross_modal(nn.Module):
    def __init__(self, device):
        super(Cross_modal, self).__init__()
        self.cross1 = Decoder(device,config.H*config.W,config.d_model,config.n_heads,config.hidden).to(device)
        self.cross2 = Decoder(device,config.H*config.W,config.d_model,config.n_heads,config.hidden).to(device)
        self.fc1 = nn.Linear(2*config.d_model,config.d_model).to(device)

    def forward(self,target,f1):
        re = self.cross1(target,f1)
        return re

# In[78]:

class Transformer_layer(nn.Module):
    def __init__(self, device, dmodel=config.d_model, num_heads=config.n_heads, num_tokens=config.H*config.W):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder(device,num_tokens,dmodel,num_heads,config.hidden).to(device) 

    def forward(self,x):  
        encoder_output = self.encoder(x) + x
        return encoder_output

class Transformer_layer_last(nn.Module):
    def __init__(self, device):
        super(Transformer_layer_last, self).__init__()
        self.encoder = Encoder_last(device,config.H*config.W,config.d_model,config.n_heads,config.hidden).to(device) 

    def forward(self,x):  
        encoder_output = self.encoder(x)
        return encoder_output


# In[79]:
class Pre_model(torch.nn.Module):
    def __init__(self, device):
        super(Pre_model, self).__init__()
        self.model, _  = clip.load('ViT-B/32', device)
        self.vit = self.model.visual
        self.pad = torch.nn.ZeroPad2d(padding=(0,1))
        self.device = device

    def vit_forward(self,x):
        x = self.vit.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.vit.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.vit.positional_embedding.to(x.dtype)
        x = self.vit.ln_pre(x)
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vit.transformer(x)

        x = x.permute(1, 0, 2)  # LND -> NLD
        
        x = self.vit.ln_post(x)
        if self.vit.proj is not None:
            x = x @ self.vit.proj
            
        return x

    def semantic_embedding(self,img_fea,text_fea,weights):
        text_fea0 = repeat(text_fea[0:1], '() f -> b f', b=img_fea.shape[0])
        text_fea1 = repeat(text_fea[1:], '() f -> b f', b=img_fea.shape[0])
        sen_embedding = torch.mul(weights[:,0].unsqueeze(1),text_fea0) + torch.mul(weights[:,1].unsqueeze(1),text_fea1)
            
        return sen_embedding.unsqueeze(1)

    def semantic_embedding01(self,img_fea,text_fea,weights):
        text_fea0 = repeat(text_fea[0:1], '() f -> b f', b=img_fea.shape[0])
        text_fea1 = repeat(text_fea[1:], '() f -> b f', b=img_fea.shape[0])
        
        values,_ = weights.max(axis=1)
        values = repeat(values.unsqueeze(1), 'b () -> b f', f=2)
        weights = (weights==values).long()

        sen_embedding = torch.mul(weights[:,0].unsqueeze(1),text_fea0) + torch.mul(weights[:,1].unsqueeze(1),text_fea1)

        return sen_embedding.unsqueeze(1)

    def semantic_embedding_label01(self,img_fea,text_fea,weights,labels):
        text_fea0 = repeat(text_fea[0:1], '() f -> b f', b=img_fea.shape[0])
        text_fea1 = repeat(text_fea[1:], '() f -> b f', b=img_fea.shape[0])
        
        values,_ = weights.max(axis=1)
        values = repeat(values.unsqueeze(1), 'b () -> b f', f=2)
        weights_ori = (weights==values).long() 
        weights_ori = weights_ori.float()
        weights = (weights_ori==labels).long() 
        weights = weights.float()
        pre = torch.sum(weights,dim=-1)
        w = torch.rand_like(pre[pre==0]).unsqueeze(1).to(self.device)
        w = torch.where(w>0.5,torch.ones_like(w,device=w.device),torch.zeros_like(w,device=w.device))
        weights_ori[pre==0] = torch.mul(w,weights_ori[pre==0]) + torch.mul((torch.ones_like(w).to(self.device)-w),labels[pre==0])
        weights = weights_ori

        sen_embedding = torch.mul(weights[:,0].unsqueeze(1),text_fea0) + torch.mul(weights[:,1].unsqueeze(1),text_fea1)

        return sen_embedding.unsqueeze(1)


    def select(self,A):
            values,_ = A.max(axis=1)
            values = repeat(values.unsqueeze(1), 'b () -> b f', f=A.shape[1])
            A = (A==values).long()

            return A

    def forward(self, img, batchlabels, text, text_label):
        img_feat = self.vit_forward(img)
        
        image_features = self.model.encode_image(img)
        text_features = self.model.encode_text(text)
        text_label_features = self.model.encode_text(text_label)

        logits_per_image, logits_per_text = self.model(img, text)
        probs = logits_per_image.softmax(dim=-1)

        if batchlabels is not None:
            batchlabels = F.one_hot(batchlabels.long()).float()
            sen_emb_label = self.semantic_embedding_label01(image_features,text_label_features,probs,batchlabels)
            sen_emb_label = repeat(sen_emb_label,'b () f -> b n f',n=img_feat.shape[1])
            sen_emb = sen_emb_label
        else:
            sen_emb = self.semantic_embedding01(image_features,text_label_features,probs)
            sen_emb = repeat(sen_emb,'b () f -> b n f',n=img_feat.shape[1])

        sen_emb = sen_emb.reshape(sen_emb.shape[0],sen_emb.shape[1],-1,2)
        sen_emb = self.pad(sen_emb).reshape(sen_emb.shape[0],sen_emb.shape[1],768)
        return sen_emb


class Pre_vit(torch.nn.Module):
    def __init__(self, device):
        super(Pre_vit, self).__init__()
        self.model, _  = clip.load('ViT-B/32', device)
        self.vit = self.model.visual

    def vit_forward(self,x,sen):
        x = self.vit.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([0.5*self.vit.class_embedding.to(x.dtype) + 0.5*sen[:,0:1,:], x], dim=1)
        x = x + self.vit.positional_embedding.to(x.dtype)
        x = self.vit.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.vit.transformer.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.vit.ln_post(x)
        return x

    def forward(self,x,sen):
        x = self.vit_forward(x,sen)
        return x
        

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)

#share
class ELIPformer(nn.Module):
    def __init__(self,device,N,output_dim):
        super(Transformer, self).__init__()

        self.output_dim = output_dim

        self.embedding_t = PatchEmbedding(device).to(device)
        self.semantic_embedding = Pre_model(device).to(device)
        self.embedding_img = Pre_vit(device).to(device)
        convert_weights(self.embedding_img)
        convert_weights(self.semantic_embedding)
        self.norm = nn.LayerNorm(config.d_model).to(device)
        self.norm2 = nn.LayerNorm(config.d_model*2).to(device)
        self.norm3 = nn.LayerNorm(config.d_model).to(device)

        #Encoder
        self.model = nn.Sequential(*[Transformer_layer(device) for _ in range(N)]).to(device)
        self.model_img = nn.Sequential(*[Transformer_layer(device) for _ in range(N)]).to(device)
        self.model_last = Transformer_layer_last(device).to(device)
        self.model_last1 = Transformer_layer_last(device).to(device)
        self.model_last_img = Transformer_layer_last(device).to(device)
        
        #cross-modal
        self.t = Cross_modal(device).to(device)
        self.f = Cross_modal(device).to(device)
        self.t1 = Cross_modal(device).to(device)
        self.f1 = Cross_modal(device).to(device)
        self.fc_eeg = nn.Linear(16*(config.d_model//16),output_dim)
        self.fc_bri = nn.Linear(16*(config.d_model//16),config.d_model)
        self.fc = nn.Linear(config.d_model*2,output_dim)
        self.fc_img = nn.Linear(config.d_model,output_dim)

        self.pool_img = nn.MaxPool1d(int(768//config.d_model),stride=int(768//config.d_model))
        self.dropout = nn.Dropout(0.2).to(device)
        self.gelu = GELU().to(device)

    def forward(self,raw,img,batch_labels,text,text_label):
        Guidence = []
        x_t = self.embedding_t(raw)
        x_t = self.model(x_t)
        with torch.no_grad(): 
            sen_emb = self.semantic_embedding(img,batch_labels,text,text_label)
            x_i = self.embedding_img(img,sen_emb.detach())
        x_i = self.pool_img(x_i.detach())
       
        #cross-modal
        x_i2 = self.f(x_i,x_t)
        x_t2 = self.t(x_t,x_i)

        x_i1 = self.f1(x_i2,x_t2)
        x_t1 = self.t1(x_t2,x_i2)

        output_EEG = self.fc_eeg(self.model_last1(x_t1+x_t))
        output_img = self.fc_img(x_i1[:,0,:])

        #fusion
        output = torch.cat((self.gelu(self.fc_bri(self.model_last(x_t1))),x_i1[:,0,:]),dim=-1)
        

        #metric learning
        output_sub0 = None
        output_sub1 = None
        dis_feature = None
        tri_tar = None
        tri_nontar = None
        tri_output = output
        tri_output_0 = None
        tri_tar_0 = None
        tri_nontar_0 = None
        if batch_labels is not None:
            dis_feature0 = torch.mean(x_t[batch_labels==0],dim=0).unsqueeze(0)
            dis_feature1 = torch.mean(x_t[batch_labels==1],dim=0).unsqueeze(0)
            dis_feature0 = dis_feature0.contiguous().view(-1, self.num_flat_features(dis_feature0))
            dis_feature1 = dis_feature1.contiguous().view(-1, self.num_flat_features(dis_feature1))
            dis_feature = [dis_feature0, dis_feature1]

            output_sub0 = x_t[batch_labels==0]
            output_sub1 = x_t[batch_labels==1]

            tri_output0 = tri_output[batch_labels==0,:]
            tri_output1 = tri_output[batch_labels==1,:]
            tri_tar0 = torch.mean(output[batch_labels==0,:],dim=0).unsqueeze(0)
            tri_tar1 = torch.mean(output[batch_labels==1,:],dim=0).unsqueeze(0)

            tri_output = tri_output1
            tri_tar = repeat(tri_tar1, '() n -> b n', b=tri_output1.shape[0])
            tri_nontar = repeat(tri_tar0, '() n -> b n', b=tri_output1.shape[0]) 
            tri_output_0 = tri_output0
            tri_tar_0 = repeat(tri_tar0, '() n -> b n', b=tri_output0.shape[0])
            tri_nontar_0 = repeat(tri_tar1, '() n -> b n', b=tri_output0.shape[0]) 

        output = self.fc(output)

        return output, output_img, output_EEG, dis_feature, [tri_output, tri_tar, tri_nontar], [tri_output_0,tri_tar_0,tri_nontar_0]

    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features = num_features*s
        return num_features