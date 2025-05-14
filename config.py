import os

class Config(object):
    def __init__(self):
        #all
        self.N = 1
        self.p = 0.2
        self.d_model = 128
        self.hidden = self.d_model * 4
        self.n_heads= 4 #feature % n_heads**2 ==0

        self.C = 64
        self.T = 250
        self.patchsizeh = 64
        self.patchsizew = 5
        self.H = self.C // self.patchsizeh
        self.W = self.T // self.patchsizew   
        
        self.batchsize = 1024 
        self.epoch = 50
        self.patience = self.epoch
        self.lr = 1e-3
        self.smooth = 0
        self.num_class = 2
        self.kl = 0.001
        self.lam_cos = 0
        self.lam_tri1 = 1
        self.lam_tri0 = 1
        self.margin = 0.5
        self.behave = False
        
        self.save = ' ' 
        self.savepre = ' '
        self.sample = 1    

        
config = Config()