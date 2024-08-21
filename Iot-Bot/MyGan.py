import torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import collections
from collections import OrderedDict
import pandas as pd
import wandb
from tqdm import tqdm 
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.autograd import Variable



class Gen(nn.Module):
    def __init__(self, args,device):
        super(Gen, self).__init__()
        self.input_dim = args.noise_size
        self.output_dim = args.n_features
        self.class_num = args.n_classes
        self.device=device
        factory_kwargs = {'device': self.device}
        self.label_emb = nn.Embedding(self.class_num,self.class_num, **factory_kwargs)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
    
        self.model = nn.Sequential(
            *block(self.input_dim + self.class_num, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1028),
            *block(1028, 2056),
            *block(2056,self.output_dim)
        )

    def forward(self, noise ,label):
        x = torch.cat((self.label_emb(label).squeeze(),noise), 1)
        x = self.model(x)
        return x

# class Dis(nn.Module):
#     def __init__(self,args,device):
#         super(Dis, self).__init__()
#         self.input_dim = args.n_features
#         self.output_dim = args.n_features
#         self.class_num = args.n_classes
#         self.device=device
#         factory_kwargs = {'device': self.device}
#         self.label_emb = nn.Embedding(self.class_num,self.class_num, **factory_kwargs)

#         self.model = nn.Sequential(
#             nn.Linear((self.class_num + self.input_dim), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 512),
#             nn.Dropout(0.4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 1),
#             nn.Sigmoid()
#         )

#     def forward(self, input ,label):
#         # Concatenate label embedding and image to produce input
#         x = torch.cat((self.label_emb(label).squeeze(), input), 1)
#         x = self.model(x)
#         return x


class Dis(nn.Module):
    def __init__(self, args, device):
        super(Dis, self).__init__()
        self.input_dim = args.n_features
        self.output_dim = args.n_features
        self.class_num = args.n_classes
        self.device = device
        factory_kwargs = {'device': self.device}
        self.label_emb = nn.Embedding(self.class_num, self.class_num, **factory_kwargs)

        self.model = nn.Sequential(
            nn.Linear(self.class_num + self.input_dim, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, input, label):
        # Concatenate label embedding and image to produce input
        x = torch.cat((self.label_emb(label).squeeze(), input), 1)
        x = self.model(x)
        return x

class CGAN():
    def __init__(self, args,device):
        # parameters
        self.device = device
        self.epoch = args.Gan_epochs
        self.batch_size = args.GAC_batch_size
        self.fbatch_size=args.G_batch_size
        self.z_dim = args.noise_size
        self.class_num = args.n_classes
        #location of Gen and Dis parameters keys
        self.D_params_key=[]
        self.G_params_key=[]


        # networks init
        self.G = Gen(args,self.device).to(self.device)
        self.D = Dis(args,self.device).to(self.device)


        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lrG)
        # self.D_optimizer = optim.RMSprop(self.D.parameters(), lr=args.lrD, alpha=0.9)
        self.D_optimizer=optim.Adam(self.D.parameters(), lr=args.lrD)
        self.MSE_loss = torch.nn.MSELoss()
        self.BCE_loss=nn.BCELoss()
        self.CE_loss=nn.CrossEntropyLoss()
    def to(self,device):
        self.G.to(device)
        self.D.to(device)
    def freedata_train(self,m_model,criterion,round="0",name_log="Generator" ):
        # wandb.watch(self.G,criterion,log="all",log_freq=10)
        history = {'loss': [],'accuracy':[]}
        print('training start!!')
        for epoch in tqdm(range(self.epoch), desc="Freedata Training progress"):
            self.G.train()
            y_ = torch.randint(0, self.class_num, (self.fbatch_size,))
            y_=y_.squeeze()
            y_=torch.tensor(y_).long().to(self.device)
            self.G_optimizer.zero_grad()
            z_=torch.rand((self.fbatch_size, self.z_dim)).to(self.device)
            G_ = self.G(z_, y_).to(self.device)
            M = m_model(G_).to(self.device)
            G_loss,G_accuracy=m_model.Quick_evaluate(M,y_,criterion)
            G_loss.backward()
            self.G_optimizer.step()
            # print(f"epoch {epoch} G loss : {G_loss}")
            history["loss"].append(G_loss)
            history["accuracy"].append(G_accuracy)
            # wandb.define_metric(f"{name_log}/ round {round} loss", step_metric="epochs")
            # wandb.define_metric(f"{name_log}/ round {round} accuracy", step_metric="epochs")
            # wandb.log({f"{name_log}/ round {round} accuracy": G_accuracy, "epochs":epoch})
            # wandb.log({f"{name_log}/ round {round} loss":G_loss, "epochs":epoch})
        return history
                

    # def train(self,trainset):
    #     self.train_hist = {}
    #     self.train_hist['D_loss'] = []
    #     self.train_hist['G_loss'] = []
    #     print('training start!!')
    #     for epoch in range(self.epoch):
    #         print(f"starting epoch: {epoch}")
    #         for iter, (X,y) in enumerate(trainset):
    #             x_=X.to(self.device)
    #             y_=y.to(self.device)
    #             z_ = Variable(torch.FloatTensor(np.random.normal(0, 1, (len(y_), self.z_dim)))).to(self.device)
    #             self.y_real_, self.y_fake_ = torch.ones(len(y_), 1).to(self.device), torch.zeros(len(y_), 1).to(self.device)
    #             # update D network
    #             self.D_optimizer.zero_grad()
    #             D_real = self.D(x_, y_)
    #             D_real_loss = self.MSE_loss(D_real, self.y_real_)
    #             G_ = self.G(z_, y_)
    #             D_fake = self.D(G_, y_)
    #             D_fake_loss = self.MSE_loss(D_fake, self.y_fake_)
    #             D_loss = D_real_loss + D_fake_loss
    #             self.train_hist['D_loss'].append(D_loss.item())
    #             D_loss.backward()
    #             self.D_optimizer.step()

    #              # update G network
    #             self.G_optimizer.zero_grad()
    #             z_=torch.rand((len(y_), self.z_dim)).to(self.device)
    #             G_ = self.G(z_, y_)
    #             D_fake = self.D(G_, y_)
    #             G_loss = self.MSE_loss(D_fake, self.y_real_)
    #             self.train_hist['G_loss'].append(G_loss.item())
    #             G_loss.backward()
    #             self.G_optimizer.step()
                
    #             print(f"D loss: {D_loss.item()} , G loss: {G_loss.item()}")

    #     print("Training finish!")
    #     return self.train_hist
    def train(self, trainset):
        self.train_hist = {'D_loss': [], 'G_loss': []}
        print('Training start!!')
        for epoch in range(self.epoch):
            print(f"Starting epoch: {epoch}")
            for iter, (X, y) in enumerate(trainset):
                x_ = X.to(self.device)
                y_ = y.to(self.device)
                z_ = torch.randn((len(y_), self.z_dim)).to(self.device)
                self.y_real_ = torch.ones(len(y_), 1).to(self.device)
                self.y_fake_ = torch.zeros(len(y_), 1).to(self.device)
                
                # Update D network
                self.D_optimizer.zero_grad()
                D_real = self.D(x_, y_)
                D_real_loss = self.MSE_loss(D_real, self.y_real_)
                G_ = self.G(z_, y_)
                D_fake = self.D(G_.detach(), y_)  # Detach G_ to avoid gradients
                # D_fake_loss = self.MSE_loss(D_fake, self.y_fake_)
                # D_loss = D_real_loss + D_fake_loss
                D_loss=D_real_loss
                self.train_hist['D_loss'].append(D_loss.item())
                D_loss.backward()
                self.D_optimizer.step()
    
                # # Update G network
                self.G_optimizer.zero_grad()
                z_ = torch.randn((len(y_), self.z_dim)).to(self.device)
                G_ = self.G(z_, y_)
                D_fake = self.D(G_, y_)
                G_loss = self.MSE_loss(D_fake, self.y_real_)
                self.train_hist['G_loss'].append(G_loss.item())
                G_loss.backward()
                self.G_optimizer.step()
    
                if iter % 100 == 0:
                    print(f"Iter: {iter}, D loss: {D_loss.item()}, G loss: {G_loss.item()}")
    
            print(f"Epoch {epoch} complete. D loss: {D_loss.item()}, G loss: {G_loss.item()}")
    
        print("Training finish!")
        return self.train_hist

        
    def get_parameters(self):
        params=self.D.state_dict()
        D_parameters=[]
        D_keys=[]
        for key,tensor in params.items():
            D_parameters.append(tensor)
            D_keys.append(key)
        self.D_params_key=D_keys

        params=self.G.state_dict()
        G_parameters=[]
        G_keys=[]
        for key,tensor in params.items():
            G_parameters.append(tensor)
            G_keys.append(key)
        self.G_params_key=G_keys

        return D_parameters,G_parameters
        
    def load_parameter(self,D_parameters,G_parameters):
        #load Dis parameter
        if isinstance(D_parameters, OrderedDict):
            self.D.load_state_dict(D_parameters)
        else:
            tensor=[]
            for par in D_parameters:
                tensor.append(torch.tensor(par))
            params = collections.OrderedDict(zip(self.D_params_key,tensor))
            self.D.load_state_dict(params)
        
        #load Gen parameters
        if isinstance(G_parameters, OrderedDict):
            self.G.load_state_dict(G_parameters)
        else:
            tensor=[]
            for par in G_parameters:
                tensor.append(torch.tensor(par))
            params = collections.OrderedDict(zip(self.G_params_key,tensor))
            self.G.load_state_dict(params)
            
    def sample(self,y,n_samples): #chưa logic lắm n_samples=len(y)
        if isinstance(y, pd.DataFrame):
            y=torch.tensor(y.values).to(self.device)
        
        z= torch.rand(n_samples,self.z_dim).to(self.device)
        fdata=self.G(z,y)
        y = y.unsqueeze(1) 
        fdata=torch.cat((y,fdata),dim=1).to(self.device)
        return fdata
