import torch
import DL_model
import DL_model_batch
import sys
import torch.nn as nn
import MyGan as Gan
import os
from torch.utils.data import TensorDataset, DataLoader
import wandb
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # p_t
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()


'''Client'''
class client():
    def __init__(self, cid, config, trainloader, testloader,device):
       self.cid=cid
       self.trainset=trainloader.to(device) #tensor
        
       # self.testset=testloader.to(device) #tensor
       self.labels = torch.unique(self.trainset[:,:1].squeeze())
       self.Gan=Gan.CGAN(config,device)
       if config["batch_norm"]==True:
           self.model=DL_model_batch.Net(config.Dropout_rate,device).to(device)   
       else:
           self.model=DL_model.Net(config.Dropout_rate,device).to(device)
       self.model_config=config
       
    def update_model(self,params):
        self.model.load_parameters(params)
    def update_Gan(self,params):
        self.Gan.load_parameters(params)
    def Gan_fit(self):
        X=self.trainset[:,1:].float()
        y=self.trainset[:,:1]
        y=y.squeeze().tolist()
        y=torch.tensor(y).long()
        dataset=TensorDataset(X,y)
        dataloader=DataLoader(dataset, batch_size=self.Gan.batch_size)
        hist=self.Gan.train(dataloader)
        for G_loss in hist['G_loss']:
            wandb.log({f"client {self.cid}/ G_loss":G_loss})
        for D_loss in hist['D_loss']:
            wandb.log({f"client {self.cid}/ D_loss":D_loss})    
        #code Generator fit data
    def model_fit(self,data,criterion, round=""):
        if len(data)!=0:
            X=data[:,1:].float()
            y=data[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        else:
            X=self.trainset[:,1:].float()
            y=self.trainset[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        dataset=TensorDataset(X,y)
        dataloader=DataLoader(dataset, batch_size=128)
        his=self.model.fit(dataloader,self.model_config.learning_rate,criterion,self.model_config.epochs,f"client {self.cid}",f"round {round}" )
        for loss in his["loss"]:
            wandb.log({f"client {self.cid}/model_loss": loss})
        # wandb.log({f"test_local_model_{self.cid}":his["loss"][-1]})
        #model classify fit
        return his
    def get_parameters(self):
        D_parameters,G_parameters=self.Gan.get_parameters()
        M_parameters=self.model.get_parameters()
        return M_parameters,D_parameters, G_parameters
    def evaluate(self,X,y,criterion):
        y=y.squeeze()
        y=y.long()
        loss, accuracy, precision,  recall, f1, mcc=self.model.evaluate(X,y,criterion)
        # wandb.log({f"client {self.cid}/model_accuracy":accuracy })
        return loss, accuracy, precision,  recall, f1, mcc
    def Gen_synthetic(self,required):
        required=required.squeeze()
        print(f"generating {len(required)} synthetic data")
        wandb.log({"quantity_of_syntheticData": len(required)})
        return self.Gan.sample(required,len(required))
    def Gen_fake(self,n_samples):
        y=torch.randint(0, 8, (n_samples,))
        y=y.squeeze().to(self.device)
        wandb.log({"quantity_of_syntheticData":n_samples})
        return self.Gan.sample(y,n_samples)
    
def fn_client(cid,config, trainloader, testloader,device)->client:
    return client(cid, config, trainloader, testloader,device)

'''server'''
class server():
    def __init__(self,config, trainloader,testloader,device):
       self.trainset= trainloader.to(device)#tensor
       self.testset= testloader.to(device) #tensor
       self.Gan=Gan.CGAN(config,device)
       if config["batch_norm"]==True:
           self.model=DL_model_batch.Net(config.Dropout_rate,device).to(device)   
       else:
           self.model=DL_model.Net(config.Dropout_rate,device).to(device)
       self.model_config=config
       self.device=device
        
    def update_model(self,params):
        self.model.load_parameters(params)
    def update_Gan(self,params):
        self.Gan.load_parameters(params)
    def Gan_freedata_fit(self,criterion,round="0"):
        his=self.Gan.freedata_train(self.model,criterion,round,"server_generator")
        for loss in his["loss"]:
            wandb.log({"server/Generator_loss":loss})
        for accuracy in his["accuracy"]:
            wandb.log({"server/Generator_accuracy":accuracy})     
        #code Generator fit data
    def model_fit(self,data,criterion):
        if len(data)!=0:
            X=data[:,1:].float()
            y=data[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        else:
            X=self.trainset[:,1:].float()
            y=self.trainset[:,:1]
            y=y.squeeze().tolist()
            y=torch.tensor(y).long()
        dataset=TensorDataset(X,y)
        dataloader=DataLoader(dataset, batch_size=128)
        his=self.model.fit(dataloader,self.model_config.learning_rate,criterion,self.model_config.epochs, " ","server_DeepLearningModel")
        for loss in his["loss"]:
            wandb.log({f"server/model_loss": loss})
        for accuracy in his["accuracy"]:
            wandb.log({f"server/model_accuracy": accuracy})
        return his
    def get_parameters(self):
        D_parameters,G_parameters=self.Gan.get_parameters()
        M_parameters=self.model.get_parameters()
        return M_parameters,D_parameters, G_parameters
    def evaluate(self,criterion):
        X=self.testset[:,1:].float().to(self.device)
        y=self.testset[:,:1].to(self.device)
        y=y.squeeze().tolist()
        y=torch.tensor(y).long()
        loss, accuracy, precision,  recall, f1, mcc=self.model.evaluate(X,y,criterion)
        return loss, accuracy, precision,  recall, f1, mcc
    def Gen_synthetic(self,required):
        required=required.squeeze()
        print(f"generating {len(required)} synthetic data")
        wandb.log({"quantity_of_syntheticData": len(required)})
        return self.Gan.sample(required,len(required))
    def Gen_fake(self,n_samples):
        y=torch.randint(0, 8, (n_samples,))
        y=y.squeeze().to(self.device)
        wandb.log({"quantity_of_syntheticData":n_samples})
        return self.Gan.sample(y,n_samples)


'''Frame work'''
class Federated_Learning():
    def __init__(self,config, trainloaders, testloaders, serverdata, testdata,device ):
        self.server=server(config, serverdata, testdata,device)
        self.testset=testdata.to(device)
        self.clients=[]
        self.n_clients=config.n_clients
        for i in range(self.n_clients):
            cl=fn_client(i,config, trainloaders[i], testloaders[i],device)
            self.clients.append(cl)
        self.criterion=torch.nn.CrossEntropyLoss()
        # self.criterion=FocalLoss()
        self.device=device
        self.synthetic_start_round=config.synthetic_start_round
        self.target_acc=config.target_acc
        self.num_fake=config.n_each_label
        self.best_model=DL_model.Net(config.Dropout_rate,device).to(device)
        self.best=0.00
    def save_best(self, total):
        if total> self.best:
            self.best=total
            self.best_model=self.server.model
    def client_M_update(self):
        M_params=self.server.model.get_parameters()
        for i in range(self.n_clients):
            self.clients[i].update_model(M_params)
    def server_M_update(self):
        params=self.clients[0].model.get_parameters()
        for i in range(len(params)):
            # for k in range(1,self.n_clients):
            for k in range(1,self.n_clients):
                params[i]=params[i]+self.clients[k].model.get_parameters()[i]
            params[i]=params[i]/self.n_clients
        self.server.update_model(params)
        print("finished AVG model")
    def server_M_update_base(self,list_weight):
        params=self.clients[0].model.get_parameters()
        sum=np.sum(list_weight)
        for i in range(len(params)):
            # for k in range(1,self.n_clients):
            for k in range(1,self.n_clients):
                params[i]=params[i]+(self.clients[k].model.get_parameters()[i]*list_weight[k])
            params[i]=params[i]/sum
                
        self.server.update_model(params)
        print("finished AVG model_base")

    
    def normal_FL(self,rounds):
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        for round in range(rounds):
            self.client_M_update()
            for i in range(self.n_clients):
                print(f"processing client {i}")
                fit_data=self.clients[i].trainset.to(self.device)
                self.clients[i].model_fit(fit_data,self.criterion,round)
            self.server_M_update()
            loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
            
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":round+1 })

            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":round+1 })
            print(f"round {round} accuracy for server: {accuracy} mcc {mcc}")
        return accuracy_hist



    def FEDADMM_simulation(self,rounds):
        history = {'loss': [], 'accuracy': []}
        rho=0.1
        dual_var = [[torch.zeros_like(param) for param in self.server.model.parameters()] for _ in range(self.n_clients)]
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        for round in range(rounds):
            # server_params,_,_=self.server.get_parameters()
            server_params=self.server.model.parameters()
            self.client_M_update()
            for i in range(self.n_clients):
                print(f"processing client {i}")
                fit_data=self.clients[i].trainset.to(self.device)
                X=fit_data[:,1:].float()
                y=fit_data[:,:1]
                y=y.squeeze().tolist()
                y=torch.tensor(y).long()
                dataset=TensorDataset(X,y)
                dataloader=DataLoader(dataset, batch_size=128)
                optimizer = optim.RMSprop(self.clients[i].model.parameters(), lr=0.0001, alpha=0.9)
                for epoch in range(15):
                    # Tính toán đầu ra mô hình
                    for batch_idx, (X, y) in enumerate(dataloader):
                        # client_params,_,_=self.clients[i].get_parameters()
                        client_params=self.clients[i].model.parameters()
                        y=y.to(self.device)
                        optimizer.zero_grad()
                        # client_params=client_params.requires_grad()
                        outputs = self.clients[i].model(X.float()).to(self.device)
                        loss,accuracy=self.clients[i].model.Quick_evaluate(outputs,y.long(),self.criterion)
                        # print(" loss original")
                        # print(loss)
                        # print(loss)
                        # params=server_params
                        # for z in range(len(params)):
                        # # for k in range(1,self.n_clients):
                        #     params[z]=client_params[z]-params[z]
                        #     params[z]=params[z].detach()
                        # print(params)
                        for cli in client_params:
                            cli.requires_grad
                        for ser in server_params:
                            ser.detach
                        total_norm = sum(torch.norm(cli-ser)**2 for cli,ser in zip(client_params,server_params))
                        dual_var_sum = sum(torch.sum(dual * (cli-ser)) for dual, cli,ser in zip(dual_var[i], client_params,server_params))
                        # params= torch.cat(params).requires_grad_()
                        # print(params)
                        loss = loss + (rho / 2) * total_norm + dual_var_sum
                        loss=loss.sum()
                        # print("loss after")
                        # print(loss)
                        loss.backward(retain_graph=True)
                        history['loss'].append(loss.item())
                        history['accuracy'].append(accuracy)
                        loss.backward()
                        optimizer.step()
                    print("done")
                with torch.no_grad():
                    for dual, cli,ser in zip(dual_var[i], client_params,server_params):
                        dual = dual + rho * (cli - ser)
                
                
            self.server_M_update()
            loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
            
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":round+1 })

            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":round+1 })
            print(f"round {round} accuracy for server: {accuracy} mcc {mcc}")
        return accuracy_hist
    def free_data_simulation_v2(self,rounds):
            #sinh dữ liệu random 
            accuracy_hist=[]
            print(f"initial setup for free data training")
            accuracy_hist=[]
            print(f"initial setup for free data training")
            loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
            print(f"----------initial accuracy {accuracy} ----------")
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
                
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
                
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":0 })
                
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":0 })
                
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":0})
    
            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
            round_fake_data=torch.empty(1, 29).to(self.device)
            for round in range(rounds):
                if round>self.synthetic_start_round-1:
                    f=self.server.Gen_fake(10000)
                    print(f"-------------ready for round {round}-------------")
                    # self.client_M_update()
                    round_fake_data=f.to(self.device)
                    round_fake_data.detach()
                self.client_M_update()
                for i in range(self.n_clients):
                    fit_data=torch.cat((self.clients[i].trainset,round_fake_data),dim=0).to(self.device)
                    print(f"processing client {i}")
                    fit_data=fit_data.detach()
                    self.clients[i].model_fit(fit_data,self.criterion,round)
                self.server_M_update()
                loss,accuracy=self.server.evaluate(self.criterion)
                self.save_best(accuracy)
                accuracy_hist.append(accuracy)
                print(f"round {round} accuracy for server: {accuracy}")
                print(f"starting training for generator")
                self.server.Gan_freedata_fit(self.criterion,round)
                wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
                wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
                wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
                wandb.log({f"framework/server_accuracy": accuracy, "rounds": round+1})
            return accuracy_hist
    def free_data_simulation_v5(self,rounds):
        #Gan chỉ tham gia khi đạt được accuracy
        print(f"initial setup for freedata version 3 training")
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.client_M_update()
            print(f"completed to synchronized server parameters to all clients")
            if accuracy_hist[-1]>self.target_acc:
                print(f"starting training for generator")
                self.server.Gan_freedata_fit(self.criterion,round)
            for i in range(self.n_clients):
                fit_data=self.clients[i].trainset.to(self.device)
                if accuracy_hist[-1]>self.target_acc:
                    full_range = list(range(8))
                    value_list = [x.item() for x in list(self.clients[i].labels)]
                    missing_values = [x for x in full_range if x not in value_list]
                    tensor_fake=torch.tensor(missing_values)
                    tensor_fake=tensor_fake.repeat(1,self.num_fake)
                    print(tensor_fake)
                    tensor_fake=tensor_fake.long().to(self.device)
                    f=self.server.Gen_synthetic(tensor_fake)
                    f=f.detach()
                    fit_data=torch.cat((self.clients[i].trainset,f),dim=0).to(self.device)
                    num_elements = fit_data.size(0)
                    # Tạo một permutated index ngẫu nhiên
                    permutated_indices = torch.randperm(num_elements)
                    # Sử dụng permutated index để tráo đổi tensor
                    fit_data = fit_data[permutated_indices]
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
                # self.clients[i].evaluate(self.server.testset[:,1:].float(),self.server.testset[:,:1].float(),self.criterion)
            self.server_M_update()
            loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
            
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":round+1 })

            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":round+1 })
            print(f"round {round} accuracy for server: {accuracy} mcc {mcc}")
            accuracy_hist
            accuracy_hist.to_csv("Huy/fine/Output /V4/accuracy/hist.csv")
        return accuracy_hist
    def free_data_simulation_v4(self,rounds):
        #Gan chỉ tham gia khi chuẩn bị đánh giá
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.client_M_update()
            print(f"completed to synchronized server parameters to all clients")
            if round> self.synthetic_start_round-1:
                print(f"starting training for generator")
                self.server.Gan_freedata_fit(self.criterion,round)
            for i in range(self.n_clients):
                fit_data=self.clients[i].trainset.to(self.device)
                if round> self.synthetic_start_round-1:
                    full_range = list(range(8))
                    value_list = [x.item() for x in list(self.clients[i].labels)]
                    missing_values = [x for x in full_range if x not in value_list]
                    tensor_fake=torch.tensor(missing_values)
                    tensor_fake=tensor_fake.repeat(1,self.num_fake)
                    print(tensor_fake)
                    tensor_fake=tensor_fake.long().to(self.device)
                    f=self.server.Gen_synthetic(tensor_fake)
                    f=f.detach()
                    fit_data=torch.cat((self.clients[i].trainset,f),dim=0).to(self.device)
                    num_elements = fit_data.size(0)

                    # Tạo một permutated index ngẫu nhiên
                    permutated_indices = torch.randperm(num_elements)
                    
                    # Sử dụng permutated index để tráo đổi tensor
                    fit_data = fit_data[permutated_indices]
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
                # self.clients[i].evaluate(self.server.testset[:,1:].float(),self.server.testset[:,:1].float(),self.criterion)
            self.server_M_update()
            loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
            
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":round+1 })

            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":round+1 })
            print(f"round {round} accuracy for server: {accuracy} mcc {mcc}")
        return accuracy_hist

    
    def free_data_simulation_v3(self,rounds):
        #sinh đúng dữ liệu thiếu
        accuracy_hist=[]
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.client_M_update()
            print(f"completed to synchronized server parameters to all clients")
            for i in range(self.n_clients):
                fit_data=self.clients[i].trainset.to(self.device)
                if round> self.synthetic_start_round-1:
                    full_range = list(range(8))
                    value_list = [x.item() for x in list(self.clients[i].labels)]
                    missing_values = [x for x in full_range if x not in value_list]
                    tensor_fake=torch.tensor(missing_values)
                    tensor_fake=tensor_fake.repeat(1,10000 )
                    print(tensor_fake)
                    tensor_fake=tensor_fake.long().to(self.device)
                    f=self.server.Gen_synthetic(tensor_fake)
                    f=f.detach()
                    fit_data=torch.cat((self.clients[i].trainset,f),dim=0).to(self.device)
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
                # self.clients[i].evaluate(self.server.testset[:,1:].float(),self.server.testset[:,:1].float(),self.criterion)
            self.server_M_update()
            loss,accuracy=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            print(f"round {round} accuracy for server: {accuracy}")
            print(f"starting training for generator")
            self.server.Gan_freedata_fit(self.criterion,round)
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds": round+1})
        return accuracy_hist
        
    def free_data_simulation_v1(self,rounds):
        # sử dụng dữ liệu của server để có thể huấn luyện ban đầu cho GAN
        accuracy_hist=[]
        print(self.server.Gan.G)
        print(f"initial setup for free data training")
        self.server.model_fit([],self.criterion)
        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"initial server model, accuracy: {accuracy}, loss: {loss}")
        total_syntheticdata=self.server.trainset
        round_fake_data=torch.empty(1, 29).to(self.device)
        for round in range(rounds):
            self.server.Gan_freedata_fit(self.criterion,round )
            if round>5:
                f=self.server.Gen_fake(10000)
                total_syntheticdata=torch.cat((total_syntheticdata,f),dim=0).detach().to(self.device)
                print(f"-------------ready for round {round}-------------")
                self.client_M_update()
                round_fake_data=f.to(self.device)
                round_fake_data.detach()
            print(f"complete to update client's M model")
            for i in range(self.n_clients):
                fit_data=torch.cat((self.clients[i].trainset,round_fake_data),dim=0)
                print(f"processing client {i}")
                fit_data=fit_data.detach()
                self.clients[i].model_fit(fit_data,self.criterion,round)
            self.server_M_update()
            loss,accuracy=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds": round})
            print(f"round {round} accuracy for server: {accuracy}")

        loss,accuracy=self.server.evaluate(self.criterion)
        print(f"----------last accuracy {accuracy} ----------")
        return accuracy_hist
    def Gan_for_all_clients(self,rounds):
        print(f"initial setup for training")
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        for round in range(rounds):
            synthetic_data=torch.empty(1, 29).to(self.device)
            for i in range(self.n_clients):
                self.client_M_update()
                self.clients[i].Gan_fit()
                required=torch.tensor(self.clients[i].labels)
                required=required.repeat(1,10000)
                required=required.long().to(self.device)
                f=self.clients[i].Gen_synthetic(required)
                f=f.detach()
                synthetic_data=torch.cat((synthetic_data,f),dim=0).to(self.device)
                self.clients[i].model_fit(self.clients[i].trainset,self.criterion,round)

            
            self.server_M_update()
            fit_data=synthetic_data
            num_elements = fit_data.size(0)
            permutated_indices = torch.randperm(num_elements)
            fit_data = fit_data[permutated_indices]
            self.server.model_fit(fit_data,self.criterion)
            loss, accuracy, precision,recall, f1, mcc=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
            
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":round+1 })

            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":round+1 })
            print(f"round {round} accuracy for server: {accuracy} mcc {mcc}")
    
    def dataSharing_simulation(self,rounds):
        accuracy_hist=[]
        accuracy_hist=[]
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        a=0.03
        for round in range(rounds):
            self.client_M_update()
            for i in range(self.n_clients):
                print(f"processing client {i}")
                _,share=train_test_split(self.server.trainset,test_size=a)
                fit_data=torch.cat((self.clients[i].trainset,share),dim=0).to(self.device)
                num_elements = fit_data.size(0)

                    # Tạo một permutated index ngẫu nhiên
                permutated_indices = torch.randperm(num_elements)
                    
                    # Sử dụng permutated index để tráo đổi tensor
                fit_data = fit_data[permutated_indices]
                self.clients[i].model_fit(fit_data,self.criterion,round)
            self.server_M_update()
            loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
            
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":round+1 })

            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":round+1 })
            print(f"round {round} accuracy for server: {accuracy} mcc {mcc}")
        return accuracy_hist

    def FEDBS_simulation(self,rounds):
        accuracy_hist=[]
        hist_round=np.zeros(self.n_clients)
        print(f"initial setup for free data training")
        loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
        print(f"----------initial accuracy {accuracy} ----------")
        accuracy_hist.append(accuracy)
        wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
        wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": 0})
            
        wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
        wandb.log({f"framework/server_accuracy": accuracy, "rounds":0 })
            
        wandb.define_metric(f"framework/server_precision", step_metric="rounds")
        wandb.log({f"framework/server_precision": precision, "rounds":0 })
            
        wandb.define_metric(f"framework/server_recall", step_metric="rounds")
        wandb.log({f"framework/server_recall": recall, "rounds":0 })
            
        wandb.define_metric(f"framework/server_f1", step_metric="rounds")
        wandb.log({f"framework/server_f1": f1, "rounds":0})

        wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
        wandb.log({f"framework/server_mcc": mcc, "rounds":0 })
        for round in range(rounds):
            self.client_M_update()
            for i in range(self.n_clients):
                print(f"processing client {i}")
                fit_data=self.clients[i].trainset.to(self.device)
                hist=self.clients[i].model_fit(fit_data,self.criterion,round)
                hist_round[i]=hist["loss"][-1]
            sum=np.sum(hist_round)
            for i in range(len(hist_round)):
                hist_round[i]=(sum-hist_round[i])/sum
            print(hist_round)
            self.server_M_update_base(hist_round)
            loss, accuracy, precision,  recall, f1, mcc=self.server.evaluate(self.criterion)
            self.save_best(accuracy)
            accuracy_hist.append(accuracy)
            wandb.define_metric(f"framework/max_accuracy", step_metric="rounds")
            wandb.log({f"framework/max_accuracy": max(accuracy_hist), "rounds": round+1})
            
            wandb.define_metric(f"framework/server_accuracy", step_metric="rounds")
            wandb.log({f"framework/server_accuracy": accuracy, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_precision", step_metric="rounds")
            wandb.log({f"framework/server_precision": precision, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_recall", step_metric="rounds")
            wandb.log({f"framework/server_recall": recall, "rounds":round+1 })
            
            wandb.define_metric(f"framework/server_f1", step_metric="rounds")
            wandb.log({f"framework/server_f1": f1, "rounds":round+1 })

            wandb.define_metric(f"framework/server_mcc", step_metric="rounds")
            wandb.log({f"framework/server_mcc": mcc, "rounds":round+1 })
            print(f"round {round} accuracy for server: {accuracy} mcc {mcc}")
        return accuracy_hist
        
