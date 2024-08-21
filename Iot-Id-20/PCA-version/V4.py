import pandas as pd 
import FL
import data_work
import torch
from sklearn.model_selection import train_test_split
import wandb
import os
# Đặt GPU mà bạn muốn sử dụng
# Thay '0' bằng số index của GPU mà bạn muốn sử dụng
gpu_index = 0
if torch.cuda.is_available():
    torch.cuda.set_device(gpu_index)
    print(f"Đã chọn GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")
else:
    print("Không có GPU nào có sẵn hoặc PyTorch không được cài đặt với hỗ trợ CUDA.")
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")
# device="cpu"
print(f"device : {device}")

def main():
    num_clients=16
    server_data = pd.read_csv("./dataset/server_data.csv", index_col=0)
    test_data=pd.read_csv("./dataset/test_data.csv", index_col=0)
    server_data=torch.tensor(server_data.values).to(device)
    test_data=torch.tensor(test_data.values).to(device)
    trainloaders=[]
    testloaders=[]
    for i in range(num_clients):
        temp=pd.read_csv(f"./dataset/trainloader_{i}.csv", index_col=0)
        
        trainloaders.append(temp)
        trainloaders[i]=torch.tensor(trainloaders[i].values).to(device)

        temp=pd.read_csv(f"./dataset/testloader_{i}.csv", index_col=0)
        testloaders.append(temp)
        testloaders[i]=torch.tensor(testloaders[i].values).to(device)
        # testloaders=[]
        print(trainloaders[i][:,:1].min(), trainloaders[i][:,:1].max())
    print(f"number of clients : {len(trainloaders)}")
    config=dict(
    target_acc=0.75,
    Gan_epochs=500,
    noise_size=100,
    n_features=16,
    n_classes=5,
    lrG=0.3,
    lrD=0.003,
    epochs=15,
    classes=8,
    batch_size=128,
    G_batch_size=128,
    GAC_batch_size=128,
    learning_rate=0.001,
    IID=False,
    n_clients=num_clients,
    n_cli_samples=len(trainloaders[0]),
    test_size=len(test_data),
    criterion=torch.nn.CrossEntropyLoss(),
    Dropout_rate=0.1,
    batch_norm=False,
    synthetic_start_round= 10,
    n_each_label=9000
   )
    with wandb.init(project="iotid20", name=f"V4_9k_r10", config=config):
        conf=wandb.config
        ser_test=FL.Federated_Learning(conf,trainloaders,testloaders,server_data,test_data,device)
        ac=ser_test.free_data_simulation_v4(25)
        wandb.finish()
for i in range(1):
    main()