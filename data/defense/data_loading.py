
import os, json, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.extmath import randomized_svd
from scipy.signal import decimate
scaler=MinMaxScaler()
### Helper functions
def min_max_norm(data):
    min_ = min(data)
    max_ = max(data)
    return (data-min_) / (max_-min_)
def relu(signal):
    return np.maximum(0,signal.real)
#def preprocess(data):
#    return scaler.fit_transform(data.T).T
def preprocess(data):
    data = decimate(data,q=16)
    data = relu(data)
    return scaler.fit_transform(data.T).T
def low_rank_approx(data,rank=5):
        random_svd = randomized_svd(
            data,              # Input matrix (2D array)
            n_components=rank,   # Number of singular values/vectors to compute
            n_iter=5,       # Number of power iterations (default=5)
            random_state=42  # Random seed for reproducibility
        )
        return torch.from_numpy(random_svd[0][:,:rank] @ np.diag(random_svd[1])[:rank,:rank] @ random_svd[2][:rank,:])
def torch_relu(signal):
    return torch.clamp(signal.real, min=0)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

def build_dataset(dataset_dir_path, input_dim,train_val_or_test='train', return_labels=True):
    print(" building dataset")
    data_directory = os.path.join(dataset_dir_path,train_val_or_test)
    agents = sorted(os.listdir(data_directory))
    dataset = []
    targets = []
    labels = []
    thres = int(input_dim - 256**2)
    print(f"    {train_val_or_test}")
    for i,agent in enumerate(agents):

        data = np.load(os.path.join(data_directory,agent,'data.npy'))[:,thres:]#[:,thres:input_dim]
        #data = low_rank_approx(data,5)
        data = preprocess(data)
        data = torch.tensor(data).float()
        dataset.append(data)
        labels.append(torch.ones(data.size(0))*i)
        print(f"        {data.shape}")
    dataset = torch.cat(dataset, dim=0).unsqueeze(1)#.float()
    labels = torch.cat(labels).long()
    print(f"    {dataset.shape} | {labels.shape}")
    if return_labels: 
        return TensorDataset(dataset,labels)
    else:
        return MyDataset(dataset)


def build_specific_agent_dataset(dataset_dir_path, input_dim, agent, label, train_val_or_test='train'):
    print(f"--- building dataset for {agent}")
    data_directory = os.path.join(dataset_dir_path,train_val_or_test)
    thres = int(input_dim - 256**2)
    print(f"  - {train_val_or_test}")
    print(f"  - label is {label}")
    data = np.load(os.path.join(data_directory,agent,'data.npy'))[:,thres:]
    #data = low_rank_approx(data,5)
    data = preprocess(data)
    data = torch.tensor(data).unsqueeze(1).float()
    #data = torch.tensor(preprocess(data)).unsqueeze(1).float()
    labels = torch.ones(data.size(0))*label
    print(f"    {data.shape} | {labels.shape}")
    return data,labels# TensorDataset(data,labels)