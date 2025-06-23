import torch

print("Pytorch version: ", torch.__version__)
print("Number of GPU: ", torch.cuda.device_count())
print("GPU Name: ", torch.cuda.get_device_name())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# import Q_Learning

# class DQN(Q_learning):





