import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.version

default_mod_weight_path = 'torch_mod_w.pt'

class MyModel(nn.Module):
    def __init__(self, input_dim:int=4, hid_dim:int=16, bias:bool=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.fc2 = nn.Linear(hid_dim, input_dim, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

    def save_weight(self, path=default_mod_weight_path):
        torch.save(self.state_dict(), path)
    
    def load_weight(self, path=default_mod_weight_path):
        self.load_state_dict(torch.load(path))

if __name__ == '__main__':
    import os

    model = MyModel()
    if not os.path.exists(default_mod_weight_path):
        model.save_weight()
    model.load_weight()
    model.eval()

    x = torch.Tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])
    y = model(x)

    print(torch.__version__) # 2.7.1+cu126
    print(x)
    print(y)
