import torch
import torch.nn as nn
import torch.nn.functional as F

default_mod_weight_path = 'torch_mod_w.pt'
default_onnx_path = 'torch_mod.onnx'

class MyModel(nn.Module):
    def __init__(self, input_dim:int=4, hid_dim:int=16, bias:bool=True):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(input_dim, hid_dim, bias=bias)
        self.fc2 = nn.Linear(hid_dim, input_dim, bias=bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=-1)
        return x

    def save_weight(self, path=default_mod_weight_path):
        torch.save(self.state_dict(), path)
    
    def load_weight(self, path=default_mod_weight_path):
        self.load_state_dict(torch.load(path))

    def example_input(self, *prefix_shape:int) -> torch.Tensor:
        """
        样例输入 model.example_input(2, 3) -> (2, 3, input_dim)
        """
        shape = list(prefix_shape) + [self.input_dim]
        return torch.rand(*shape)

    def to_onnx(self, example_input:torch.Tensor, path=default_onnx_path):
        # pip install --upgrade onnx onnxscript
        self.eval()
        torch.onnx.export(
            model=self,
            args=(example_input, ),
            f = path,
            input_names=['input'],
            output_names=['output'],
            dynamo=True
        )


def fix_seed(seed:int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed) 

if __name__ == '__main__':
    import os

    model = MyModel()
    if not os.path.exists(default_mod_weight_path):
        fix_seed(1)
        model.save_weight()
    model.load_weight()
    model.eval()

    x = torch.Tensor([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8]
    ])
    y = model(x)

    print(torch.__version__) # 2.9.0.dev20250719+cpu
    print(x)

    # tensor([[0.2262, 0.2880, 0.2197, 0.2661],
    #         [0.2218, 0.3190, 0.1965, 0.2627]], grad_fn=<SoftmaxBackward0>)
    print(y)

    model.to_onnx(x)
