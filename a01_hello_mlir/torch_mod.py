import torch
import torch.nn as nn
import torch.nn.functional as F

default_mod_weight_path = 'torch_mod_w.pt'
default_onnx_path = 'torch_mod.onnx'

class MyModel(nn.Module):
    def __init__(self, seq_len:int=4096, vocab_size:int=32000, 
            embed_dim:int=256, hid_dim:int=512, bias:bool=True,
            temperature:float=0.7):
        super().__init__()
        self.seq_len = seq_len
        self.temperature = temperature
        self.em = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hid_dim, bias=bias)
        self.fc2 = nn.Linear(hid_dim, vocab_size, bias=bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.em(x) # (batch, len, embed_dim)
        x = self.fc1(x) # (batch, len, hid_dim)
        x = F.relu(x)
        logits = self.fc2(x) # (batch, len, vocab_size)
        last_logits = logits[:, -1, :] # (batch, vocab_size)
        scaled_logits = last_logits / self.temperature
        probabilities = F.softmax(scaled_logits, dim=-1)
        sampled_tokens = torch.argmax(probabilities, dim=-1) # (batch,)
        return sampled_tokens

    def save_weight(self, path=default_mod_weight_path):
        torch.save(self.state_dict(), path)
    
    def load_weight(self, path=default_mod_weight_path):
        self.load_state_dict(torch.load(path))

    def example_input(self, batch_size:int) -> torch.Tensor:
        """
        样例输入 (batch_size, seq_len)
        """
        token_ids = torch.randint(0, vocab_size, (batch_size, self.seq_len))
        return token_ids

    def save_onnx(self, example_input:torch.Tensor, path=default_onnx_path):
        # pip install --upgrade onnx onnxscript
        import os
        if not os.path.exists(path):
            self.eval()
            # torch.onnx.export(
            #     model=self,
            #     args=(example_input, ),
            #     f = path,
            #     dynamo=True
            # )
            onnx_program = torch.onnx.export(self, (example_input, ), dynamo=True)
            onnx_program.save(path)


def fix_seed(seed:int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed) 

if __name__ == '__main__':
    import os
    fix_seed(1)

    model = MyModel()
    if not os.path.exists(default_mod_weight_path):
        model.save_weight()
    model.load_weight()
    model.eval()

    x = torch.Tensor([
        [0] * (4096 - 4) + [1,2,3,4],
        [0] * (4096 - 4) + [5,6,7,8]
    ]).type(torch.int64)
    print("input", x, sep='\n')

    # tensor([12971, 31737])
    print("torch infer", model(x), sep='\n')

    model.save_onnx(example_input=x)

    from onnx_infer import infer

    print("onnx infer")
    # [array([12971, 31737], dtype=int64)]
    print(infer(
        path=default_onnx_path,
        input_data=x.numpy()
    ))
