import torch 
from torch import nn

class model(nn.Module): 
    def __init__(self, ):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear2(self.linear1(x))

if __name__ == "__main__":
    net = model() 
    __import__("pprint").pprint(net.state_dict())
