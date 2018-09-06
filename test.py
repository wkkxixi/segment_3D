import torch

x = torch.randn(2, 3)
print(torch.cat((x,x,x), 0).size())
print(torch.cat((x,x,x), 1).size())