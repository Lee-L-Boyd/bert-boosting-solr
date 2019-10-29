import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable




a = torch.tensor([[1.,3.,5.], [1.,6.,7.]])
b = torch.tensor([[4.,5.,6.], [400.,500.,600.]])
print("a " + str(a))
#print("b " + str(b))
#print("a * b " + str(a * b))
#print("sqrt(a * b) " + str(torch.sqrt(a * b)))

layernorm = nn.LayerNorm(a.size()[-1])
for p in layernorm.parameters():
  print(p)
print("layernorm a: " + str(layernorm(a[-1])))
u = a.mean(-1, keepdim=True)
s = (a - u).pow(2).mean(-1, keepdim = True)
print(u)
print(s)
print((a-u) / torch.sqrt(s + 1e-5))
