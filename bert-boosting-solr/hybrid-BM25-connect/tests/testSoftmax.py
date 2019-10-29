
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
a = torch.tensor([[1.,2,3],[40,5,6]])
y = F.softmax(a)
print(y)
