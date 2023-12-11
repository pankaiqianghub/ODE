import pdb

import torch
from torch import autograd

x = torch.rand(3, 4)
x.requires_grad_()

y = x ** 2
grad = autograd.grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y), create_graph=True)[0]
pdb.set_trace()
grad2 = autograd.grad(outputs=grad, inputs=x, grad_outputs=torch.ones_like(grad))[0]
print(grad2)