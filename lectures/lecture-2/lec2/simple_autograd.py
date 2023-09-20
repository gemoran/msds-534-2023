
import torch

x = torch.tensor([1,2,3,4], dtype=torch.float, requires_grad=True)

h = x ** 2

##g = torch.log(h)

out = h.sum()

out.backward()

print(x.grad)


## foo = x.sum()
## foo.backward()

##x.grad.zero_()
