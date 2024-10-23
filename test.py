from tinygrad import Tensor
from tinygrad import dtypes


a = Tensor([2.0])
b = Tensor([3.0])
c = a.dot(b)

d = c.numpy()
print(d)

