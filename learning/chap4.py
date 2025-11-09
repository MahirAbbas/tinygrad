from tinygrad import Tensor, TinyJit, nn
from tinygrad.nn.datasets import mnist
from tinygrad.nn.optim import SGD
from tinygrad.helpers import trange
import time

X = Tensor.ones(3,4)

X_train, y_train, X_test, y_test = mnist(fashion=True)
# print(len(X_train))

labels = ["t-shirt","trouser","pullover","sandal","shirt","sneaker","bag","dress","coat","ankle boot"]

def validation_step(self, batch):
  Y_hat = self(*batch[:1])

@TinyJit

class Model():
  def __init__(self):
    self.layers = nn.Linear(in_features=784, out_features=10)
    self.layers: list[callable[[Tensor],Tensor]] = [
        lambda x: x.flatten(1), nn.Linear(784, 10)
    ]
    
  def __call__(self, X:Tensor) -> Tensor: 
    return X.sequential(self.layers)


model = Model()
opt = SGD(nn.state.get_parameters(model), lr=0.1)
epochs = 10

samples = Tensor.randint(256, high=X_train.shape[0])
# print(X_train[samples])

@TinyJit
@Tensor.train()
def train_step() -> Tensor:
  opt.zero_grad()
  samples = Tensor.randint(256, high=X_train.shape[0])
  loss = model(X_train[samples]).cross_entropy(y_train[samples]).backward()
  return loss.realize(*opt.schedule_step())

def accuracy() -> Tensor:
  return (model(X_test).argmax(axis=1) == y_test).mean()

test_acc = float('nan')
for i in (t:= trange(1000)):
  loss = train_step()
  if i%10 == 9: test_acc = accuracy().item()
  t.set_description(f"loss: {loss.item():6.2f}, accuracy: {test_acc:5.2f}")

print(accuracy().numpy())
