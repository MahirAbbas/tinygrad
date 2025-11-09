import matplotlib.pyplot as plt
import math
import time
import numpy as np
from tinygrad import Tensor, nn, TinyJit, GlobalCounters
from tinygrad.helpers import trange

class SyntheticRegressionData:
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        self.w = w
        self.b = b
        self.noise_level = noise
        self.num_train = num_train
        self.num_val = num_val
        self.n = num_train + num_val
        self.batch_size = batch_size

        self.noise = Tensor.randn(self.n, 1) * noise
        self.X = Tensor.randn(self.n, len(w))
        self.y = Tensor.matmul(self.X, w.reshape(-1,1)) + b + noise
    
        
class LinearRegressionScratch:
    def __init__(self):
        self.layers = nn.Linear(2,1)
        
    def __call__(self, x: Tensor):
        return self.layers(x)

if __name__=="__main__":
        
    data = SyntheticRegressionData(w = Tensor([2, -3.5]), b=4.2)

    X_train, y_train = data.X[:1000], data.y[:1000]
    X_val, y_val = data.X[1000:], data.y[1000:]

    model = LinearRegressionScratch()

    opt = nn.optim.SGD(nn.state.get_parameters(model), lr=0.03, weight_decay=1e-4)
    def mse_loss(y_hat: Tensor, y_true: Tensor):
        diff = y_hat - y_true
        return diff.square().mean()

    @TinyJit
    @Tensor.train()
    def train_step() -> Tensor:
        opt.zero_grad()
        bs = 32
        samples = Tensor.randint(32,high=X_train.shape[0])
        y_hat = model(X_train[samples])
        loss = mse_loss(y_hat,y_true=y_train[samples])
        loss.backward()
        return loss.realize(*opt.schedule_step())

    for i in trange(5000):
        GlobalCounters.reset()
        loss = train_step()
        
        
print(model.layers.weight.numpy())
print(model.layers.bias.numpy())




