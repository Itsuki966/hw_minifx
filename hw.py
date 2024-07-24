import numpy as np
import torch

alpha = 0.01
x = torch.tensor(1.0, requires_grad=True)

for epoch in range(1000):
    fx = x**2 + 4*x + 4
    x.retain_grad()
    fx.backward()

    x = x - alpha * x.grad

    x.grad = None

print(f"関数f(x)が最小値を取るときのxの値:{x}")