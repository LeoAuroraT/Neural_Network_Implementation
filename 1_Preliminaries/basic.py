import torch
import pandas as pd
import os


# Part 1: create tensor
x = torch.arange(12, dtype = torch.float32)

print(x)
print(x.numel()) #count the element
print(x.shape)

x_reshape = x.reshape(3, 4)
print(x_reshape)


X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0)) 
print(torch.cat((X, Y), dim=1))


# Part 2: load from .csv

data_file = os.path.join('..', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
print(data)

inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
print(targets)

inputs = inputs.fillna(inputs.mean())
print(inputs)

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))

print(X)
print(y)


