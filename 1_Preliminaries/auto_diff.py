import torch

x = torch.arange(4.0)

# telling PyTorch to keep track of computations involving x so we can compute derivatives (gradients) with respect to x later.
x.requires_grad_(True)

# attribute of x to store all gradient, default to None now
print(x.grad)

#vector input, scalar output
y = 2 * torch.dot(x, x)
print(y)

#take the gradient of y with respect to x
y.backward()

print(x.grad)
print(x.grad == 4 * x)

# clear the gradients and another function
x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

# now the input is vector but output is also a vector
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y))) # give equal weight to each element in y, same as y.sum().backward()
print(x.grad) #The derivative of y[i] = x[i] * x[i] with respect to x[i] is 2 * x[i]



"""
What Happens When We Detach y:

y = x * x: This is an element-wise square of x, producing y = [1., 4., 9., 16.]. Since x.requires_grad=True, y is part of the computation graph, and PyTorch tracks its relationship to x.
u = y.detach(): detach() creates a new tensor u that has the same values as y but breaks the gradient-tracking relationship with x. Now, u is treated as a constant in subsequent calculations.
z = u * x: Here, z depends on x directly but not on y in a way that allows gradients to flow back to y and x. Since u is detached, the gradient of z with respect to x only considers u as a fixed value, like a constant.
Result of z.sum().backward():

z.sum().backward() computes the gradient of z with respect to x.
Since z = u * x, where u is treated as a constant, the gradient dz/dx is simply u, so x.grad becomes [0., 2., 4., 6.], which is exactly u.
"""
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
print(x.grad == u)



"""
Here, we calculate y.sum().backward(), which does include the full gradient of y = x * x with respect to x:

Gradient of y.sum() with respect to x: The derivative of y[i] = x[i]^2 with respect to x[i] is 2 * x[i].
"""

x.grad.zero_()  # Clear the gradient
y.sum().backward()
print(x.grad)