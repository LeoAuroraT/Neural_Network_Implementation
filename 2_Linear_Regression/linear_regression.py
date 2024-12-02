import torch
from synthetic_data import SyntheticRegressionData


class LinearRegression():
	def __init__(self, num_features, lr, sigma = 0.01):
		#random initialization of w and b
		"""
		In PyTorch, requires_grad=True is a property of a tensor that tells PyTorch to track operations on this tensor 
		and compute gradients for it during backpropagation.
		For building the computation graph
		"""
		self.w = torch.normal(0, sigma, (num_features, 1), requires_grad=True)
		self.b = torch.zeros(1, requires_grad=True)
		self.lr = lr

	def forward(self, X):
		return torch.matmul(X, self.w) + self.b

	def loss(self, y_hat, y):
	    L = (y_hat - y) ** 2 / 2
	    return L.mean()

	def sgd_step(self):
		"""
		torch.no_grad() is a context manager that disables gradient computation. 
		It’s typically used when you want to perform operations on tensors without tracking them in the computational graph.
		"""
		with torch.no_grad():
			self.w -= self.lr * self.w.grad
			self.b -= self.lr * self.b.grad
			#zero gradients after updating
			self.w.grad.zero_()
			self.b.grad.zero_()


true_w = torch.tensor([2.0, -3.4])
true_b = 4.2

synthetic_data = SyntheticRegressionData(w=true_w, b=true_b)

train_loader = synthetic_data.get_dataloader(training_step = True)
test_loader = synthetic_data.get_dataloader(training_step = False)

model = LinearRegression(num_features = 2, lr = 0.001)
epoches = 100

for epoch in range(epoches):
	for batch_x, batch_y in train_loader:
		y_hat = model.forward(batch_x)
		"""
		by the computation graph, pytorch track which tensor contribute to the loss, here is w and b
		"""
		loss = model.loss(y_hat, batch_y)
		loss.backward()
		model.sgd_step()

	print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


# Print learned parameters
print("Learned w:", model.w.detach().flatten())
print("Learned b:", model.b.item())
print("True w:", true_w)
print("True b:", true_b)


"""
Note:
Gradients Accumulate:
If w and b are used in multiple functions (e.g., different loss functions), and .backward() is called on those losses sequentially without clearing the gradients, 
the gradients for w and b will accumulate (i.e., added together) in their .grad attributes.
PyTorch accumulates gradients by default to support scenarios where multiple losses contribute to the parameter updates.
"""