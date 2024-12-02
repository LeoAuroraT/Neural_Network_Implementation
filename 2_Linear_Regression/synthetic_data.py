import torch
from torch.utils.data import DataLoader, TensorDataset

class SyntheticRegressionData():
	"""
	generate synthetic data for linear regression
	step: tensor -> TensorDataset -> DataLoader
	"""
	def __init__(self, w, b, noise = 0.01, num_train = 1000, num_val = 1000, batch_size = 32):
		self.num_train = num_train
		self.num_val = num_val
		self.num_total = num_train + num_val

		#random features
		self.X = torch.randn(self.num_total, len(w))
		noise = torch.randn(self.num_total, 1)
		self.y = torch.matmul(self.X, w.reshape(-1, 1)) + b + noise

		#transform to TensorDataset
		self.train_data = TensorDataset(self.X[:num_train], self.y[:num_train])
		self.val_data = TensorDataset(self.X[num_train:], self.y[num_train:])

		self.batch_size = batch_size

	def get_dataloader(self, training_step = True):
		dataset = self.train_data if training_step else self.val_data

		return DataLoader(dataset, batch_size = self.batch_size, shuffle = training_step)
