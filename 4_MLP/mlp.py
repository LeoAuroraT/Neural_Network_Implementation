"""
example code of the implementation of Multilayer Perceptron with dropout for classification of mnist
"""

import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, dropout_rate):
		super(MLP, self).__init__()
		self.fc1 = nn.Linear(input_dim, 256)
		self.fc2 = nn.Linear(256, 256)
		self.fc3 = nn.Linear(256, output_dim)

		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, x):
		x = self.fc1(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc2(x)
		x = self.relu(x)
		x = self.dropout(x)

		x = self.fc3(x) #raw logits, no activation if you use cross entropy loss

		return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = FashionMNIST(
    root='./data',         # Directory to download/load the dataset
    train=True,            # Load the training set
    transform=transform,   # Apply the transformations
    download=True          # Download the dataset if it's not already downloaded
)

test_dataset = FashionMNIST(
    root='./data',         # Directory to download/load the dataset
    train=False,           # Load the test set
    transform=transform,   # Apply the transformations
    download=True          # Download the dataset if it's not already downloaded
)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(train_dataset, batch_size = 64, shuffle = False)



input_dim = 28 * 28
output_dim = 10
dropout_rate = 0.2

learning_rate = 0.001
num_epochs = 100

#define model, loss function, optimizer
model = MLP(input_dim = input_dim, output_dim = output_dim, dropout_rate = dropout_rate).to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training
model.train()
for epoch in range(num_epochs):
	running_loss = 0.0
	for images, labels in train_loader:
		batch_size = images.size(0)
		images = images.view(batch_size, -1)

		#move data to device
		images = images.to(device)
		labels = labels.to(device)

		"""
		PyTorch allows you to register hooks (e.g., for debugging, logging, or gradient modifications). 
		These hooks are triggered when you call model(images) but are skipped if you call model.forward(images) directly.
		"""
		output = model(images)

		loss = loss_function(output, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss += loss.item()

	print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


#evaluation
model.eval()

correct = 0
total = 0

with torch.no_grad():
	for images, labels in test_loader:
		batch_size = images.size(0)
		images = images.view(batch_size, -1)

		#move data to device
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		_, predicted_labels = torch.max(outputs, 1) #get the predicted labels as the class with largest predicted prob
		total+=labels.size(0)
		correct += (predicted_labels == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")

