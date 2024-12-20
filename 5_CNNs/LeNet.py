"""
Ref: GradientBased Learning Applied to Document Recognition 1998
"""

import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

class LeNet(nn.Module):
	def __init__(self, input_channel, output_channel):
		super(LeNet, self).__init__()
		self.conv1 = nn.Conv2d(in_channels = input_channel, out_channels = 6, kernel_size = 5, padding = 2)
		self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)

		self.avepool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

		self.acti_function = nn.Sigmoid()

		self.fc1 = nn.Linear(16*5*5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, output_channel)

	def forward(self, x):
		x = self.conv1(x)
		x = self.acti_function(x)
		x = self.avepool1(x)

		x = self.conv2(x)
		x = self.acti_function(x)
		x = self.avepool1(x)

		x = x.view(x.size(0), -1)  # Flatten to (batch_size, num_features)

		x = self.fc1(x)
		x = self.fc2(x)
		x = self.fc3(x)

		return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
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


learning_rate = 0.001
num_epochs = 10

model = LeNet(input_channel = 3, output_channel = 10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

model.train()
for epoch in range(num_epochs):
	running_loss = 0.0
	for images, labels in train_loader:
		images = images.to(device)
		labels = labels.to(device)

		output = model(images)
		loss = criterion(output, labels)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		running_loss+=loss.item()
	print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
	for images, labels in test_loader:
		images = images.to(device)
		labels = labels.to(device)

		output = model(images)
		_, predicted_labels = torch.max(output, 1)
		total+=labels.size(0)
		correct += (predicted_labels == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")


