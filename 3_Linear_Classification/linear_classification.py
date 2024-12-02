import torch
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


class LinearClassification():
    def __init__(self, num_features, num_classes, lr):
        self.w = torch.randn(num_features, num_classes, requires_grad=True)
        self.b = torch.zeros(num_classes, requires_grad=True)
        self.lr = lr

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, logits, true_label):
        #true_label_in_one_hot * log(softmax(raw_predicted_value, i.e ligits))
        y_one_hot = torch.nn.functional.one_hot(true_label, num_classes=logits.size(1)).float()
        log_prods = torch.log_softmax(logits, dim = 1)
        loss = -torch.sum(y_one_hot*log_prods)/true_label.size(0)

        return loss


    def sgd_step(self):
        with torch.no_grad():
            self.w -= self.lr * self.w.grad
            self.b -= self.lr * self.b.grad
            #zero gradients after updating
            self.w.grad.zero_()
            self.b.grad.zero_()





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



input_dim = 28*28

model = LinearClassification(num_features = input_dim, num_classes = 10, lr = 0.001)


epoches = 100
for epoch in range(epoches):
    running_loss = 0.0
    for images, labels in train_loader:
        batch_size = images.size(0)
        images = images.view(batch_size, -1)  # [batch_size, 28*28]

        logits = model.forward(images)

        loss = model.loss(logits, labels)

        loss.backward()
        model.sgd_step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")



# testing
correct = 0
total = 0

model.w.requires_grad = False
model.b.requires_grad = False

for images, labels in test_loader:
    batch_size = images.size(0)
    images = images.view(batch_size, -1)  # [batch_size, 28*28]

    logits = model.forward(images)
    predictions = torch.argmax(logits, dim=1)

    correct += (predictions == labels).sum().item()
    total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")