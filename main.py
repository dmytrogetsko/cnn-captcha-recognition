import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from core.dataset import CaptchaDataset
from core.model import CaptchaCNN
from utils.validation import visualize_predictions

### Train ###

batch_size = 64
epochs = 10
captcha_length = 5
num_classes = 62

transform = transforms.Compose([
    transforms.Resize((64, 128)),
    transforms.ToTensor()
])

full_dataset = CaptchaDataset('./samples', captcha_length, transform=transform)

# Separate dataset to train (80%) and predict (20%) captchas
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = total_size - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss function, optimizator
model = CaptchaCNN(num_classes=num_classes, captcha_length=captcha_length)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    model.train()

    for images, labels in train_loader:
        outputs = model(images)
        loss = 0
        for i in range(captcha_length):
            loss += criterion(outputs[:, i, :], labels[:, i])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')


### Output ###

visualize_predictions(full_dataset, model, val_loader, num_examples=5)