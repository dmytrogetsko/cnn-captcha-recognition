import torch.nn as nn
import torch

class CaptchaCNN(nn.Module):
    def __init__(self, num_classes, captcha_length):
        super(CaptchaCNN, self).__init__()
        self.captcha_length = captcha_length
        self.num_classes = num_classes
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 128)
            out = self.conv_layers(dummy_input)
            self.flattened_size = out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1024, captcha_length * num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(-1, self.captcha_length, self.num_classes)
        return x