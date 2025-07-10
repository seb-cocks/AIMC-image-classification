import torch.nn as nn
# %%
class MainModel(nn.Module):
    def __init__(self, num_classes):
        super(MainModel, self).__init__()

        # C1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=1)

        self.relu = nn.ReLU()
        
        # S1
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        # C2
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=(5, 5), stride=1)
        
        # S2
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        # Flatten
        self.flat = nn.Flatten()

        # FC
        self.fc = nn.Linear(in_features=588, out_features=num_classes)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = self.flat(x)
        x = self.fc(x)

        return x
