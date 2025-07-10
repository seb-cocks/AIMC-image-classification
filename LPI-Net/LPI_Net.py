import torch
import torch.nn as nn

class module_thing(nn.Module):
    def __init__(self):
        super(module_thing, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,3), stride=1,padding=(0,1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=1,padding=(1,0))

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1, stride=1)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,3), stride=1,padding=(0,1))
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,1), stride=1,padding=(1,0))

        self.conv6 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1)

    def forward(self, x):

        x = self.pool1(x)
        x0 = x

        x1 = self.conv1(x)
        x2 = self.conv2(x)

        x_cat1 = torch.cat((x1,x2), dim=1)
        x  = self.conv3(x_cat1)

        x1 = self.conv4(x)
        x2 = self.conv5(x)

        x = torch.cat((x1,x2,x_cat1), dim=1)

        x = self.conv6(x)

        x+=x0

        return x

class MainModel(nn.Module):
    def __init__(self, num_classes):
        super(MainModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=1, padding=0
        )

        self.m1 = module_thing()
        self.m2 = module_thing()
        self.m3 = module_thing()

        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.fc1= nn.Linear(in_features=64, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)

        x = self.m1(x)
        x = self.m2(x)
        x = self.m3(x)

        x = self.gap(x)
        x = torch.flatten(x,1)

        x = self.fc1(x)
        x = self.dropout(x)
        output_class = self.fc2(x)

        return output_class
