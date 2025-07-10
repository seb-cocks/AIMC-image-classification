import torch
import torch.nn as nn

class Inception1(nn.Module):
    def __init__(self):
        super(Inception1, self).__init__()

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        self.conv1 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=2, stride=2)

    def forward(self, x):

        x1 = self.pool1(x)

        x2 = self.conv1(x)
        x2 = self.conv2(x2)

        x3 = self.conv3(x)
        x3 = self.conv4(x3)
        x3 = self.conv5(x3)

        x = torch.cat((x1, x2, x3), dim=1)

        return x

class Inception2a(nn.Module):
    def __init__(self):
        super(Inception2a, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(
            in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # Added padding

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=1, stride=1)
        self.conv6 = nn.Conv2d(
            in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # Added padding
        self.conv7 = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # Added padding

    def forward(self, x):

        x1 = self.conv1(x)

        x2 = self.pool1(x)
        x2 = self.conv2(x2)
        x2 = (
            nn.functional.pad(x2, (0, -1, 0, -1)) if x2.shape[-1] > 32 else x2
        )  # Crop if needed

        x3 = self.conv3(x)
        x3 = self.conv4(x3)

        x4 = self.conv5(x)
        x4 = self.conv6(x4)
        x4 = self.conv7(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        return x

class Inception2b(nn.Module):
    def __init__(self):
        super(Inception2b, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=96, out_channels=24, kernel_size=1, stride=1)

        # Added padding=1 to prevent size reduction
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=24, kernel_size=1, stride=1)

        self.conv3 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1)
        self.conv4 = nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )  # Padding added
        self.conv5 = nn.Conv2d(
            in_channels=48,
            out_channels=48,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )  # Padding added

        self.conv6 = nn.Conv2d(in_channels=96, out_channels=32, kernel_size=1, stride=1)
        self.conv7 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )  # Padding added
        self.conv8 = nn.Conv2d(
            in_channels=32,
            out_channels=32,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )  # Padding added
        self.conv9 = nn.Conv2d(
            in_channels=32,
            out_channels=48,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )  # Padding added
        self.conv10 = nn.Conv2d(
            in_channels=48,
            out_channels=48,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )  # Padding added

    def forward(self, x):

        x1 = self.conv1(x)

        x2 = self.pool1(x)
        x2 = self.conv2(x2)
        x2 = (
            nn.functional.pad(x2, (0, -1, 0, -1)) if x2.shape[-1] > 16 else x2
        )  # Crop if needed

        x3 = self.conv3(x)
        x3 = self.conv4(x3)
        x3 = self.conv5(x3)

        x4 = self.conv6(x)
        x4 = self.conv7(x4)
        x4 = self.conv8(x4)
        x4 = self.conv9(x4)
        x4 = self.conv10(x4)

        x = torch.cat([x1, x2, x3, x4], 1)
        return x

class Inception3a(nn.Module):
    def __init__(self):
        super(Inception3a, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=144, out_channels=32, kernel_size=1, stride=1
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=144, out_channels=32, kernel_size=1, stride=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=144, out_channels=48, kernel_size=1, stride=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=48,
            out_channels=48,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )  # Added padding
        self.conv5 = nn.Conv2d(
            in_channels=48,
            out_channels=48,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )  # Added padding

        self.conv6 = nn.Conv2d(
            in_channels=144, out_channels=32, kernel_size=1, stride=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1
        )  # Added padding
        self.conv8 = nn.Conv2d(
            in_channels=48,
            out_channels=16,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )  # Added padding
        self.conv9 = nn.Conv2d(
            in_channels=48,
            out_channels=16,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )  # Added padding

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.pool1(x)
        x2 = self.conv2(x2)
        x2 = (
            nn.functional.pad(x2, (0, -1, 0, -1)) if x2.shape[-1] > 8 else x2
        )  # Crop if needed

        x3_pre_split = self.conv3(x)
        x3 = self.conv4(x3_pre_split)
        x4 = self.conv5(x3_pre_split)

        x5_pre_split = self.conv6(x)
        x5_pre_split = self.conv7(x5_pre_split)
        x5 = self.conv8(x5_pre_split)
        x6 = self.conv9(x5_pre_split)

        x = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        return x

class Inception3b(nn.Module):
    def __init__(self):
        super(Inception3b, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=192, out_channels=32, kernel_size=1, stride=1
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=192, out_channels=32, kernel_size=1, stride=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=192, out_channels=48, kernel_size=1, stride=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )  # Added padding
        self.conv5 = nn.Conv2d(
            in_channels=48,
            out_channels=64,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )  # Added padding

        self.conv6 = nn.Conv2d(
            in_channels=192, out_channels=32, kernel_size=1, stride=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=32, out_channels=48, kernel_size=3, stride=1, padding=1
        )  # Added padding
        self.conv8 = nn.Conv2d(
            in_channels=48,
            out_channels=32,
            kernel_size=(1, 3),
            stride=1,
            padding=(0, 1),
        )  # Added padding
        self.conv9 = nn.Conv2d(
            in_channels=48,
            out_channels=32,
            kernel_size=(3, 1),
            stride=1,
            padding=(1, 0),
        )  # Added padding

    def forward(self, x):
        x1 = self.conv1(x)

        x2 = self.pool1(x)
        x2 = self.conv2(x2)
        x2 = (
            nn.functional.pad(x2, (0, -1, 0, -1)) if x2.shape[-1] > 4 else x2
        )  # Crop if needed

        x3_pre_split = self.conv3(x)
        x3 = self.conv4(x3_pre_split)
        x4 = self.conv5(x3_pre_split)

        x5_pre_split = self.conv6(x)
        x5_pre_split = self.conv7(x5_pre_split)
        x5 = self.conv8(x5_pre_split)
        x6 = self.conv9(x5_pre_split)

        x = torch.cat([x1, x2, x3, x4, x5, x6], 1)
        return x

class deconv_block(nn.Module):
    def __init__(self):
        super(deconv_block, self).__init__()

        # Deconv 1: Input (4x4x192) -> Output (8x8x128)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=192,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        # Deconv 2: Input (8x8x128) -> Output (16x16x96)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=96,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        # Deconv 3: Input (16x16x96) -> Output (32x32x48)
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=96,
            out_channels=48,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

        # Deconv 4: Input (32x32x48) -> Output (64x64x1)
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=48,
            out_channels=1,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )

    def forward(self, x):

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)

        return x


class MainModel(nn.Module):
    def __init__(self, num_classes):
        super(MainModel, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )

        self.inc1 = Inception1()

        self.inc2a = Inception2a()

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inc2b = Inception2b()

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inc3a = Inception3a()

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deconv = deconv_block()

        self.inc3b = Inception3b()

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(in_features=1024, out_features=num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.conv1(x)

        x = self.inc1(x)

        x = self.inc2a(x)

        x = self.pool1(x)

        x = self.inc2b(x)

        x = self.pool2(x)

        x = self.inc3a(x)

        x = self.pool3(x)

        output_image = self.deconv(x)

        x = self.inc3b(x)

        x = self.pool4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)
        output_class = x

        return output_image, output_class
