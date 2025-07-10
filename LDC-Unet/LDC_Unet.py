import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()

        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,padding=1)

        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2,padding=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1)

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.relu(x1)

        x2 = self.conv2(x)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)

        x = x1+x2

        return x

class decoder(nn.Module):
    def __init__(self, c1, c2, c3, c4):
        super(decoder, self).__init__()

        self.relu = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=64, kernel_size=3 , stride=1,padding=1)

        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=64, kernel_size=3 , stride=1,padding=1)

        self.upsample1 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(in_channels=c3, out_channels=64, kernel_size=3 , stride=1,padding=1)

        self.upsample2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.conv4 = nn.Conv2d(in_channels=c4, out_channels=64, kernel_size=3 , stride=1,padding=1)



    def forward(self, x1, x2, x3, x4):

        x1 = self.pool1(x1)
        x1 = self.conv1(x1)

        x2 = self.conv2(x2)

        x3 = self.upsample1(x3)
        x3 = self.conv3(x3)
        
        x4 = self.upsample2(x4)
        x4 = self.conv4(x4)

        x = torch.cat((x1,x2,x3,x4),dim=1)

        return x

class decoder_4(nn.Module):
    def __init__(self, c1, c2, c3):
        super(decoder_4, self).__init__()

        self.relu = nn.ReLU()

        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv1 = nn.Conv2d(in_channels=c1, out_channels=64, kernel_size=3 , stride=1,padding=1)

        self.conv2 = nn.Conv2d(in_channels=c2, out_channels=64, kernel_size=3 , stride=1,padding=1)

        self.upsample1 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.conv3 = nn.Conv2d(in_channels=c3, out_channels=64, kernel_size=3 , stride=1,padding=1)


    def forward(self, x1, x2, x3):

        x1 = self.pool1(x1)
        x1 = self.conv1(x1)

        x2 = self.conv2(x2)

        x3 = self.upsample1(x3)
        x3 = self.conv3(x3)
    
        x = torch.cat((x1,x2,x3),dim=1)

        return x

class LDC_Unet(nn.Module):
    def __init__(self):
        super(LDC_Unet, self).__init__()

        self.e1 = encoder(in_channels=3, out_channels=64)
        self.e2 = encoder(in_channels=64, out_channels=128)
        self.e3 = encoder(in_channels=128, out_channels=256)
        self.e4 = encoder(in_channels=256, out_channels=256)
        self.e5 = encoder(in_channels=256, out_channels=512)

        self.d4 = decoder_4(c1=256 , c2=256, c3=512)
        self.d3 = decoder(c1=128 , c2=256, c3=256, c4=192)
        self.d2 = decoder(c1=64 , c2=128, c3=256, c4=256)
        self.d1 = decoder(c1=3 , c2=64, c3=128, c4=256)

        self.upsample = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1)  # Change padding to 0
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1)  # Increase padding to 2

        

    def forward(self, x):

        x0 = x

        x_e1 = self.e1(x0)

        x_e2 = self.e2(x_e1)
        x_e3 = self.e3(x_e2)
        x_e4 = self.e4(x_e3)
        x_e5 = self.e5(x_e4)

        x_d4 = self.d4(x_e3, x_e4, x_e5)
        x_d3 = self.d3(x_e2, x_e3, x_e4, x_d4)
        x_d2 = self.d2(x_e1, x_e2, x_e3, x_d3)
        x_d1 = self.d1(x0, x_e1, x_e2, x_d2)
        
        x = self.upsample(x_d1)
        x = self.conv1(x)
        x = self.conv2(x)

        return x

class DCNN(nn.Module):
    def __init__(self, num_classes):
        super(DCNN, self).__init__()
        
        # Load pre-trained VGG19
        self.vgg19 = models.vgg19(pretrained=True)

        # Remove fully connected layers
        self.features = self.vgg19.features  # Keep convolutional layers

        # Custom classification block
        self.classification_block = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # Ensure compatibility
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 256)  # Features before final classification
        )

        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dense_last = nn.Linear(256, num_classes)  # Final classification layer

    def forward(self, x):
        x = self.features(x)  # Extract CNN features
        features = self.classification_block(x)  # Intermediate feature representation
        features = self.relu(features)
        features = self.dropout1(features)

        logits = self.dense_last(features)  # Final classification output

        return features, logits  # Return both features and logits


class MainModel(nn.Module):
    def __init__(self, num_classes):
        super(MainModel, self).__init__()

        self.ldc_unet = LDC_Unet()

        self.dcnn = DCNN(num_classes)
        

    def forward(self, x):

        x = self.ldc_unet(x)

        f, l = self.dcnn(x)

        return f, l