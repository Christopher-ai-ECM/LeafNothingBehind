import torch
from torch import nn


class Down(nn.Module):
    # Contracting Layer

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels), #2D ou 3D ici ?
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    # Expanding Layer

    def __init__(self, in_channels, out_channels, dropout_probability):
        super().__init__()

        self.up = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


# class UNet(nn.Module):

#     def __init__(self, input_channels, output_classes, hidden_channels, dropout_probability):
#         super(UNet, self).__init__()

#         # Initial Convolution Layer
#         self.inc = nn.Sequential(
#             nn.Conv2d(input_channels, hidden_channels, kernel_size=(3, 3), padding='same'),
#             nn.Dropout(dropout_probability),
#             nn.BatchNorm2d(hidden_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(3, 3), padding='same'),
#             nn.Dropout(dropout_probability),
#             nn.BatchNorm2d(hidden_channels),
#             nn.ReLU(inplace=True))

#         # Contracting Path
#         self.down1 = Down(hidden_channels, 2 * hidden_channels, dropout_probability)
#         self.down2 = Down(2 * hidden_channels, 4 * hidden_channels, dropout_probability)
#         self.down3 = Down(4 * hidden_channels, 8 * hidden_channels, dropout_probability)
#         self.down4 = Down(8 * hidden_channels, 8 * hidden_channels, dropout_probability)

#         # Expanding Path
#         self.up1 = Up(16 * hidden_channels, 4 * hidden_channels, dropout_probability)
#         self.up2 = Up(8 * hidden_channels, 2 * hidden_channels, dropout_probability)
#         self.up3 = Up(4 * hidden_channels, hidden_channels, dropout_probability)
#         self.up4 = Up(2 * hidden_channels, hidden_channels, dropout_probability)

#         # Output Convolution Layer
#         self.outc = nn.Conv2d(hidden_channels, output_classes, kernel_size=1) #3D ou 2D ici ??
#         self.softmax = nn.Tanh()#sigmoid pas softmax ici
#         #self.softmax = nn.ReLU()

#     def forward(self, x):
#         # Initial Convolution Layer
#         x1 = self.inc(x)

#         # Contracting Path
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)

#         # Expanding Path
#         x6 = self.up1(x5, x4)
#         x7 = self.up2(x6, x3)
#         x8 = self.up3(x7, x2)
#         x9 = self.up4(x8, x1)

#         # Output Convolution Layer
#         logits = self.outc(x9)
#         output = self.softmax(logits)
#         return output


class UNet(nn.Module):

    def __init__(self, input_channels, output_classes, hidden_channels, dropout_probability, kernel_size=(3, 3)):
        super(UNet, self).__init__()

        # Initial Convolution Layer
        self.inc = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=kernel_size, padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding='same'),
            nn.Dropout(dropout_probability),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True))

        # Contracting Path
        self.down1 = Down(hidden_channels, 2 * hidden_channels, dropout_probability)
        self.down2 = Down(2 * hidden_channels, 4 * hidden_channels, dropout_probability)
        self.down3 = Down(4 * hidden_channels, 4 * hidden_channels, dropout_probability) #modif hidden_channels ici !!!
        #self.down4 = Down(8 * hidden_channels, 8 * hidden_channels, dropout_probability)

        # Expanding Path
        #self.up1 = Up(16 * hidden_channels, 4 * hidden_channels, dropout_probability)
        self.up2 = Up(8 * hidden_channels, 2 * hidden_channels, dropout_probability)
        self.up3 = Up(4 * hidden_channels, hidden_channels, dropout_probability)
        self.up4 = Up(2 * hidden_channels, hidden_channels, dropout_probability)

        # Output Convolution Layer
        self.outc = nn.Conv2d(hidden_channels, output_classes, kernel_size=1) #3D ou 2D ici ??
        self.softmax = nn.Sigmoid() #sigmoid pas softmax ici
        #self.softmax = nn.ReLU()

    def forward(self, x):
        # Initial Convolution Layer
        x1 = self.inc(x)

        # Contracting Path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
      
        x7 = self.up2(x4, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)

        # Output Convolution Layer
        logits = self.outc(x9)
        output = self.softmax(logits)
        return output