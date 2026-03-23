import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class Conv3d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3d, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(True),
        )

        self.residual = nn.Conv3d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.residual(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=kernel_size, stride=stride),
            Conv3d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, x1_in, x2_in, out_channel, kernel_size, stride, padding):
        super(Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(x1_in, x1_in, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.ReLU()
        )

        self.conv = Conv3d(x1_in + x2_in, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffC = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffC // 2, diffC - diffC // 2, diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ECAAttention(nn.Module):
    def __init__(self, channel, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, t, h, w]
        b, c, _, _, _ = x.size()
        # feature descriptor on the global spatial information
        y = self.avg_pool(x).view(b, c, 1)
        # Two different branches of ECA module
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.view(b, c, 1, 1, 1)


class OutConv(nn.Module):
    def __init__(self, out_channel, kernel_size, stride, padding):
        super(OutConv, self).__init__()

        self.convtranspose_x6 = nn.Sequential(
            nn.ConvTranspose3d(64, 16, kernel_size=[1, 5, 5],
                               stride=[1, 2, 2], padding=[0, 2, 2], bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.convtranspose_x6_2 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=[1, 5, 5],  # 将32x32扩展到64x64
                               stride=[1, 2, 2], padding=[0, 1, 1], bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.convtranspose_x6_3 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=[1, 5, 5],  # 将32x32扩展到64x64
                               stride=[1, 2, 2], padding=[0, 1, 1], output_padding=[0, 1, 1], bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3d_x6 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )


        
        self.convtranspose_x7 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=[1, 3, 3],
                               stride=[1, 2, 2], padding=[0, 1, 1], output_padding=[0, 1, 1], bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.convtranspose_x7_2 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=[1, 3, 3],
                               stride=[1, 2, 2], padding=[0, 1, 1], output_padding=[0, 1, 1], bias=True),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3d_x7 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        self.convtranspose_x8 = nn.Sequential(
            nn.ConvTranspose3d(16, 16, kernel_size=[1, 3, 3],
                               stride=[1, 2, 2], padding=[0, 1, 1], output_padding=[0, 1, 1], bias=True),  # 不使用 output_padding
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        self.conv3d_x8 = nn.Sequential(
            nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )

        
        self.conv = nn.Sequential(
            nn.Conv3d(16, out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(),
            nn.Conv3d(out_channel, out_channel, kernel_size=1, stride=1, padding=0)
        )


        self.eca_attention = ECAAttention(channel=12)

    def forward(self, x6, x7, x8, x9):

        x6 = self.convtranspose_x6(x6)  
        x6 = self.convtranspose_x6_2(x6)
        x6 = self.convtranspose_x6_3(x6)
        x6 = self.conv3d_x6(x6)  

        x7 = self.convtranspose_x7(x7)  
        x7 = self.convtranspose_x7_2(x7) 
        x7 = self.conv3d_x7(x7)  

        x8 = self.convtranspose_x8(x8) 
        x8 = self.conv3d_x8(x8)  

        x_last = torch.cat([x6, x7, x8, x9], dim=2)
        # print("x_last:", x_last.shape)
        # x_last = self.out(x_last)
        x_last = self.conv(x_last)
        # print("x_last:", x_last.shape)
        x_last = self.eca_attention(x_last)
        # return self.conv(x_last)
        return x_last


class Unet3D(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet3D, self).__init__()

        self.inc = Conv3d(in_channel, 16, kernel_size=3, stride=1, padding=1)
        self.down1 = Down(16, 32, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down2 = Down(32, 64, kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.down3 = Down(64, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])
        self.down4 = Down(128, 128, kernel_size=[2, 2, 2], stride=[2, 2, 2])

        self.up1 = Up(128, 128, 64, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up2 = Up(64, 64, 32, kernel_size=[2, 2, 2], stride=[2, 2, 2], padding=0)
        self.up3 = Up(32, 32, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)
        self.up4 = Up(16, 16, 16, kernel_size=[1, 2, 2], stride=[1, 2, 2], padding=0)

        # self.outc = OutConv([64, 32, 16, 16], out_channel, kernel_size=[18, 1, 1], stride=[1, 1, 1], padding=0)
        # self.outc = OutConv([64, 32, 16, 16], out_channel, kernel_size=[17, 1, 1], stride=[1, 1, 1], padding=0)
        self.outc = OutConv(out_channel, kernel_size=[17, 1, 1], stride=[1, 1, 1], padding=0)

    def forward(self, x):
        batch, _, _, _, _ = x.shape
        
        x1 = self.inc(x)  # [64, 16, 8, 64, 64]
        x2 = self.down1(x1)  # [64, 32, 8, 32, 32]
        x3 = self.down2(x2)  # [64, 64, 8, 16, 16]
        x4 = self.down3(x3)  # [64, 128, 4, 8, 8]
       
        x5 = self.down4(x4)  # [64, 128, 2, 4, 4]
        x6 = self.up1(x5, x4)  # [64, 64, 4, 8, 8]
        x7 = self.up2(x6, x3)  # [64, 32, 8, 16, 16]
        x8 = self.up3(x7, x2)  # [64, 16, 8, 32, 32]

        x9 = self.up4(x8, x1)  # [64, 16, 8, 64, 64]
        out = self.outc(x6, x7, x8, x9)  # [64, 1, 11, 64, 64]

        return out


if __name__ == '__main__':
    x = torch.randn((64, 1, 8, 64, 64)).cuda()
    model = Unet3D(1, 1).cuda()
    net = Unet3D(1, 1).cuda()
    out = net(x)
    print(out.shape)
