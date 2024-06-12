"""
Author  : Anonymous
Time    : created by 2024.1.28

"""
from torch import nn
import torch
class Channel_attention(nn.Module):
    def __init__(self,channel,ratio=16):
        super(Channel_attention,self).__init__()
        self.max_pool=nn.AdaptiveAvgPool2d(1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)

        self.fc=nn.Sequential(
            nn.Linear(channel,channel // ratio,bias=False),
            nn.ReLU(),
            nn.Linear(channel // ratio,channel,bias=False),
         )
        
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        b,c,h,w=x.size()
        max_pool_out=self.max_pool(x).view(b,c)
        avg_pool_out=self.avg_pool(x).view(b,c)

        max_fc_out=self.fc(max_pool_out)
        avg_fc_out=self.fc(avg_pool_out)

        out=max_fc_out+avg_fc_out
        out=self.sigmoid(out).view(b,c,1,1)
        return out*x


class Spacial_attention(nn.Module):
    def __init__(self,kernel_size=7):
        super(Spacial_attention,self).__init__()
        padding=kernel_size//2
        self.conv=nn.Conv2d(2,1,kernel_size,1,padding,bias=False)
        self.sigmoid=nn.Sigmoid()

    def forward(self,x):
        b,c,h,w=x.size()
        max_pool_out,_=torch.max(x,dim=1,keepdim=True)
        mean_pool_out,_=torch.max(x,dim=1,keepdim=True)
        pool_out=torch.cat([max_pool_out,mean_pool_out],dim=1)
        out=self.conv(pool_out)
        out=self.sigmoid(out)
        return out*x

class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=5 // 2)
        self.conv5 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)
        self.channel_attention = Channel_attention(num_channels, ratio=16)
        self.spacial_attention = Spacial_attention(kernel_size=7)

    def forward(self, x):
        residual = x  

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)


        #x = x + residual

        #x = self.channel_attention(x)
        #x = self.spacial_attention(x)
        return x
