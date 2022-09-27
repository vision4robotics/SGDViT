import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.models.utils.windows import Mlp
class Fmodule(nn.Module):
    """
    input: template: 1, 256, 6, 6
           search:   1, 256, 26, 26

    output: s:       1, 256, 17, 17
            x:       1, 256, 17, 17
            mask:    1, 1, 26, 26
    """
    def __init__(self,embed_dim=256,mlp_ratio=4.,dim=384):
        super().__init__()
        self.layer1= nn.Sequential(
            nn.Conv2d(embed_dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            )
        self.layer2= nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            )
        self.layer3= nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer7= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer8= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer4= nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer5= nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=1,stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            )
        self.layer6= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1,stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.layer9= nn.Sequential(
            nn.Conv2d(dim, 256, kernel_size=4),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer10= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer11= nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer12= nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            )
        self.layer= nn.Conv2d(dim,dim,kernel_size=2)

        self.mlp= Mlp(in_features=dim, out_features=dim,hidden_features=int(dim * mlp_ratio))
    def xcorr_depthwise(self,x, kernel):
        """
        depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def cforward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        return x

    def CDConv(self,x):
        x=self.layer3(x)
        x=self.layer7(x)
        x=self.layer8(x)
        x=self.layer4(x)
        return x
    
    def FASNet(self,x):
        x=self.layer9(x)
        x=self.layer10(x)
        x=self.layer11(x)
        x=self.layer12(x)
        return x

    def forward(self, x,z):
        x=self.cforward(x)
        z=self.cforward(z)
        saliency = self.xcorr_depthwise(x, z).permute(2,3,0,1)
      
        s=self.mlp(saliency)
        s=s.permute(2,3,0,1)
        s=self.CDConv(s)

        #the output of saliency features and saliency map
        mask=self.layer5(s)
        s=self.layer6(s)

        #process the X
        x=self.layer(x)
        x=self.FASNet(x)

        return s,x,mask,saliency

    


