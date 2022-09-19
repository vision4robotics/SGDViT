import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.models.neck.Maskembed import MaskEmbed
from pysot.models.neck.Fmodule import Fmodule
from pysot.models.neck.Gdecoder import Gdecoder


class SGDViT(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # self.layer1 = Encoder().cuda()
        self.mask = MaskEmbed().cuda()
        self.layer2=Fmodule().cuda()
        self.layer3=Gdecoder().cuda()
        channel=256
        self.convloc = nn.Sequential(
                nn.Conv2d(channel,channel, kernel_size=4,stride=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),                
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 4,  kernel_size=3, stride=1,padding=1),
                )
        
        self.convcls = nn.Sequential(
                nn.Conv2d(channel,channel, kernel_size=4,stride=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                )
        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)


    def template_forward(self,template):
        template=self.layer1(template)
        return template

    def search_forward(self,search):
        search=self.layer1(search)
        return search

    def forward(self,x,z):
        # x=self.search_forward(x)
        # z=self.template_forward(z)
        s,x,mask,smp=self.layer2(x,z)
        x=self.mask(x,mask)
        # print(x.shape)
        # print(z.shape)
        # print(s.shape)
        result=self.layer3(x,z,s)
        # print(result.shape)
        loc=self.convloc(result)
        acls=self.convcls(result)

        cls1=self.cls1(acls)
        cls2=self.cls2(acls)

        return loc,cls1,cls2

