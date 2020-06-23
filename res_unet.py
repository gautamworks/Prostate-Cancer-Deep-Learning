#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# In[2]:


res34 = models.resnet34(pretrained=True)


# In[10]:


# print(res34)


# In[ ]:





# In[31]:


class Decoderblock(nn.Module):
    
    def __init__(self,in_channels,out_channels,up_kernel_size=3,up_stride=2,up_padding=1):
        super(Decoderblock,self).__init__()
        self.upsampler = nn.ConvTranspose2d(in_channels,out_channels,kernel_size=up_kernel_size,stride=up_stride,padding=up_padding,output_padding=1)
        self.activation = nn.ReLU()
        self.first_block = nn.Sequential(
            nn.Conv2d(2*out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)   
        )
        self.second_block = nn.Sequential(
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,x,skip):
        x = self.activation(self.upsampler(x))
#         print(f'Up : {x.size()}')
        x = torch.cat([x,skip],dim=1)
#         print(f'Skip : {x.size()}')
        x = self.first_block(x)
#         print(f'1 : {x.size()}')
        x = self.second_block(x)
#         print(f'2 : {x.size()}')
        return x


# In[ ]:





# In[53]:


class ResUnet(nn.Module):
    
    def __init__(self,num_classes=5):
        super(ResUnet,self).__init__()
        self.num_classes = num_classes
        self.resencoder = models.resnet34(pretrained=True)
        self.first_filters = nn.Sequential(
            self.resencoder.conv1,self.resencoder.bn1,
            self.resencoder.relu
        )
        self.first_pool = self.resencoder.maxpool
        self.encoder_layers = nn.ModuleList([
            self.resencoder.layer1,self.resencoder.layer2,
            self.resencoder.layer3,self.resencoder.layer4])
        self.middle_layer = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.decoder_layers = nn.ModuleList([
            Decoderblock(512,256),
            Decoderblock(256,128),
            Decoderblock(128,64),
            Decoderblock(64,64)
        ])
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(64,64,kernel_size=3,stride=2,padding=1,output_padding=1),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,num_classes,kernel_size=1)
        )

    def require_encoder_grad(self, requires_grad):
        blocks = [
            self.first_filters,
            *self.encoder_layers
        ]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self,x):
        self.ints= []
        x = self.first_filters(x)
        self.ints.append(x)
        x = self.first_pool(x)
        # counter = 1
        for layer in self.encoder_layers:
            x = layer(x)
            self.ints.append(x)
            # counter+=1
        # counter-=2
        x = self.middle_layer(x)
        for inter,decoder_layer in zip(reversed(self.ints[:-1]),self.decoder_layers):
            x = decoder_layer(x,inter)
            # counter-=1
        del self.ints
        x = self.final_layer(x)
        # print(f' final layer {x.size()}')
        return x


# In[54]:


model = ResUnet()
# model.require_encoder_grad(requires_grad=False)
# for p in model.encoder_layers[0].parameters():
#     print(p.requires_grad)
#     break


# In[46]:


# dummy_ten = torch.rand((1,3,224,224),requires_grad=False)
# dummy_fil = nn.ConvTranspose2d(3,3,kernel_size=3,stride=2,padding=1,output_padding=1)
# print(dummy_fil(dummy_ten).size())


# In[ ]:




