from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import math
import numpy as np
import pdb


class sepConv3dBlock(nn.Module):
    '''
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1)):
        super(sepConv3dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes,stride)
        self.conv1 = sepConv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = sepConv3d(out_planes, out_planes, 3, (1,1,1), 1)
            

    def forward(self,x):
        out = F.relu(self.conv1(x),inplace=True)
        if self.downsample is not None:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out),inplace=True)
        return out




class projfeat3d(nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1,1), padding=(0,0), stride=stride[:2],bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self,x):
        b,c,d,h,w = x.size()
        x = self.conv1(x.view(b,c,d,h*w))
        x = self.bn(x)
        x = x.view(b,-1,torch.div(torch.tensor(d), self.stride[0], rounding_mode='floor'),h,w)
        return x

# original conv3d block
def sepConv3d(in_planes, out_planes, kernel_size, stride, pad,bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                         nn.BatchNorm3d(out_planes))
        


    

class disparityregression(nn.Module):
    def __init__(self, maxdisp,divisor):
        super(disparityregression, self).__init__()
        maxdisp = int(maxdisp/divisor)
        #self.disp = torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda()
        self.register_buffer('disp',torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])))
        self.divisor = divisor

    def forward(self, x,ifent: bool=False):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1) * self.divisor

        if ifent:
            # entropy
            x = x+1e-12
            ent = (-x*x.log()).sum(dim=1)
            return out,ent
        else:
            return out,None


class decoderBlock(nn.Module):
    def __init__(self, nconvs, inchannelF,channelF,stride=(1,1,1),up=False, nstride=1,pool=False):
        super(decoderBlock, self).__init__()
        self.pool=pool
        stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)
        self.convs = [sepConv3dBlock(inchannelF,channelF,stride=stride[0])]
        for i in range(1,nconvs):
            self.convs.append(sepConv3dBlock(channelF,channelF, stride=stride[i]))
        self.convs = nn.Sequential(*self.convs)

        self.classify = nn.Sequential(sepConv3d(channelF, channelF, 3, (1,1,1), 1),
                                       nn.ReLU(inplace=True),
                                       sepConv3d(channelF, 1, 3, (1,1,1),1,bias=True))

        self.up = None
        if up:
            self.up = nn.Sequential(nn.Upsample(scale_factor=(2,2,2),mode='trilinear'),
                                 sepConv3d(channelF, channelF//2, 3, (1,1,1),1,bias=False),
                                 nn.ReLU(inplace=True))

        if pool:
            self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0)])
        else:
            self.pool_convs = None
 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #torch.nn.init.xavier_uniform(m.weight)
                #torch.nn.init.constant(m.weight,0.001)
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()
            #elif isinstance(m, nn.BatchNorm3d):
            #    m.weight.data.fill_(1)
            #    m.bias.data.zero_()
            #    m.running_mean.data.fill_(0)
            #    m.running_var.data.fill_(1)


    def forward(self,fvl):
        # left
        fvl = self.convs(fvl)
        # pooling
        if self.pool and self.pool_convs is not None:
            fvl_out = fvl
            _,_,d,h,w=fvl.shape
            # # for i,pool_size in enumerate(np.linspace(1,torch.div(min(d,h,w), 2, rounding_mode='floor'),4,dtype=int)):
            # for i,pool_size in enumerate(torch.linspace(1,torch.div(torch.tensor(min(d,h,w)), 2, rounding_mode='floor').item(),4,dtype=torch.int)):
            #     kernel_size = [(int)(torch.true_divide(torch.tensor(d),pool_size).type(torch.int).item()), (int)(torch.true_divide(torch.tensor(h),pool_size).type(torch.int).item()), (int)(torch.true_divide(torch.tensor(w),pool_size).type(torch.int).item())]
            #     out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)       
            #     out = self.pool_convs[i](out)
            #     out = F.upsample(out, size=(d,h,w), mode='trilinear')
            #     fvl_out = fvl_out + 0.25*out
            pool_size = torch.linspace(1, torch.div(torch.tensor(min(d,h,w)), 2, rounding_mode='floor').item(), 4, dtype=torch.int)
            for i, layer in enumerate(self.pool_convs):
                kernel_size = [(int)(torch.true_divide(torch.tensor(d),pool_size[i]).type(torch.int).item()), (int)(torch.true_divide(torch.tensor(h),pool_size[i]).type(torch.int).item()), (int)(torch.true_divide(torch.tensor(w),pool_size[i]).type(torch.int).item())]
                out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)       
                out = layer(out)
                # out = F.upsample(out, size=(d,h,w), mode='trilinear')
                out = F.interpolate(out, size=(d,h,w), mode='trilinear')
                fvl_out = fvl_out + 0.25*out
            fvl = F.relu(fvl_out/2.,inplace=True)

       # #TODO cost aggregation
       # costl = self.classify(fvl)
       # if self.up:
       #     fvl = self.up(fvl)
        if self.training:
            # classification
            costl = self.classify(fvl)
            if self.up is not None:
                fvl = self.up(fvl)
        else:
            # classification
            if self.up is not None:
                fvl = self.up(fvl)
                costl=fvl
            else:
                costl = self.classify(fvl)

        return fvl,costl.squeeze(1)
