from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.submodule import *
from models.utils import unet

class HSMNet(nn.Module):
    def __init__(self, maxdisp, clean=-1.0):
        super(HSMNet, self).__init__()
        
        # self.maxdisp = maxdisp
        self.feature_extraction = unet()

        # block 4
        self.level = 1
        self.decoder6 = decoderBlock(6,32,32,up=True, pool=True)
        self.decoder5 = decoderBlock(6,32,32,up=True, pool=True)
        self.decoder4 = decoderBlock(6,32,32, up=True)
        self.decoder3 = decoderBlock(5,32,32, stride=(2,1,1),up=False, nstride=1)

        # self.decoder6 = decoderBlock(6,32,32,up=True, pool=True)
        # if self.level > 2:
        #     self.decoder5 = decoderBlock(6,32,32,up=False, pool=True)
        # else:
        #     self.decoder5 = decoderBlock(6,32,32,up=True, pool=True)
        #     if self.level > 1:
        #         self.decoder4 = decoderBlock(6,32,32, up=False)
        #     else:
        #         self.decoder4 = decoderBlock(6,32,32, up=True)
        #         self.decoder3 = decoderBlock(5,32,32, stride=(2,1,1),up=False, nstride=1)

        # init disparity regression 
        tmpdisp = int(maxdisp//64*64)
        if (maxdisp/64*64) > tmpdisp:
            self.maxdisp = tmpdisp + 64
        else:
            self.maxdisp = tmpdisp
        if self.maxdisp == 64: self.maxdisp=128
        self.disp_reg8 = disparityregression(self.maxdisp,16)
        self.disp_reg16 = disparityregression(self.maxdisp,16)
        self.disp_reg32 = disparityregression(self.maxdisp,32)
        self.disp_reg64 = disparityregression(self.maxdisp,64)

        if clean < 0.0:
            self.clean = -1.0
        else:
            self.clean = clean

    def feature_vol(self, refimg_fea, targetimg_fea, maxdisp: int, leftview:bool=True):
        '''
        diff feature volume
        '''
        width = refimg_fea.size()[-1]
        # cost = torch.cuda.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1], maxdisp,  refimg_fea.size()[2],  refimg_fea.size()[3]).fill_(0.)
        cost = torch.zeros([int(refimg_fea.size()[0]), int(refimg_fea.size()[1]), int(maxdisp),  int(refimg_fea.size()[2]),  int(refimg_fea.size()[3])], dtype=torch.float32, device=refimg_fea.device)
        for i in range(min(maxdisp, width)):
            feata = refimg_fea[:,:,:,i:width]
            featb = targetimg_fea[:,:,:,:width-i]
            # concat
            if leftview:
                cost[:, :refimg_fea.size()[1], i, :,i:]   = torch.abs(feata-featb)
            else:
                cost[:, :refimg_fea.size()[1], i, :,:width-i]   = torch.abs(featb-feata)
        cost = cost.contiguous()
        return cost
    
    @torch.jit.export
    def set_max_disp(self, maxdisp: int):
        updated = False
        if maxdisp > 0:
            new_maxdisp = int(maxdisp//64*64)
            if (maxdisp/64*64) > new_maxdisp:
                new_maxdisp = new_maxdisp + 64
            if new_maxdisp == 64: new_maxdisp=128

            if self.maxdisp != new_maxdisp:
                self.maxdisp = new_maxdisp
                self.disp_reg8.set_max_disp(self.maxdisp, 16)
                self.disp_reg16.set_max_disp(self.maxdisp, 16)
                self.disp_reg32.set_max_disp(self.maxdisp, 32)
                self.disp_reg64.set_max_disp(self.maxdisp, 64)
                updated = True
        return self.maxdisp, updated

    @torch.jit.export
    def set_clean(self, clean: float):
        if clean < 0.0:
            self.clean = -1.0
        else:
            self.clean = clean
        return self.clean

    @torch.jit.export
    def set_level(self, level: int):
        updated = False
        if level <= 3 and level >= 1:
            # training only possible of all levels, also scripting requires all levels to be available
            if self.training:
                self.level = 1

            if level != self.level:
                if level > 2:
                   self.decoder5.set_up(False)
                   updated = True
                else:
                    up5 = self.decoder5.up_used
                    if self.decoder5.set_up(True):
                        if level > 1:
                            self.decoder4.set_up(False)
                            updated = True
                        else:
                            if self.decoder4.set_up(True):
                                updated = True
                            else:
                                self.decoder5.set_up(up5)
                if updated:
                    self.level = level
        return self.level, updated

    def forward(self, left, right):
        nsample = left.size()[0]
        conv4,conv3,conv2,conv1  = self.feature_extraction(torch.cat([left,right],0))
        conv40,conv30,conv20,conv10  = conv4[:nsample], conv3[:nsample], conv2[:nsample], conv1[:nsample]
        conv41,conv31,conv21,conv11  = conv4[nsample:], conv3[nsample:], conv2[nsample:], conv1[nsample:]

        feat6 = self.feature_vol(conv40, conv41, self.maxdisp//64)
        feat5 = self.feature_vol(conv30, conv31, self.maxdisp//32)
        feat4 = self.feature_vol(conv20, conv21, self.maxdisp//16)
        feat3 = self.feature_vol(conv10, conv11, self.maxdisp//8)

        feat6_2x, cost6 = self.decoder6(feat6)
        feat5 = torch.cat((feat6_2x, feat5),dim=1)

        feat5_2x, cost5 = self.decoder5(feat5)
        if self.level > 2:
            cost3 = F.interpolate(cost5, [left.size()[2],left.size()[3]], mode='bilinear')
            cost4 = None
        else:
            feat4 = torch.cat((feat5_2x, feat4),dim=1)

            feat4_2x, cost4 = self.decoder4(feat4) # 32
            if self.level > 1:
                cost3 = F.interpolate((cost4).unsqueeze(1), [self.disp_reg8.disp.size()[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
            else:
                feat3 = torch.cat((feat4_2x, feat3),dim=1)

                feat3_2x, cost3 = self.decoder3(feat3) # 32
                cost3 = F.interpolate(cost3, [left.size()[2],left.size()[3]], mode='bilinear')

        # for script and tracing, only supports inference (to train with script, two different foward functions would be
        # necessary)
        if torch.jit.is_scripting() or torch.jit.is_tracing():
            if self.level > 2:
                final_reg = self.disp_reg32
                if self.clean<=-1.0:
                    pred3, _ = final_reg(F.softmax(cost3,1)); entropy = pred3  # to save memory
                else:
                    pred3, entropy = final_reg(F.softmax(cost3,1),ifent=True)
                    if entropy is not None:
                        pred3[entropy>self.clean] = torch.inf
            else:
                final_reg = self.disp_reg8
                if self.clean<=-1.0:
                    pred3, _ = final_reg(F.softmax(cost3,1)); entropy = pred3  # to save memory
                else:
                    pred3, entropy = final_reg(F.softmax(cost3,1),ifent=True)
                    if entropy is not None:
                        pred3[entropy>self.clean] = torch.inf

            if entropy is not None:
                entropy = torch.squeeze(entropy)
            return pred3, entropy
        # for pytorch model
        else:
            if self.level > 2:
                final_reg = self.disp_reg32
            else:
                final_reg = self.disp_reg8
    
            if self.training or self.clean<=-1.0:
                pred3, _ = final_reg(F.softmax(cost3,1)); entropy = pred3  # to save memory
            else:
                pred3, entropy = final_reg(F.softmax(cost3,1),ifent=True)
                pred3[entropy>self.clean] = torch.inf

            if self.training:
                cost6 = F.interpolate((cost6).unsqueeze(1), [self.disp_reg8.disp.size()[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
                cost5 = F.interpolate((cost5).unsqueeze(1), [self.disp_reg8.disp.size()[1], left.size()[2],left.size()[3]], mode='trilinear').squeeze(1)
                cost4 = F.interpolate(cost4, [left.size()[2],left.size()[3]], mode='bilinear')
                pred6 = self.disp_reg16(F.softmax(cost6,1))
                pred5 = self.disp_reg16(F.softmax(cost5,1))
                pred4 = self.disp_reg16(F.softmax(cost4,1))
                stacked = [pred3,pred4,pred5,pred6]   
                return stacked,entropy
            else:
                return pred3,torch.squeeze(entropy)
