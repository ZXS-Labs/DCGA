import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet_cbam import BasicBlock
from .pvt import PVT
from .galerkin import *

from thop import profile
import os

from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold,datasets
from sklearn.manifold import TSNE


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


def convt_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, output_padding=0,
                  bn=True, relu=True):
    # assert (kernel % 2) == 1, \
    #     'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.ConvTranspose2d(ch_in, ch_out, kernel, stride, padding,
                                     output_padding, bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


# class Backbone(nn.Module):
#     def __init__(self, args, mode='rgbd'):
#         super(Backbone, self).__init__()
#         self.args = args
#         self.b = args.batch_size
#         self.mode = mode
#         self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

#         # Encoder
#         if mode == 'rgbd':
#             self.conv1_rgb = conv_bn_relu(3, 48, kernel=3, stride=1, padding=1,
#                                           bn=False)
#             self.conv1_dep = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
#                                           bn=False)
#             self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
#                                       bn=False)
#         elif mode == 'rgb':
#             self.conv1 = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1,
#                                       bn=False)
#         elif mode == 'd':
#             self.conv1 = conv_bn_relu(1, 64, kernel=3, stride=1, padding=1,
#                                       bn=False)
#         else:
#             raise TypeError(mode)

#         self.former = PVT(in_chans=64, patch_size=2, pretrained='./pretrained/pvt.pth',)

#         # channels = [64, 128, 64, 128, 320, 512]
#         channels = [64, 128, 256, 512,1024]
#         # Shared Decoder
#         # 1/16
#         # self.dec6 = nn.Sequential(
#         #     convt_bn_relu(channels[5], 256, kernel=3, stride=2,
#         #                   padding=1, output_padding=1),
#         #     BasicBlock(256, 256, stride=1, downsample=None, ratio=16),
#         # )
#         # 1/8
#         # self.dec5 = nn.Sequential(
#         #     convt_bn_relu(256+channels[4], 128, kernel=3, stride=2,
#         #                   padding=1, output_padding=1),
#         #     BasicBlock(128, 128, stride=1, downsample=None, ratio=8),

#         # )
#         # 1/4
#         # 修改
#         self.dec4 = nn.Sequential(
#             convt_bn_relu(channels[3], 64, kernel=3, stride=2,
#                           padding=1, output_padding=1),
#             BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
#         )

#         # 1/2
#         self.dec3 = nn.Sequential(
#             convt_bn_relu(64 + channels[2], 64, kernel=3, stride=2,
#                           padding=1, output_padding=1),
#             BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
#         )

#         # 1/1
#         self.dec2 = nn.Sequential(
#             convt_bn_relu(64 + channels[1], 64, kernel=3, stride=2,
#                           padding=1, output_padding=1),
#             BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
#         )


#         # Init Depth Branch
#         # 1/1
#         self.dep_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
#                                      padding=1)
#         self.dep_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
#                                      padding=1, bn=False, relu=True)
#         # Guidance Branch
#         # 1/1
#         self.gd_dec1 = conv_bn_relu(64+channels[0], 64, kernel=3, stride=1,
#                                     padding=1)
#         self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
#                                     padding=1, bn=False, relu=False)

#         if self.args.conf_prop:
#             # Confidence Branch
#             # Confidence is shared for propagation and mask generation
#             # 1/1
#             self.cf_dec1 = conv_bn_relu(64+channels[0], 32, kernel=3, stride=1,
#                                         padding=1)
#             self.cf_dec0 = nn.Sequential(
#                 nn.Conv2d(32+64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
#                 nn.Sigmoid()
#             )

#         self.encoder_galerkin1 = simple_attn(64,4)
#         self.downsample1 = PatchMerging(64)
#         self.encoder_galerkin2 = simple_attn(128,4)
#         self.downsample2 = PatchMerging(128)
#         self.encoder_galerkin3 = simple_attn(256,4)
#         self.downsample3 = PatchMerging(256)
#         self.encoder_galerkin4 = simple_attn(512,4)
#         self.downsample4 = PatchMerging(512)
#         self.encoder_galerkin5 = simple_attn(512,4)
#         self.downsample5 = PatchMerging(1024)
#         self.encoder_galerkin6 = simple_attn(512,4)

#     def _concat(self, fd, fe, dim=1):
#         # Decoder feature may have additional padding
#         _, _, Hd, Wd = fd.shape
#         _, _, He, We = fe.shape

#         fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

#         f = torch.cat((fd, fe), dim=dim)

#         return f

#     def forward(self, rgb=None, depth=None):

#         B_R,C_R,H_R,W_R = rgb.shape
#         if H_R % 8 != 0:
#             t_R = torch.zeros([ B_R, 3, 8-H_R%8, W_R]).cuda()
#             t_D = torch.zeros([ B_R, 1, 8-H_R%8, W_R]).cuda()
#             rgb = torch.cat((rgb,t_R),2)
#             depth = torch.cat((depth,t_D),2)
#         if W_R % 8 != 0:
#             t_R = torch.zeros([ B_R, 3, H_R, 8-W_R%8]).cuda()
#             t_D = torch.zeros([ B_R, 1, H_R, 8-W_R%8]).cuda()
#             rgb = torch.cat((rgb,t_R),3)
#             depth = torch.cat((depth,t_D),3)

#         # Encoding
#         if self.mode == 'rgbd':
#             fe1_rgb = self.conv1_rgb(rgb)
#             fe1_dep = self.conv1_dep(depth)
#             fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
#             fe1 = self.conv1(fe1)
#         elif self.mode == 'rgb':
#             fe1 = self.conv(rgb)
#         elif self.mode == 'd':
#             fe1 = self.conv(depth)
#         else:
#             raise TypeError(self.mode)
        
#         B,C,H,W = fe1.shape

#         # fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
#         fe2 = self.encoder_galerkin1(fe1) # 64
#         fe3 = self.downsample1(fe2.permute(0,2,3,1).view(B,H*W,C), [H,W]) # 128
#         fe3 = self.encoder_galerkin2(fe3.view(B,C*2,int(H/2),int(W/2))) # 128
#         fe4 = self.downsample2(fe3.permute(0,2,3,1).view(B,int(H*W/4),C*2), [int(H/2),int(W/2)]) # 256
#         fe4 = self.encoder_galerkin3(fe4.view(B,C*4,int(H/4),int(W/4))) # 256
#         fe5 = self.downsample3(fe4.permute(0,2,3,1).view(B,int(H*W/16),C*4), [int(H/4),int(W/4)]) # 512
#         fe5 = self.encoder_galerkin4(fe5.view(B,C*8,int(H/8),int(W/8))) # 512

#         # Shared Decoding
#         # fd6 = self.dec6(fe7)
#         # fd5 = self.dec5(self._concat(fd6, fe6))
#         fd4 = self.dec4(fe5)
#         fd3 = self.dec3(self._concat(fd4, fe4))
#         fd2 = self.dec2(self._concat(fd3, fe3))

#         # Init Depth Decoding
#         dep_fd1 = self.dep_dec1(self._concat(fd2, fe2))
#         init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))

#         # Guidance Decoding
#         gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
#         guide = self.gd_dec0(self._concat(gd_fd1, fe1))

#         if self.args.conf_prop:
#             # Confidence Decoding
#             cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
#             confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
#         else:
#             confidence = None

#         return init_depth, guide, confidence

class Backbone(nn.Module):
    def __init__(self, args, mode='rgbd'):
        super(Backbone, self).__init__()
        self.args = args
        self.mode = mode
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        # Encoder
        if mode == 'rgbd':
            self.conv1_rgb = conv_bn_relu(3, 32, kernel=3, stride=1, padding=1,
                                          bn=False)
            self.conv1_dep = conv_bn_relu(1, 32, kernel=3, stride=1, padding=1,
                                          bn=False)
            self.conv1 = conv_bn_relu(64, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
    
            # add
            self.mlp_rgb = MLP(3, 16, 32)
            self.attn_rgb = simple_attn(32, 4)

            self.mlp_dep = MLP(1,16,32)
            self.attn_dep = cross_attn(32, 4)
            self.attn = simple_attn(64, 4)

            self.conv1_attn = torch.nn.Conv1d(3, 64, 1)
            self.bn1 = nn.BatchNorm1d(64)

            self.geofeature = GeometryFeature()


        elif mode == 'rgb':
            self.conv1 = conv_bn_relu(3, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        elif mode == 'd':
            self.conv1 = conv_bn_relu(1, 64, kernel=3, stride=1, padding=1,
                                      bn=False)
        else:
            raise TypeError(mode)

        self.former = PVT(in_chans=64, patch_size=2, pretrained='./pretrained/pvt.pth',)

        channels = [64, 128, 64, 128, 320, 512]
        # Shared Decoder
        # 1/16
        self.dec6 = nn.Sequential(
            convt_bn_relu(channels[5], 256, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(256, 256, stride=1, downsample=None, ratio=16),
        )
        # 1/8
        self.dec5 = nn.Sequential(
            convt_bn_relu(256+channels[4], 128, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(128, 128, stride=1, downsample=None, ratio=8),

        )
        # 1/4
        self.dec4 = nn.Sequential(
            convt_bn_relu(128 + channels[3], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/2
        self.dec3 = nn.Sequential(
            convt_bn_relu(64 + channels[2], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )

        # 1/1
        self.dec2 = nn.Sequential(
            convt_bn_relu(64 + channels[1], 64, kernel=3, stride=2,
                          padding=1, output_padding=1),
            BasicBlock(64, 64, stride=1, downsample=None, ratio=4),
        )


        # Init Depth Branch
        # 1/1
        self.dep_dec1 = conv_bn_relu(64+64, 64, kernel=3, stride=1,
                                     padding=1)
        self.dep_dec0 = conv_bn_relu(64+64, 1, kernel=3, stride=1,
                                     padding=1, bn=False, relu=True)
        # Guidance Branch
        # 1/1
        self.gd_dec1 = conv_bn_relu(64+channels[0], 64, kernel=3, stride=1,
                                    padding=1)
        self.gd_dec0 = conv_bn_relu(64+64, self.num_neighbors, kernel=3, stride=1,
                                    padding=1, bn=False, relu=False)

        if self.args.conf_prop:
            # Confidence Branch
            # Confidence is shared for propagation and mask generation
            # 1/1
            self.cf_dec1 = conv_bn_relu(64+channels[0], 32, kernel=3, stride=1,
                                        padding=1)
            self.cf_dec0 = nn.Sequential(
                nn.Conv2d(32+64, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.Sigmoid()
            )

    def _concat(self, fd, fe, dim=1):
        # Decoder feature may have additional padding
        _, _, Hd, Wd = fd.shape
        _, _, He, We = fe.shape

        fd = F.interpolate(fd, size=(He, We), mode='bilinear', align_corners=True)

        f = torch.cat((fd, fe), dim=dim)

        return f

    def forward(self, rgb=None, depth=None):

        # model = simple_attn(128, 4)
        # input = torch.randn(1, 228*304, 128)
        # flops, params = profile(model, inputs=(input,228,304))
        # print('flops:{}'.format(flops))
        # print('params:{}'.format(params))
        # os.system("pause")

        # Encoding
        B_r,C_r,H_r,W_r = rgb.shape
        if self.mode == 'rgbd':
            fe1_rgb = self.conv1_rgb(rgb)
            fe1_dep = self.conv1_dep(depth)
            fe1_dep = self.attn_dep(fe1_dep.reshape(B_r,H_r*W_r,-1), fe1_rgb.reshape(B_r,H_r*W_r,-1), fe1_rgb.reshape(B_r,H_r*W_r,-1),H_r,W_r).reshape(B_r,-1,H_r,W_r)
            fe1 = torch.cat((fe1_rgb, fe1_dep), dim=1)
            fe1 = self.attn(fe1.reshape(B_r,H_r*W_r,-1),H_r,W_r).reshape(B_r,-1,H_r,W_r)
            # fe1 = self.conv1(fe1)
        elif self.mode == 'rgb':
            fe1 = self.conv(rgb)
        elif self.mode == 'd':
            fe1 = self.conv(depth)
        else:
            raise TypeError(self.mode)
        

        tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        fe1_tsne = tsne.fit_transform(fe1.reshape(64,H_r,W_r))
        x_min, x_max = fe1_tsne.min(0), fe1_tsne.max(0)
        X_norm = (fe1_tsne - x_min) / (x_max - x_min)  # 归一化
        plt.figure(figsize=(8, 8))
        for i in range(X_norm.shape[0]):
            plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
                    fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        plt.savefig('/home/xuyinuo/completionFormer/CompletionFormer/experiments/savefig_example.png')

        fe2, fe3, fe4, fe5, fe6, fe7 = self.former(fe1)
        # Shared Decoding
        fd6 = self.dec6(fe7)
        fd5 = self.dec5(self._concat(fd6, fe6))
        fd4 = self.dec4(self._concat(fd5, fe5))
        fd3 = self.dec3(self._concat(fd4, fe4))
        fd2 = self.dec2(self._concat(fd3, fe3))

        # Init Depth Decoding
        dep_fd1 = self.dep_dec1(self._concat(fd2, fe2))
        init_depth = self.dep_dec0(self._concat(dep_fd1, fe1))

        # Guidance Decoding
        gd_fd1 = self.gd_dec1(self._concat(fd2, fe2))
        guide = self.gd_dec0(self._concat(gd_fd1, fe1))

        if self.args.conf_prop:
            # Confidence Decoding
            cf_fd1 = self.cf_dec1(self._concat(fd2, fe2))
            confidence = self.cf_dec0(self._concat(cf_fd1, fe1))
        else:
            confidence = None

        return init_depth, guide, confidence, fe1

