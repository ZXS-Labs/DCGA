"""
    CompletionFormer
    ======================================================================

    CompletionFormer implementation
"""

from .nlspn_module import NLSPN
from .backbone import Backbone
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from thop import profile


from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold,datasets
from sklearn.manifold import TSNE

class CompletionFormer(nn.Module):
    def __init__(self, args):
        super(CompletionFormer, self).__init__()

        self.args = args
        self.prop_time = self.args.prop_time
        self.num_neighbors = self.args.prop_kernel*self.args.prop_kernel - 1

        self.backbone = Backbone(args, mode='rgbd')

        if self.prop_time > 0:
            self.prop_layer = NLSPN(args, self.num_neighbors, 1, 3,
                                    self.args.prop_kernel)

    def forward(self, sample):
        rgb = sample['rgb']
        dep = sample['dep']

        pred_init, guide, confidence = self.backbone(rgb, dep)
        pred_init = pred_init + dep

        # tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
        # fe1_tsne = tsne.fit_transform(fe1)
        # x_min, x_max = fe1_tsne.min(0), fe1_tsne.max(0)
        # X_norm = (fe1_tsne - x_min) / (x_max - x_min)  # 归一化
        # plt.figure(figsize=(8, 8))
        # for i in range(X_norm.shape[0]):
        #     plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
        #             fontdict={'weight': 'bold', 'size': 9})
        # plt.xticks([])
        # plt.yticks([])
        # plt.savefig('/home/xuyinuo/completionFormer/CompletionFormer/experiments/savefig_example.png')

        # Diffusion
        y_inter = [pred_init, ]
        conf_inter = [confidence, ]
        if self.prop_time > 0:
            y, y_inter, offset, aff, aff_const = \
                self.prop_layer(pred_init, guide, confidence, dep, rgb)
        else:
            y = pred_init
            offset, aff, aff_const = torch.zeros_like(y), torch.zeros_like(y), torch.zeros_like(y).mean()

        # Remove negative depth
        y = torch.clamp(y, min=0)
        # best at first
        y_inter.reverse()
        conf_inter.reverse()
        if not self.args.conf_prop:
            conf_inter = None

        output = {'pred': y, 'pred_init': pred_init, 'pred_inter': y_inter,
                  'guidance': guide, 'offset': offset, 'aff': aff,
                  'gamma': aff_const, 'confidence': conf_inter}

        return output
