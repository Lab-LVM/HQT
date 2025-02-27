# reference by https://github.com/liyunqianggyn/Deep-Unsupervised-Image-Hashing/blob/main/ImageHashing/Cifar10_I.py

import torch
import torch.nn.functional as F
from torch import nn
from models.utils import BihalfHash, load_backbone

def hash_layer(input, gamma):
    return BihalfHash.apply(input, gamma)

class BihalfNet(nn.Module):
    def __init__(self, encode_length=16, gamma=6, arch="vgg16", use_timm=False):
        super(BihalfNet, self).__init__()
        self.gamma = gamma
        self.use_timm = use_timm
        self.model = load_backbone(arch, use_timm)
        self.fc_encode = nn.Linear(4096, encode_length, bias=False)

    def forward(self, x, _):
        if self.use_timm:
            x = self.model.forward_features(x)
            x = self.model.pre_logits(x)
            x = x.squeeze()
        else:
            x = self.model.features(x)
            x = x.view(x.size(0), -1)
            x = self.model.classifier(x)
        h = self.fc_encode(x)
        b = hash_layer(h,self.gamma)

        return x, h, b

class BihalfLoss(nn.Module):
    def __init__(self, hqt=False, alpha=0.5):
        super(BihalfLoss, self).__init__()
        self.hqt = hqt
        self.alpha = alpha

    def forward(self, x, b, labels):
        target_b = F.cosine_similarity(b[:int(labels.size(0) / 2)], b[int(labels.size(0) / 2):])
        target_x = F.cosine_similarity(x[:int(labels.size(0) / 2)], x[int(labels.size(0) / 2):])
        loss = F.mse_loss(target_b, target_x)
        if self.hqt:
            hqt_loss = F.binary_cross_entropy_with_logits(b, labels.float())
            loss = (1 - self.alpha) * loss + self.alpha * hqt_loss
        return loss

