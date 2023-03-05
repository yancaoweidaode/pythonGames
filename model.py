import os
import numpy as np
import math
import random
import time
import scipy.io

from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
# from confusion_matrix import plot_confusion_matrix
# from cm_no_normal import plot_confusion_matrix_nn
# from torchsummary import summary

# 和混淆矩阵有关
import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn


cudnn.benchmark = False
cudnn.deterministic = True

# writer = SummaryWriter('/home/syh/Documents/MI/code/Trans/TensorBoardX/')

# torch.cuda.set_device(6)
gpus = [0]
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, gpus))


from scipy.signal import butter, lfilter
def butter_bandpass(lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Linear filter application
def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandstop(lowstop, highstop, fs, order=2):
    nyq = 0.5 * fs
    low = lowstop / nyq
    high = highstop / nyq
    b, a = butter(order, [low, high], btype='bandstop')
    return b, a

# Linear filter application
def butter_bandstop_filter(data, lowstop, highstop, fs, order=2):
    b, a = butter_bandstop(lowstop, highstop, fs, order=order)
    y = lfilter(b, a, data)
    return y

# double filter function: first stop, then pass
def filter_eeg(data, fs, bandstop=None, bandpass=None, order=2):
    if bandstop:
        lowstop, highstop = bandstop
        data = butter_bandstop_filter(data, lowstop, highstop, fs, order)
    if bandpass:
        lowpass, highpass = bandpass
        data = butter_bandpass_filter(data, lowpass, highpass, fs, order)
    return data


## 这个的输出的通道为，经过两次卷积之后信号变成了N*10*1*T，然后经过Rearrange之后，变成了N*T*10
class PatchEmbedding(nn.Module):
    def __init__(self, emb_size):
        # self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(1, 2, (1, 51), (1, 1)),
#             nn.Conv2d(1, 2, (1, 25), (1, 1)),
            nn.BatchNorm2d(2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2, emb_size, (9, 10), stride=(1, 5)),
            nn.BatchNorm2d(emb_size),
            nn.LeakyReLU(0.2),
            Rearrange('b e (h) (w) -> b (h w) e'),   ## 重组向量就是将b*e*h*w的向量重组为b*(h*w)*e的向量，在这里h是1
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        # self.positions = nn.Parameter(torch.randn((100 + 1, emb_size)))
        # self.positions = nn.Parameter(torch.randn((2200 + 1, emb_size)))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.projection(x)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
#         print('out put of patchembedding')
#         print(x.shape)
        # position
        # x += self.positions
        return x


## 残差连接
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


# b n e 分别是batch_size,time_steps,emb_size
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        out = self.clshead(x)
        return x, out
    
class Transform(nn.Module):
    def __init__(self, p_d_model=10, p_nhead=5, p_dropout=0.5, p_num_layers=3):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=p_d_model, nhead=p_nhead,dropout = p_dropout, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=p_num_layers)
    def forward(self, x):
#         print('in put of trandform')
#         print(x.shape)
        out = self.transformer_encoder(x)
#         print('out put of trandform')
#         print(out.shape)
        return out

class ViT(nn.Sequential):
    def __init__(self, emb_size=10, depth=3, n_classes=3, **kwargs):
        super().__init__(
            # channel_attention(),
            # ResidualAdd(
            #     nn.Sequential(
            #         nn.LayerNorm(1000),
            #         # channel_attention(),
            #         # Matrix(),
            #         nn.Dropout(0.5),
            #     )
            # ),
            nn.LayerNorm(1000),
            nn.Dropout(0.5),

            PatchEmbedding(emb_size),
            Transform(),
#             TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes)
        )