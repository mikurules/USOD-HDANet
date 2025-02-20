import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import RoIPool
from Code.lib.MSAC import MSACBlock

def weight_init(module):
    for n, m in module.named_children():
        # print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Softmax)):
            pass
        else:
            try:
                m.initialize()
            except:
                pass


############################################################################

class PatchTransformerEncoder(nn.Module):
    def __init__(self, in_channels, patch_size=10, embedding_dim=128, num_heads=4):
        super(PatchTransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, num_heads, dim_feedforward=1024)#batch_first=True??
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=4)  # takes shape S,N,E 

        self.embedding_convPxP = nn.Conv2d(in_channels, embedding_dim,
                                           kernel_size=patch_size, stride=patch_size, padding=0)
        #如果 `embeddings.shape[2]`（修补和扁平化后的序列长度）超过 500（预初始化的位置编码数量），这可能会导致运行时错误。
        self.positional_encodings = nn.Parameter(torch.rand(500, embedding_dim), requires_grad=True)

    def forward(self, x):
        embeddings = self.embedding_convPxP(x).flatten(2)  # .shape = n,c,s = n, embedding_dim, s
        # embeddings = nn.functional.pad(embeddings, (1,0))  # extra special token at start ?
        embeddings = embeddings + self.positional_encodings[:embeddings.shape[2], :].T.unsqueeze(0)
        # change to S,N,E format required by transformer
        embeddings = embeddings.permute(2, 0, 1)
        x = self.transformer_encoder(embeddings)  # .shape = S, N, E
        return x


class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)
class mViT(nn.Module):
    def __init__(self, in_channels, n_query_channels=128, patch_size=16, dim_out=256,
                 embedding_dim=128, num_heads=4, norm='linear'):
        super(mViT, self).__init__()
        self.norm = norm
        self.n_query_channels = n_query_channels
        self.patch_transformer = PatchTransformerEncoder(in_channels, patch_size, embedding_dim, num_heads)
        self.dot_product_layer = PixelWiseDotProduct()

        self.conv3x3 = nn.Conv2d(in_channels, embedding_dim, kernel_size=3, stride=1, padding=1)
        # self.regressor = nn.Sequential(nn.Linear(embedding_dim, 256),
        #                                nn.LeakyReLU(),
        #                                nn.Linear(256, 256),
        #                                nn.LeakyReLU(),
        #                                nn.Linear(256, dim_out))

    def forward(self, x):
        # n, c, h, w = x.size()
        tgt = self.patch_transformer(x.clone())  # .shape = S, N, E
        x = self.conv3x3(x)
        #regression_head, queries = tgt[0, ...], tgt[1:self.n_query_channels + 1, ...]
        queries = tgt[1:self.n_query_channels + 1, ...]
        # Change from S, N, E to N, S, E
        queries = queries.permute(1, 0, 2)
        range_attention_maps = self.dot_product_layer(x, queries)  # .shape = n, n_query_channels, h, w

        return range_attention_maps


class HDEM(nn.Module):
    def __init__(self, Cs1, Cs2, K1, K2):#Cs1=32, Cs2=64, K1=9, K2=9
        super(HDEM, self).__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, Cs1, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=Cs1),
            nn.ReLU(inplace=True),
            nn.Conv2d(Cs1, Cs2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=Cs2),
            nn.ReLU(inplace=True)
        )

        self.lk_block1 = MSACBlock(Cs2, K1)
        self.lk_block2 = MSACBlock(Cs2, K2)


        n_query_channels=128
        self.thead = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(Cs2, Cs2, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=Cs2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            ############
            mViT(Cs2),            
            ############
            nn.Conv2d(n_query_channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )



    def forward(self, x,not_acc):
        atm = torch.mean(x, dim=(2, 3), keepdim=True)
        #print(x.shape)
        x = self.stem(x)
        #print(x.shape)
        x = self.lk_block1(x)
        #print(x.shape)
        x = self.lk_block2(x)
        ###########
        if not_acc==False:
            loss_acc=self.accloss(x,atm)
        else:
            loss_acc=None
        ##########
        trans=self.thead(x)
        trans = torch.clamp(trans, min=0.01,max=1)
        return trans, atm,loss_acc
    def initialize(self):
        weight_init(self)
    

class MainModel(nn.Module):
    def __init__(self):
        super().__init__()				
        self.estimation = HDEM(Cs1=32, Cs2=64, K1=9, K2=9)	

    def forward(self, x,not_acc):
        ori_x_c=x
        #print(x*10)
        trans, atm,loss_acc = self.estimation(x,not_acc)
        atm = atm.expand_as(ori_x_c)
        #print(atm)
        out = (ori_x_c - (1 - trans)*atm)/trans
        return trans, atm, out,loss_acc

    def initialize(self):
        weight_init(self)

