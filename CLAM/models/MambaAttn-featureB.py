import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
def initialize_weights(module):
    for m in module.modules():
        if isinstance(m,nn.Linear):
            # ref from clam
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

from part.CBAM import CBAM
from part.HaloAttention import HaloAttention
from part.CSAM import CSAM
from part.TokenSelect import TokenSelect #输入图片的去噪，合并
from part.WTConv2d import WTConv2d #卷积
from part.DFF import DFF #给特征加权重
from part.GLSA import GLSA #分别输入全局空间注意（GSA）模块和局部空间注意力 （LSA） 模块
from mamba.mamba_ssm import SRMamba
from mamba.mamba_ssm import BiMamba
from mamba.mamba_ssm import Mamba
class UnsqueezeLayer(nn.Module):  
    def forward(self, x):  
        return x.permute(0, 2, 1).unsqueeze(-1)
class squeezeLayer(nn.Module):  
    def forward(self, x):  
        return x.squeeze(-1).permute(0, 2, 1)

class MambaAttn(nn.Module):
    def __init__(self,num_classes=1,dropout=0,act='ReLU',in_dim = 512,survival = False, layer=2, rate=10, type="SRMamba"):
        super(MambaAttn, self).__init__()
        #self.in_dim = in_dim
        #self.num_classes = num_classes
        #self.L = 512 #512
        #self.D = 128 #128
        #self.K = 1
        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]

        self._fc1 = nn.Sequential(*self._fc1)
        self.normA = nn.LayerNorm(512)
        self.normB = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        #self.attention = nn.Sequential(
        #    nn.Linear(self.L, self.D),
        #    nn.Tanh(),#差距大
        #    nn.Linear(self.D, self.K)
        #)
        #self.classifier = nn.Sequential(
        #    nn.Linear(self.L*self.K, self.num_classes),
        #)

        self.survival = survival
        self.rate = rate
        self.type = type
        self.apply(initialize_weights)
        #self.Mamba = Mamba(d_model=1024, d_state=16, d_conv=4, expand=2)
        #self.PVMamba = PVMamba(input_dim=1024,output_dim=512)
        #self.CSAM = CSAM(num_slices=1,num_channels=512)
        #self.WTConv2d = WTConv2d(512,512)
        self.layers2.append(
            nn.Sequential(
                nn.LayerNorm(512),
                UnsqueezeLayer(),
                WTConv2d(512,512),
                squeezeLayer(),
                )
        )
        self.layers2.append(
            nn.Sequential(
                nn.LayerNorm(512),
                UnsqueezeLayer(),
                GLSA(input_dim=512,embed_dim=512),
                squeezeLayer(),
                )
        )
        
        self.DFF = DFF(512)
        #self.GLSA = GLSA(input_dim=512,embed_dim=512)

        #可训练门控
        self.sigmoid = nn.Sigmoid()
        #self.prelu = nn.PReLU()
        #self.FFN = nn.Sequential(  
        #    nn.Linear(512, 1024),  
        #    nn.Tanh(),  
        #    nn.Linear(1024, 512),  
        #)

        
        #选择适合的token
        self.TokenSelect = TokenSelect(
            expansion_step=[0,8,16],#[0,8,16] [0, 30, 60]
            keep_rate=[0.5,0.75,1.0],
            initialization_keep_rate=0.25,
            expansion_multiple_stage=2,
            distance="cosine"
        )
        self.classifier = nn.Linear(512, 4)
        #VIT cls token模块
        self.pos_embed = nn.Parameter(torch.randn(1, 1, 512), requires_grad=True)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512), requires_grad=True)
        self.pos_drop = nn.Dropout(dropout)  
        self.head = nn.Linear(512, num_classes) if num_classes > 0 else Identity()
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        if type == "SRMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        SRMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "Mamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        Mamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        elif type == "BiMamba":
            for _ in range(layer):
                self.layers.append(
                    nn.Sequential(
                        nn.LayerNorm(512),
                        BiMamba(
                            d_model=512,
                            d_state=16,  
                            d_conv=4,    
                            expand=2,
                        ),
                        )
                )
        else:
            raise NotImplementedError("Mamba [{}] is not implemented".format(type))

        #cross
        #self.liner_q = nn.Linear(512, 512)
        #self.liner_k = nn.Linear(512, 512)
        #self.liner_v = nn.Linear(512, 512)

        #self.GLSAmamba1 = GLSAmamba(input_dim=512,output_dim=512)  
        #self.GLSAmamba2 = GLSAmamba(input_dim=512,output_dim=512) 
        #self.GLSAmamba3 = GLSAmamba(input_dim=512,output_dim=512) 
        #self.GLSAmamba4 = GLSAmamba(input_dim=512,output_dim=512) 
        

    
    def forward(self, x, return_attn=False,no_norm=False):
       #feature = self.feature(x)
        #size.(1,token,512)
        #分成两部分token 可训练mask
        if len(x.shape) == 2:
            x = x.expand(1, -1, -1)
        x = x.float()  # [B, n, 1024]
        
        #x = self._fc1(x)  # [B, n, 512]
        x = self._fc1(x)  # [B, n, 512]
        feature = x
        #feature = x
        #VIT cls token模块
        #M, N, _ = feature.shape
        #cls_tokens = self.cls_token.expand((M, -1, -1))
        #pos_embed = self.pos_embed[:, :N+1, :]
        #feature = torch.cat((cls_tokens, feature), dim=1)
        #feature = feature + pos_embed
        feature = self.pos_drop(feature)
        #size.(1,token+1,512),后续省略+1]

        #A = self.Mamba(feature)#适合低频（全局）；卷积/GAT/GCN 
        
        #size.(1,512,token,1)
        #A = self.WTConv2d(A)#适合低频（全局）；卷积/GAT/GCN

        for layer in self.layers2:
            B_ = feature
            B = layer[0](feature)
            B = layer[1](B).squeeze(-1).permute(0, 2, 1)
            B = B + B_
        B = self.normA(B)

        feature = B
        
        #feature = torch.cat((A, B), dim=1) 

        A_raw = self.attention(feature) # [B, n, K]
        A = torch.transpose(A_raw, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        feature = torch.bmm(A, feature) # [B, K, 512]
        feature = feature.squeeze(0)

        logits = self.classifier(feature)  # [B, n_classes]
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = A_raw
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, A_raw, Y_prob
        return logits, Y_prob, Y_hat, A_raw, results_dict
        
