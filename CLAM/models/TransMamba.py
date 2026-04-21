import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from nystrom_attention import NystromAttention
from timm.models.layers import trunc_normal_
from mamba.mamba_ssm import SRMamba


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

'''
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
'''
class TransLayer(nn.Module):

    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim = dim,
            dim_head = dim//8,
            heads = 8,
            num_landmarks = dim//2,
            pinv_iterations = 6,
            residual = True,
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x

'''
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
'''
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

'''
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
'''
class Trans_Agg(nn.Module):
    def __init__(self):
        super(Trans_Agg, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)


    def forward(self, **kwargs):

        h = kwargs['data'].float()
        
        h = self._fc1(h)
        
        #---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1)

        #---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        #---->Translayer x1
        h = self.layer1(h)

        #---->PPEG
        h = self.pos_layer(h, _H, _W)
        
        #---->Translayer x2
        h = self.layer2(h)

        return h


class Dynamic_AgentAttn(nn.Module):
    def __init__(self, dim=512, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=86, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(int(agent_num ** 0.5), int(agent_num ** 0.5)))
        self.softmax = nn.Softmax(dim=-1)

        # Define agent_tokens as a learnable parameter
        self.agent_tokens = nn.Parameter(torch.randn(1, agent_num, dim))

    def forward(self, x):
        device_1 = x.device
        b, c, n = x.shape
        window_size = (int(n ** 0.5), int(n ** 0.5))

        # Dynamic initialization of bias terms
        ah_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, window_size[0] // self.sr_ratio, 1))
        aw_bias = nn.Parameter(torch.zeros(1, self.num_heads, self.agent_num, 1, window_size[1] // self.sr_ratio))
        ha_bias = nn.Parameter(torch.zeros(1, self.num_heads, window_size[0], 1, self.agent_num))
        wa_bias = nn.Parameter(torch.zeros(1, self.num_heads, 1, window_size[1], self.agent_num))

        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x.transpose(1, 2)).transpose(1, 2)

        if self.sr_ratio > 1:
            x_ = x.reshape(b, c, int(n ** 0.5), int(n ** 0.5))
            x_ = self.sr(x_).reshape(b, c, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x.transpose(1, 2)).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        # Expand agent_tokens to match batch size
        agent_tokens = self.agent_tokens.expand(b, -1, -1)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)

        # Agent-to-patch attention bias
        kv_size = (window_size[0] // self.sr_ratio, window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1).to(device_1)
        position_bias2 = (ah_bias + aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1).to(device_1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # Patch-to-agent attention bias
        agent_bias1 = nn.functional.interpolate(self.na_bias, size=window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1).to(device_1)
        agent_bias2 = (ha_bias + wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1).to(device_1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, int(n ** 0.5) // self.sr_ratio, int(n ** 0.5) // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(int(n ** 0.5), int(n ** 0.5)), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransMamba(nn.Module):
    def __init__(self, in_dim, n_classes, dropout, act='gelu', survival = True, layer=2, rate=5, agent_num=86):
        super(TransMamba, self).__init__()

        self._fc1 = [nn.Linear(in_dim, 512)]
        if act.lower() == 'relu':
            self._fc1 += [nn.ReLU()]
        elif act.lower() == 'gelu':
            self._fc1 += [nn.GELU()]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*self._fc1)

        self.norm = nn.LayerNorm(512)
        self.layers = nn.ModuleList()
        self.survival = survival

        self.Trans_Agg = Trans_Agg()
        self.Dynamic_AgentAttn = Dynamic_AgentAttn(num_heads=8, agent_num=agent_num)

        for _ in range(layer):
            self.layers.append(nn.Sequential(nn.LayerNorm(512),SRMamba(d_model=512, d_state=16, d_conv=4, expand=2,),))

        self.n_classes = n_classes
        self.classifier = nn.Linear(512, self.n_classes)
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.rate = rate
        self.agent_num = agent_num

        self.apply(initialize_weights)
        

    def forward(self, x):

        if len(x.shape) == 2:
            x = x.expand(1, -1, -1) 
        h = x.float()
        h = self._fc1(h)

        h = self.Trans_Agg(data = h)

        H_2 = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H_2))), int(np.ceil(np.sqrt(H_2)))
        add_length = _H * _W - H_2
        h = torch.cat([h, h[:, :add_length, :]], dim=1)
        h = h.permute(0, 2, 1)

        h = self.Dynamic_AgentAttn(h)

        for layer in self.layers:
            h_ = h
            h = layer[0](h)
            h = layer[1](h, rate=self.rate)
            h = h + h_ 

        h = self.norm(h) 
        A_raw = self.attention(h) 
        A = torch.transpose(A_raw, 1, 2)
        A = F.softmax(A, dim=-1)
        h = torch.bmm(A, h)
        h = h.squeeze(0) 

        logits = self.classifier(h)

        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        results_dict = None
        if self.survival:
            Y_hat = torch.topk(logits, 1, dim = 1)[1]
            hazards = torch.sigmoid(logits)
            S = torch.cumprod(1 - hazards, dim=1)
            return hazards, S, Y_hat, A_raw, Y_prob
        return logits, Y_prob, Y_hat, A_raw, results_dict
    
    def relocate(self):
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_tensor = torch.rand(22260, 1024).to(device)
    attention_module = TransMamba(in_dim=1024, n_classes=4, dropout=0.25, act='gelu', survival=True, layer=2, rate=5).to(device)
    output_tensor = attention_module(input_tensor)