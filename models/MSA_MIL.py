import torch
import torch.nn as nn
import torch.nn.functional as F
from part.ASPP import ASPP
from part.EMA import EMA

class MagnificationAdaptiveModule(nn.Module):
    def __init__(self, in_dim):
        super(MagnificationAdaptiveModule, self).__init__()
        self.mag_detector = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.feature_adapter = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        # x shape: (batch_size, n_patches, feat_dim)
        B, N, D = x.shape
        
        # 检测放大倍率
        patch_features = x.mean(dim=1)  # (B, feat_dim)
        mag_score = self.mag_detector(patch_features)  # (B, 1)
        
        # 特征自适应调整
        adapted_features = self.feature_adapter(x.view(-1, D)).view(B, N, D)
        adapted_features = adapted_features * mag_score.unsqueeze(-1)
        
        return adapted_features


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels)
        )
        
    def forward(self, x):
        b, n, c = x.size()
        x = x.transpose(1, 2)
        avg_out = self.fc(self.avg_pool(x).squeeze(-1))
        max_out = self.fc(self.max_pool(x).squeeze(-1))
        out = torch.sigmoid(avg_out + max_out)
        return out.view(b, 1, c)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return torch.sigmoid(x).transpose(1, 2)

class MSA_MIL(nn.Module):
    def __init__(self, L=512, D=128, K=1, in_dim=512, num_classes=2, dropout=0.3):  # 增加dropout率
        super(MSA_MIL, self).__init__()
        # 添加自适应模块
        self.mag_adapter = MagnificationAdaptiveModule(in_dim)
        
        # 原有的模型结构
        self.n_classes = num_classes
        
        self.L = L
        self.D = D
        self.K = K
        
        # 使用导入的ASPP模块
        self.aspp = ASPP(in_dim, L)
        self.norm1 = nn.LayerNorm(L)
        
        # 注意力机制
        self.attention_U = nn.Sequential(
            nn.Linear(L, D),
            nn.LayerNorm(D),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.attention_weights = nn.Linear(D, K)
        
        # 通道和空间注意力
        self.channel_attention = ChannelAttention(L)
        self.spatial_attention = SpatialAttention()
        self.norm2 = nn.LayerNorm(L)
        
        # 使用导入的EMA模块
        self.ema = EMA(L)
        
        # 替换原有分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 4)
        )
        # 优化 feature_extractor 结构
        self.feature_extractor = nn.Sequential(
            nn.Linear(L, L//4),  # 降维
            nn.GELU(),  # 使用GELU替代ReLU
            nn.LayerNorm(L//4),
            nn.Dropout(dropout),
            nn.Linear(L//4, L//4),
            nn.GELU(),
            nn.LayerNorm(L//4),
            nn.Dropout(dropout),
            nn.Linear(L//4, L),
            nn.LayerNorm(L)
        )

    def forward(self, x, return_features=False, return_mag_score=False):
        # x: (batch_size, n, L) or (n, L)

        # 确保输入是3维张量
        if len(x.shape) == 2:
            x = x.unsqueeze(0)  # 添加batch维度
        B, N, D = x.shape
        patch_features = x.mean(dim=1)  # (B, feat_dim)
        mag_score = self.mag_adapter.mag_detector(patch_features)
        x = self.mag_adapter(x)
        
        # 确保数据类型是float
        x = x.float()
        
        # 调整维度以适应ASPP的输入要求 [B, C, H, W]
        x = x.unsqueeze(-1)  # 添加W维度
        x = x.transpose(1, 2)  # 将channel维度调整到正确位置
        
        # 多尺度特征提取
        x = self.aspp(x)
        
        # 调整回原始维度 [B, N, C]
        x = x.squeeze(-1).transpose(1, 2)
        x = self.norm1(x)
        
        # 通道注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        # 空间注意力
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        x = self.norm2(x)
        identity = x
        x = self.feature_extractor(x)
        x = x + identity  # 残差连接
        x = x * torch.sigmoid(x)  # SE-like attention
        
        # 添加特征聚合
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x = x + x_mean + x_max
        
        # 调整维度以适应EMA的输入要求 [B, C, H, W]
        x_ema = x.transpose(1, 2).unsqueeze(-1)  # [B, C, H, 1]
        ema_features = self.ema(x_ema)
        # 调整回原始维度 [B, N, C]
        ema_features = ema_features.squeeze(-1).transpose(1, 2)
        feature = x + ema_features

        # 特征变换和注意力权重计算
        A_raw = self.attention_weights(self.attention_U(feature))  # (B, n, K)
        A = torch.transpose(A_raw, 1, 2)
        A = F.softmax(A, dim=-1) # [B, K, n]
        feature = torch.bmm(A, feature) # [B, K, 512]
        feature = feature.squeeze(0)
        
        # 在最终分类前添加特征增强
        feature = feature * torch.sigmoid(feature)  # 特征重校准
        
        # 添加标签平滑和温度缩放
        if self.training:
            logits = self.classifier(feature) / 1.5  # 温度缩放
            Y_prob = F.softmax(logits, dim=1) * 0.9 + 0.1 / logits.size(1)  # 标签平滑
        else:
            logits = self.classifier(feature)
            Y_prob = F.softmax(logits, dim=1)
        
        # 添加标签平滑
        if self.training:
            Y_prob = F.softmax(logits, dim=1) * 0.9 + 0.1 / logits.size(1)
        else:
            Y_prob = F.softmax(logits, dim=1)
            
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        A_raw = A_raw
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        # 如果 return_mag_score 为 True，则只返回 mag_score
        if return_mag_score:
            return mag_score
        return hazards, S, Y_hat, A_raw, Y_prob