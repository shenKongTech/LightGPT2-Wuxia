import torch
import torch.nn as nn
from .layers import MultiHeadAttention, FeedForward, LayerNorm
class TransformerBlock(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg['emb_dim'],
            d_out = cfg['emb_dim'],
            context_length = cfg['context_length'],
            dropout = cfg['drop_rate'],
            num_heads = cfg['n_heads'],
            qkv_bias = cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self,x):
        shortcut = x
        x = self.norm1(x)  #先层归一化再进注意力
        x = self.attn(x)  #层归一化后计算因果多头注意力
        x = self.dropout(x)  #对注意力输出作正则化
        x = x + shortcut  #残差连接

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)  #先层归一化再进前馈神经网络
        x = self.dropout(x)  #对非线性前馈变换作正则化
        x = x + shortcut  #残差连接
        return x

