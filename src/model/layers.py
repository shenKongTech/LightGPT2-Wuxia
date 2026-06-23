import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in,d_out,context_length,dropout,num_heads,qkv_bias = False):
        """
        :param d_in: 输入特征维度
        :param d_out: 输出维度
        :param context_length: 上下文窗口长度
        :param dropout: 注意力权重丢弃率
        :param num_heads: 注意力头数
        :param qkv_bias: 偏置项
        """
        assert d_out % num_heads == 0
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads  #单个注意力头的特征维度

        #QKV映射线性层
        self.W_query = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.W_key= nn.Linear(d_in,d_out,bias =qkv_bias)
        self.W_value = nn.Linear(d_in,d_out,bias = qkv_bias)
        self.out_proj = nn.Linear(d_out,d_out)  #多头输出后的融合投影层
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('causal_mask',torch.triu(torch.ones(context_length,context_length),diagonal = 1))  #注册掩码缓冲区

    def forward(self,x):
        batch,num_tokens,d_in = x.shape
        #投影得到QKV矩阵
        q = self.W_query(x)
        k = self.W_key(x)
        v = self.W_value(x)

        #多头拆分，同时调整维度
        q = q.view(batch,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        k = k.view(batch,num_tokens,self.num_heads,self.head_dim).transpose(1,2)
        v = v.view(batch,num_tokens,self.num_heads,self.head_dim).transpose(1,2)

        #计算缩放点积注意力，得到注意力分数并应用因果掩码
        attn_scores = q @ k.transpose(-2,-1) / math.sqrt(self.head_dim)
        mask = self.causal_mask.bool()[:num_tokens,:num_tokens]
        attn_scores.masked_fill_(mask,-torch.inf)

        #softmax归一化后得到注意力权重并应用正则化
        attn_weights = F.softmax(attn_scores,dim = -1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ v  #权重加权得到每个头的上下文向量

        #合并多头并展平所有头的输出
        context_vec = context_vec.transpose(1,2).contiguous()
        context_vec = context_vec.view(batch,num_tokens,self.d_out)
        return self.out_proj(context_vec)  #多头融合输出投影，返回最终特征

class LayerNorm(nn.Module):
    def __init__(self,emb_dim,eps = 1e-5):
        """
        层归一化，恢复特征表达能力
        :param emb_dim: 特征嵌入维度
        :param eps: 极小值
        :scale: 可学习缩放参数
        :shift: 可学习偏置参数
        """
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        var = x.var(dim = -1,keepdim = True,unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x,3))))

class FeedForward(nn.Module):
    """
    前馈神经网络，学习单个Token的内部深层语义
    """
    def __init__(self,cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg['emb_dim'],4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'],cfg['emb_dim'])
        )

    def forward(self,x):
        return self.net(x)
