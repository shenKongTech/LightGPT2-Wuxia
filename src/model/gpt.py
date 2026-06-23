import math
import torch
import torch.nn as nn
from .transformer_block import TransformerBlock
from .layers import LayerNorm
class GPTModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg['vocab_size'],cfg['emb_dim'])  #Token嵌入层
        self.pos_emb = nn.Embedding(cfg['context_length'],cfg['emb_dim'])  #可学习位置编码，提供序列位置信息
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])  #堆叠多个Transformer块
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head  = nn.Linear(cfg['emb_dim'],cfg['vocab_size'],bias = False)
        self.out_head.weight = self.tok_emb.weight  #权重共享

        #位置嵌入权重正态初始化
        nn.init.normal_(self.tok_emb.weight,mean = 0.0,std = 0.02)
        nn.init.normal_(self.pos_emb.weight,mean = 0.0,std = 0.02)

        for block in self.trf_blocks:
            nn.init.normal_(block.attn.out_proj.weight,mean = 0.0,std = 0.02 / math.sqrt(2 * cfg['n_layers']))  #对Transformer块的输出层应用缩放初始化，缓解梯度爆炸
            nn.init.normal_(block.ff.net[-1].weight,mean = 0.0,std = 0.02 / math.sqrt(2 * cfg['n_layers']))

    def forward(self,in_idx):
        batch_size,seq_len = in_idx.shape
        tok_emb = self.tok_emb(in_idx)  #词嵌入
        pos = torch.arange(seq_len,device = in_idx.device)  #生成位置索引
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb  #词嵌入与位置嵌入融合
        x = self.drop_emb(x)
        x = self.trf_blocks(x)  #多层Transformer块进行特征提取
        x = self.final_norm(x)
        logits = self.out_head(x)  #映射得到预测分数
        return logits
