import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
def get_lr_scheduler(optimizer,total_steps,warmup_ratio = 0.02,min_lr_ratio = 0.1):
    """
    线性预热 + 余弦退火衰减
    :param optimizer: 优化器实例
    :param total_steps: 训练全程迭代步数
    :param warmup_ratio: 预热步数占比
    :param min_lr_ratio: 最低学习率
    :return: 学习率调度器
    """
    warmup_steps = int(warmup_ratio * total_steps)
    max_lr = optimizer.param_groups[0]['lr']  #初始峰值学习率
    min_lr = max_lr * min_lr_ratio  #衰减下限学习率

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)  #线性上升预热，从0逐步升到max_lr
        progress = (step - warmup_steps) / max(1,total_steps - warmup_steps)
        return min_lr / max_lr + 0.5 * (1 - min_lr / max_lr) * (1 + math.cos(math.pi * progress))  #预热完成后，余弦退火下降到min_lr

    return torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda)

