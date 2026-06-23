import time
import torch
import torch.nn.functional as F
from typing import Optional
def generate_stream(model,prompt,tokenizer,max_new = 250,temperature = 0.7,top_k = 40):
    """
    流式逐字生成函数
    :param model: 训练完成的GPT模型
    :param prompt: 用户开头输入的提示词
    :param max_new: 最多续写的token数
    :param temperature: 温度系数，控制生成随机性
    :param top_k: top_k采样，抑制生僻乱码
    :return: 完整拼接后的生成文本
    """
    ctx_len = model.cfg['context_length']
    encoded = torch.tensor([tokenizer.encode(prompt)],dtype = torch.long).to(DEVICE)
    print('\n>>>',end='',flush = True)

    for _ in range(max_new):
        idx_cond = encoded[:,-ctx_len:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        #Top_k截断，只取概率最高的Top_k个token，其余置负无穷
        if top_k > 0:
            top_vals,top_idx = torch.topk(logits,top_k)
            logits = torch.full_like(logits,-torch.inf).scatter_(1,top_idx,top_vals)
        logits /= temperature  #温度缩放，降低和提高分布平滑度
        probs = F.softmax(logits,dim = -1)
        idx_next = torch.multinomial(probs,num_samples = 1)  #跟据概率随机采样下一个token id

        encoded = torch.cat((encoded,idx_next),dim = -1)
        token_next = tokenizer.decode(idx_next[0].tolist())  #解码单token转为文字，实时打印输出
        print(token_next,end='',flush = True)
        time.sleep(0.03)  #打印延时，实现流式输出效果
    print('\n')
    return tokenizer.decode(encoded[0].tolist())
