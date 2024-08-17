import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
class AddNorm(nn.Module):
    def __init__(self, size, sublayer, dropout=0):   #size为隐藏层大小
        super(TransformerSublayerWithAddNorm, self).__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        "将'Add and Norm'步骤应用到任意一个子层上"
        # Sublayer的输出通过dropout, 然后与原始输入x相加, 最后进行层归一化
        return self.norm(x + self.dropout(self.sublayer(x)))

class ResBlock(nn.Module):
    
    def __init__(self,dim,dropout = 0):#downsample=None表示虚线的残差结构
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU()
        self.w1 = nn.Linear(dim,dim)
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim,dim)
    def forward(self,x):
        x0 = x
        
        out = self.w1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.w2(out)
        
        return out + x0
    
    

# 构建SPP层(空间金字塔池化层)
class SPPLayer(torch.nn.Module):

    def __init__(self, level_list,pool_type):
        super(SPPLayer, self).__init__()
        #assert num_levels%2 == 0
        self.num_levels = level_list #数字列表
        self.pool_type = pool_type
        

    def forward(self, x):
        num_batch, num_seq, h = x.size() # num:样本数量 c:通道数 h:高 w:宽
        flag = True
        for level in self.num_levels:  
            kernel_size = (math.ceil(num_seq/level),1)
            padding = (math.floor((kernel_size[0]*level-num_seq+1)/2), 0)
            if padding[0] * 2 >= kernel_size[0]:
                kernel_size = (math.floor(num_seq/level),1)
                padding = (math.floor((-kernel_size[0]*level+num_seq+1)/2), 0)
            # 选择池化方式 
            if self.pool_type == 'max_pool':
                t = F.max_pool2d(x, kernel_size=kernel_size, padding = padding)
            else:
                t = F.avg_pool2d(x, kernel_size=kernel_size, padding = padding)

            # 展开、拼接
            if flag:
                flag = False
                x_flatten = t.view(num_batch,1, -1)
            else:
                x_flatten = torch.cat((x_flatten,t.view(num_batch,1,-1)),2)
        return x_flatten
    
# 构建Variable pooling layer层(空间金字塔池化层)
# class VPLayer(torch.nn.Module):
class VPLayer(nn.Module):

    def __init__(self, blocks_num_list, factor=10, device='cpu',statistic = ["mean","var"],variable = False):
        super(VPLayer, self).__init__()
        self.device = device
        self.statistic = statistic
        self.factor = factor
        self.blocks_num_list = blocks_num_list
        if variable:
            print("variable")
            self.blocks_score = nn.ParameterList([nn.Parameter(torch.zeros(k, device=device)) for k in self.blocks_num_list])
        else:
            print("not variable")
            self.blocks_score = nn.ParameterList([torch.zeros(k, device=device) for k in self.blocks_num_list])
    def integrate_mean(self, x, p, q):
        # 向量化实现，一次性计算所有段
        assert p < q
        q = torch.clamp(q, max=x.size(1)-1e-2)
        if torch.floor(p) == torch.floor(q):
                    return x[:,int(torch.floor(p))]
        p_floor = torch.floor(p).item()  # 这里假设 p 和 q 是单个元素的张量
        q_floor = torch.floor(q).item()

        floor_values = x[:, int(p_floor)] * (torch.ceil(p) - p)  # 使用 int() 确保索引是整数
        ceil_values = x[:, int(q_floor)] * (q - q_floor)

        # 中间值的向量化求和
        ranges = torch.arange(int(torch.ceil(p).item()), int(torch.floor(q).item()), device=self.device).long()
        med_values = x[:, ranges].sum(dim=1) if len(ranges) > 0 else 0

        return (med_values + floor_values + ceil_values) / (q - p)
    
    def integrate_var(self, x, p, q, mean):       
        assert p < q
        q = torch.clamp(q, max=x.size(1) - 1e-2)
        if torch.floor(p) == torch.floor(q):
            return (x[:, int(torch.floor(p))] - mean) ** 2
        
        p_floor = torch.floor(p).item()
        q_floor = torch.floor(q).item()
        
        # 计算方差的各个部分
        floor_values = (x[:, int(p_floor)] - mean) ** 2 * (torch.ceil(p) - p)
        ceil_values = (x[:, int(q_floor)] - mean) ** 2 * (q - q_floor)
        
        ranges = torch.arange(int(torch.ceil(p).item()), int(torch.floor(q).item()), device=self.device).long()
        med_values_sq = ((x[:, ranges] - mean.unsqueeze(1)) ** 2).sum(dim=1) if len(ranges) > 0 else 0
        
        # 计算最终的方差的标准差
        variance = torch.sqrt((med_values_sq + floor_values + ceil_values) / (q - p))
        return variance
    
    def forward(self, x):
        num_batch, num_seq, _ = x.size()
        result = []

        for block_scores in self.blocks_score:
            block_percent = F.softmax(block_scores * self.factor, dim=0)
            block_pos = block_percent.cumsum(dim=0) * num_seq
            block_pos = torch.cat([torch.tensor([0.0], device=self.device), block_pos])
            
            # 向量化集成函数
            if "mean" in self.statistic:
                mean = torch.stack([self.integrate_mean(x, block_pos[j], block_pos[j+1]) for j in range(len(block_pos)-1)], dim=1)
                result.append(mean)
            if "var" in self.statistic:
                if not "mean" in self.statistic:
                    mean = torch.stack([self.integrate_mean(x, block_pos[j], block_pos[j+1]) for j in range(len(block_pos)-1)], dim=1)
                var = torch.stack([self.integrate_var(x, block_pos[j], block_pos[j+1], mean[:,j,:]) for j in range(len(block_pos)-1)], dim=1)
                result.append(var)
        result = torch.cat(result, dim=1)
        #result = result.reshape(num_batch,-1)
        return result
    
class MyAttentionChainPool(nn.Module):
    """Pools residue-based representations to chain-based representations using a chain mask and attention.
    Args:
        n_head (int): number of attention heads
        d_model (int): dimension of embeddings to be pooled

    Inputs:
        h (torch.tensor): of size (batch_size, sequence_length, d_model)
        C (torch.tensor): of size (batch_size, sequence_length)

    Outputs:
        output (torch.tensor): of size (batch_size, n_chains, d_model)
        chain_mask (torch.tensor): of size (batch_size, n_chains)
    """

    def __init__(self, n_head = 1, d_model = 512, poolnum = 16,dropout = 0.0):
        super().__init__()
        self.poolnum =  poolnum
        self.attention = nn.ModuleList([nn.MultiheadAttention(embed_dim = d_model, num_heads = n_head, dropout = dropout,batch_first=True)\
                                        for i in range(poolnum)])

    def get_query(self, x):
        return torch.ones(x.size(0), 1, x.size(2)).type(x.dtype).to(x.device)

    def forward(self, h, C):
        bs, num_res = C.size()
        chains = C.abs().unique()
        chains = (
            chains[chains > 0].unsqueeze(-1).repeat(1, bs).reshape(-1).unsqueeze(-1)
        )
        num_chains = len(chains.unique())

        h_repeat = h.repeat(num_chains, 1, 1)
        C_repeat = C.repeat(num_chains, 1)
        mask = (C_repeat == chains).unsqueeze(-2)
        
        seqlen = h_repeat.size()[1]
        per = torch.ones(self.poolnum,device = h.device) / self.poolnum * seqlen
        per = per.cumsum(dim=0)
        per = torch.cat([torch.tensor([0.0], device=h.device), per])
        floor = torch.floor(per).int()
        ceil = torch.ceil(per).int()
        output = torch.tensor([],device = h.device)
        for j in range(len(per)-1):   
            a = self.attention[j]
            h_get = h_repeat[:,floor[j]:ceil[j+1],:]
            output = torch.cat([ output ,a(self.get_query(h_get), h_get, h_get)[0] ], dim=0)
       
        output = torch.cat(output.split(bs), 1)
        chain_mask = torch.stack(mask.squeeze(1).any(dim=-1).split(bs), -1)
        return output, chain_mask
    
    
    
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask

        attn = self.softmax(u) # 4.Softmax
        output = torch.bmm(attn, v) # 5.Output

        return attn, output
    
class MyMultiHeadAttention(nn.Module):
    """ Multi-Head Attention """

    def __init__(self, n_head, d_k_, d_v_, d_k, d_v, d_o):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc_q = nn.Linear(d_k_, n_head * d_k)
        self.fc_k = nn.Linear(d_k_, n_head * d_k)
        self.fc_v = nn.Linear(d_v_, n_head * d_v)

        self.attention = ScaledDotProductAttention(scale=np.power(d_k, 0.5))

        self.fc_o = nn.Linear(n_head * d_v, d_o)

    def forward(self, q, k, v, mask=None):

        n_head, d_q, d_k, d_v = self.n_head, self.d_k, self.d_k, self.d_v

        batch, n_q, d_q_ = q.size()
        batch, n_k, d_k_ = k.size()
        batch, n_v, d_v_ = v.size()

        q = self.fc_q(q) # 1.单头变多头
        k = self.fc_k(k)
        v = self.fc_v(v)
        q = q.view(batch, n_q, n_head, d_q).permute(2, 0, 1, 3).contiguous().view(-1, n_q, d_q)
        k = k.view(batch, n_k, n_head, d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, d_k)
        v = v.view(batch, n_v, n_head, d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1, 1)
        attn, output = self.attention(q, k, v, mask=mask) # 2.当成单头注意力求输出

        output = output.view(n_head, batch, n_q, d_v).permute(1, 2, 0, 3).contiguous().view(batch, n_q, -1) # 3.Concat
        output = self.fc_o(output) # 4.仿射变换得到最终输出

        return attn, output
