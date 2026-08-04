"""
3.5 Transformer 架构
这个文件实现了 Transformer 编码器的核心组件，包括：
- 多头注意力机制
- 位置编码
- 前馈网络
- 残差连接和层归一化
"""

# ============================================================
# 导入必要的库
# ============================================================

import torch                          # PyTorch 深度学习框架
import torch.nn as nn                 # 神经网络模块，包含各种层和函数
import math                           # 数学库，用于 sqrt 等数学运算


# ============================================================
# 组件1：多头注意力机制（Multi-Head Attention）
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    核心思想：
    - 用多个"头"同时学习不同的关注方式
    - 每个头有自己的权重矩阵，学习不同的关注模式
    - 最后把所有头的结果拼接并融合
    
    数学公式：
    - MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
    - 其中 head_i = Attention(Q × W_i^Q, K × W_i^K, V × W_i^V)
    - Attention(Q, K, V) = softmax(QKᵀ / √d_k) V
    """
    
    def __init__(self, d_model, num_heads):
        """
        初始化多头注意力层
        
        参数：
        - d_model: 模型维度，每个词用多少个数字表示（比如64）
        - num_heads: 注意力头的数量（比如8）
        """
        super().__init__()  # 调用父类 nn.Module 的初始化方法
        
        # 保存参数
        self.d_model = d_model          # 模型维度，比如64
        self.num_heads = num_heads      # 头的数量，比如8
        self.d_k = d_model // num_heads # 每个头的维度 = 总维度 / 头数
                                        # 例如：64 / 8 = 8，每个头处理8维向量
        
        # 创建4个线性变换层（权重矩阵）
        # 这些权重在训练过程中会自动学习
        
        # W_Q：用于生成 Query（查询）的权重矩阵
        # 输入维度 d_model，输出维度 d_model
        self.W_Q = nn.Linear(d_model, d_model)
        
        # W_K：用于生成 Key（键）的权重矩阵
        self.W_K = nn.Linear(d_model, d_model)
        
        # W_V：用于生成 Value（值）的权重矩阵
        self.W_V = nn.Linear(d_model, d_model)
        
        # W_O：用于融合多头输出的权重矩阵
        self.W_O = nn.Linear(d_model, d_model)
        
    def forward(self, X):
        """
        前向传播
        
        参数：
        - X: 输入张量，形状为 (batch_size, seq_len, d_model)
              - batch_size: 批次大小（一次处理几个句子）
              - seq_len: 序列长度（句子有几个词）
              - d_model: 每个词的向量维度
        
        返回：
        - output: 注意力输出，形状 (batch_size, seq_len, d_model)
        - attention_weights: 注意力权重，形状 (batch_size, num_heads, seq_len, seq_len)
        """
        
        # 获取输入的形状信息
        batch_size, seq_len, _ = X.size()
        # batch_size: 批次大小
        # seq_len: 序列长度（词的个数）
        # _: d_model（用下划线表示我们暂时不用这个值）
        
        # ============================================================
        # 第1步：生成 Q, K, V
        # ============================================================
        # Q = X × W_Q，表示"我想找什么信息"
        # K = X × W_K，表示"每个位置有什么信息"
        # V = X × W_V，表示"每个位置的实际内容"
        
        # 计算 Q，并调整形状以支持多头
        # X 的形状：(batch_size, seq_len, d_model)
        # self.W_Q(X) 的形状：(batch_size, seq_len, d_model)
        # .view(batch_size, seq_len, self.num_heads, self.d_k) 的形状：(batch_size, seq_len, num_heads, d_k)
        # .transpose(1, 2) 的形状：(batch_size, num_heads, seq_len, d_k)
        Q = self.W_Q(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算 K，同样调整形状
        K = self.W_K(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算 V，同样调整形状
        V = self.W_V(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # ============================================================
        # 第2步：计算注意力分数
        # ============================================================
        # 公式：scores = Q × Kᵀ / √d_k
        # 
        # Q 的形状：(batch_size, num_heads, seq_len, d_k)
        # K.transpose(-2, -1) 的形状：(batch_size, num_heads, d_k, seq_len)
        # scores 的形状：(batch_size, num_heads, seq_len, seq_len)
        #
        # 除以 √d_k 是为了缩放，防止点积太大导致 softmax 梯度消失
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # ============================================================
        # 第3步：softmax 归一化
        # ============================================================
        # 把分数转换成概率分布（每行加起来等于1）
        # attention_weights[i][j] 表示第i个词对第j个词的关注程度
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        # ============================================================
        # 第4步：加权求和
        # ============================================================
        # 用注意力权重对 V 进行加权求和
        # attn_output 的形状：(batch_size, num_heads, seq_len, d_k)
        
        attn_output = torch.matmul(attention_weights, V)
        
        # ============================================================
        # 第5步：拼接所有头的输出
        # ============================================================
        # attn_output 的形状：(batch_size, num_heads, seq_len, d_k)
        # .transpose(1, 2) 的形状：(batch_size, seq_len, num_heads, d_k)
        # .contiguous() 确保内存连续（transpose 后需要）
        # .view(batch_size, seq_len, self.d_model) 的形状：(batch_size, seq_len, d_model)
        #
        # 拼接后：num_heads × d_k = d_model
        # 例如：8个头 × 8维 = 64维
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # ============================================================
        # 第6步：线性变换
        # ============================================================
        # 通过 W_O 融合所有头的信息
        # 最终输出形状：(batch_size, seq_len, d_model)
        
        return self.W_O(attn_output)


# ============================================================
# 组件2：位置编码（Positional Encoding）
# ============================================================

class PositionalEncoding(nn.Module):
    """
    位置编码
    
    为什么需要位置编码？
    - 自注意力机制没有位置信息
    - 对于自注意力来说，"我爱学习"和"学习爱我"是一样的
    - 需要告诉模型每个词在什么位置
    
    数学公式：
    - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    其中：
    - pos: 位置（0, 1, 2, ...）
    - i: 维度索引（0, 1, 2, ...）
    - d_model: 模型维度
    """
    
    def __init__(self, d_model, max_len=5000):
        """
        初始化位置编码
        
        参数：
        - d_model: 模型维度
        - max_len: 最大序列长度（最多支持多少个词）
        """
        super().__init__()
        
        # 创建位置编码矩阵，形状：(max_len, d_model)
        # 例如：max_len=5000, d_model=64，就是 5000×64 的矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 生成位置向量：[0, 1, 2, ..., max_len-1]
        # .unsqueeze(1) 把一维变成二维：[[0], [1], [2], ...]
        # 形状：(max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算分母项：10000^(2i/d_model)
        # torch.arange(0, d_model, 2) 生成 [0, 2, 4, ..., d_model-2]
        # 这是因为 sin 和 cos 是成对使用的（偶数维用sin，奇数维用cos）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度（0, 2, 4, ...）用 sin 函数
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # 奇数维度（1, 3, 5, ...）用 cos 函数
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 添加批次维度：(max_len, d_model) -> (1, max_len, d_model)
        # 这样可以和输入 (batch_size, seq_len, d_model) 相加
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不参与训练
        # register_buffer 的作用：
        # 1. 会保存到模型的 state_dict 中
        # 2. 不会被优化器更新（不是可学习参数）
        # 3. 会跟随模型移动到 GPU/CPU
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        前向传播
        
        参数：
        - x: 词嵌入，形状 (batch_size, seq_len, d_model)
        
        返回：
        - 添加位置编码后的张量，形状 (batch_size, seq_len, d_model)
        """
        # self.pe[:, :x.size(1), :] 截取与输入序列长度相同的位置编码
        # x.size(1) 就是 seq_len
        # 例如：输入有10个词，就只用前10个位置的编码
        return x + self.pe[:, :x.size(1), :]


# ============================================================
# 组件3：编码器层（Encoder Layer）
# ============================================================

class EncoderLayer(nn.Module):
    """
    Transformer 编码器层
    
    结构：
    输入 x
      ↓
    多头自注意力 → attn_output
      ↓
    残差连接 + 层归一化：x = LayerNorm(x + attn_output)
      ↓
    前馈网络 → ffn_output
      ↓
    残差连接 + 层归一化：x = LayerNorm(x + ffn_output)
      ↓
    输出 x
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        初始化编码器层
        
        参数：
        - d_model: 模型维度
        - num_heads: 注意力头数
        - d_ff: 前馈网络的隐藏层维度（通常是 d_model 的 4 倍）
        - dropout: Dropout 概率（随机丢弃神经元的比例，用于防止过拟合）
        """
        super().__init__()
        
        # 多头注意力层
        self.attention = MultiHeadAttention(d_model, num_heads)
        
        # 第一个层归一化（用于注意力之后）
        # LayerNorm 会对每个样本的特征维度进行归一化
        self.norm1 = nn.LayerNorm(d_model)
        
        # 第二个层归一化（用于前馈网络之后）
        self.norm2 = nn.LayerNorm(d_model)
        
        # 前馈网络（Feed-Forward Network）
        # 结构：线性层 → ReLU 激活 → 线性层
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),   # 第一层：d_model → d_ff（扩展维度）
            nn.ReLU(),                   # 激活函数：把负数变成0
            nn.Linear(d_ff, d_model)     # 第二层：d_ff → d_model（压缩回原维度）
        )
        
        # Dropout 层：随机丢弃一些神经元，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        参数：
        - x: 输入，形状 (batch_size, seq_len, d_model)
        
        返回：
        - 输出，形状 (batch_size, seq_len, d_model)
        """
        
        # ============================================================
        # 第1部分：多头注意力 + 残差连接 + 层归一化
        # ============================================================
        
        # 多头自注意力
        attn_output = self.attention(x)
        
        # 残差连接 + Dropout + 层归一化
        # x + self.dropout(attn_output) 是残差连接：把原始输入和注意力输出相加
        # self.norm1(...) 是层归一化：稳定训练过程
        x = self.norm1(x + self.dropout(attn_output))
        
        # ============================================================
        # 第2部分：前馈网络 + 残差连接 + 层归一化
        # ============================================================
        
        # 前馈网络
        ffn_output = self.ffn(x)
        
        # 残差连接 + Dropout + 层归一化
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


# ============================================================
# 组件4：完整的 Transformer 编码器
# ============================================================

class TransformerEncoder(nn.Module):
    """
    Transformer 编码器
    
    结构：
    输入序列（词的索引）
      ↓
    词嵌入：把词的索引转换成向量
      ↓
    位置编码：添加位置信息
      ↓
    编码器层 1
      ↓
    编码器层 2
      ↓
    ...
      ↓
    编码器层 N
      ↓
    输出序列（每个词的新表示）
    """
    
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000, dropout=0.1):
        """
        初始化 Transformer 编码器
        
        参数：
        - vocab_size: 词汇表大小（一共有多少个不同的词）
        - d_model: 模型维度（每个词用多少个数字表示）
        - num_heads: 注意力头数
        - num_layers: 编码器层的数量（堆叠多少层）
        - d_ff: 前馈网络的隐藏层维度
        - max_len: 最大序列长度
        - dropout: Dropout 概率
        """
        super().__init__()
        
        # 词嵌入层
        # 把词的索引（整数）转换成向量（浮点数）
        # 例如：词索引 5 → [0.1, -0.3, 0.5, ...]（d_model 维的向量）
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # 创建多个编码器层，用 ModuleList 存储
        # ModuleList 是 PyTorch 提供的列表容器，可以正确注册子模块
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)  # 重复 num_layers 次
        ])
        
        # Dropout 层
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        前向传播
        
        参数：
        - x: 输入序列，形状 (batch_size, seq_len)
              - batch_size: 批次大小
              - seq_len: 序列长度
              - 每个元素是词的索引（整数）
        
        返回：
        - 输出序列，形状 (batch_size, seq_len, d_model)
        """
        
        # ============================================================
        # 第1步：词嵌入
        # ============================================================
        # 把词的索引转换成向量
        # x 的形状：(batch_size, seq_len)
        # embedding(x) 的形状：(batch_size, seq_len, d_model)
        x = self.embedding(x)
        
        # ============================================================
        # 第2步：添加位置编码
        # ============================================================
        # 告诉模型每个词在什么位置
        x = self.positional_encoding(x)
        
        # ============================================================
        # 第3步：Dropout
        # ============================================================
        # 随机丢弃一些神经元，防止过拟合
        x = self.dropout(x)
        
        # ============================================================
        # 第4步：通过所有编码器层
        # ============================================================
        # 逐层处理，每一层的输出是下一层的输入
        for layer in self.layers:
            x = layer(x)
        
        return x


# ============================================================
# 演示函数
# ============================================================

def demonstrate_transformer():
    """
    演示 Transformer 编码器的使用
    """
    
    print("=" * 50)
    print("Transformer 架构")
    print("=" * 50)
    print()
    
    print("Transformer 的核心组件：")
    print("  1. 自注意力机制")
    print("  2. 多头注意力")
    print("  3. 位置编码")
    print("  4. 前馈网络")
    print("  5. 残差连接和层归一化")
    print()
    
    # ============================================================
    # 创建模型
    # ============================================================
    
    # 设置超参数
    vocab_size = 1000    # 词汇表大小：一共有1000个不同的词
    d_model = 64         # 模型维度：每个词用64个数字表示
    num_heads = 8        # 注意力头数：用8个头同时学习
    num_layers = 2       # 编码器层数：堆叠2层
    d_ff = 256           # 前馈网络隐藏层维度：64 × 4 = 256
    
    # 创建 Transformer 编码器模型
    model = TransformerEncoder(vocab_size, d_model, num_heads, num_layers, d_ff)
    
    # ============================================================
    # 创建输入数据
    # ============================================================
    
    batch_size = 2       # 批次大小：一次处理2个句子
    seq_len = 10         # 序列长度：每个句子有10个词
    
    # 创建随机输入，每个元素是词的索引（0到999之间的整数）
    # 形状：(2, 10)，表示2个句子，每个句子10个词
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # ============================================================
    # 前向传播
    # ============================================================
    
    # 把输入送入模型，得到输出
    # 输出形状：(2, 10, 64)，表示2个句子，每个句子10个词，每个词64维
    output = model(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print()
    
    print("编码器层结构：")
    print("  输入")
    print("    ↓")
    print("  多头自注意力")
    print("    ↓")
    print("  残差连接 + 层归一化")
    print("    ↓")
    print("  前馈网络")
    print("    ↓")
    print("  残差连接 + 层归一化")
    print("    ↓")
    print("  输出")
    print()
    
    print("完整编码器：")
    print("  输入序列")
    print("    ↓")
    print("  词嵌入 + 位置编码")
    print("    ↓")
    print("  编码器层 1")
    print("    ↓")
    print("  编码器层 2")
    print("    ↓")
    print("  ...")
    print("    ↓")
    print("  输出序列")
    
    return model, output


def compare_with_rnn():
    """
    对比 Transformer 和 RNN 的区别
    """
    
    print()
    print("=" * 50)
    print("Transformer vs RNN")
    print("=" * 50)
    print()
    
    print("| 特性 | RNN | Transformer |")
    print("|------|-----|-------------|")
    print("| 并行计算 | ❌ 不能 | ✅ 能 |")
    print("| 长距离依赖 | ❌ 难捕捉 | ✅ 容易捕捉 |")
    print("| 梯度消失 | ❌ 有 | ✅ 通过残差连接解决 |")
    print("| 训练速度 | ❌ 慢 | ✅ 快 |")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    # 运行演示函数
    model, output = demonstrate_transformer()
    
    # 对比 Transformer 和 RNN
    compare_with_rnn()
