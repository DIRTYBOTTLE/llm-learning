"""
4.6 实战：用 GPT 生成文本
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model():
    """加载 GPT-2 模型"""
    
    print("正在加载 GPT-2 模型...")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    print("模型加载完成！")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_length=100, 
                  temperature=0.7, top_k=50, top_p=0.95):
    """
    生成文本
    
    参数：
    - model: GPT 模型
    - tokenizer: 分词器
    - prompt: 提示文本
    - max_length: 最大生成长度
    - temperature: 温度参数（控制随机性）
    - top_k: Top-K 采样参数
    - top_p: Top-P 采样参数
    """
    
    # 编码输入
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            do_sample=True,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            num_return_sequences=1,
            no_repeat_ngram_size=2  # 避免重复
        )
    
    # 解码输出
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text


def demonstrate_temperature():
    """演示温度参数的影响"""
    
    model, tokenizer = load_model()
    
    prompt = "Today the weather is"
    
    print("\n" + "=" * 50)
    print("温度参数的影响")
    print("=" * 50)
    
    for temp in [0.3, 0.7, 1.0, 1.5]:
        result = generate_text(model, tokenizer, prompt, 
                             max_length=50, temperature=temp)
        print(f"\n温度 {temp}:")
        print(f"  {result}")


def demonstrate_top_k():
    """演示 Top-K 参数的影响"""
    
    model, tokenizer = load_model()
    
    prompt = "I love programming because"
    
    print("\n" + "=" * 50)
    print("Top-K 参数的影响")
    print("=" * 50)
    
    for k in [10, 50, 100]:
        result = generate_text(model, tokenizer, prompt, 
                             max_length=50, top_k=k)
        print(f"\nTop-K {k}:")
        print(f"  {result}")


def interactive_generation():
    """交互式生成"""
    
    model, tokenizer = load_model()
    
    print("\n" + "=" * 50)
    print("交互式文本生成")
    print("=" * 50)
    print("输入提示文本，模型会生成续写")
    print("输入 'quit' 退出")
    
    while True:
        prompt = input("\n请输入提示文本: ")
        
        if prompt.lower() == 'quit':
            break
        
        result = generate_text(model, tokenizer, prompt, max_length=100)
        print(f"\n生成结果:\n{result}")


if __name__ == "__main__":
    # 演示温度参数
    demonstrate_temperature()
    
    # 演示 Top-K 参数
    demonstrate_top_k()
    
    # 交互式生成
    # interactive_generation()  # 取消注释以启用交互模式
