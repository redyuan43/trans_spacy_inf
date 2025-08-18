#!/usr/bin/env python3
"""
下载预训练模型到本地缓存
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification
import os

def download_model():
    """下载bert-base-chinese模型到本地"""
    
    model_name = "bert-base-chinese"
    
    print(f"正在下载模型: {model_name}")
    print("这可能需要几分钟时间...")
    
    try:
        # 下载tokenizer
        print("下载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 下载model
        print("下载model...")
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=7  # 我们的标签数量
        )
        
        # 保存到本地缓存目录
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"模型已下载到缓存目录: {cache_dir}")
        print("下载完成！")
        
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 使用镜像站点:")
        print("   export HF_ENDPOINT=https://hf-mirror.com")
        print("3. 手动下载模型文件")
        return False

if __name__ == "__main__":
    success = download_model()
    if success:
        print("\n现在可以运行训练脚本了:")
        print("python train_model.py")