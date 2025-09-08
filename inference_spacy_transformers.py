#!/usr/bin/env python3
"""
SpaCy-Transformers模型推理示例
展示如何使用训练好的spacy-transformers模型进行NER预测
"""

import spacy
import json
from pathlib import Path
import time

class SpacyTransformersInference:
    def __init__(self, model_path="./trained_models/spacy_transformers"):
        """初始化spacy-transformers推理器"""
        print(f"加载SpaCy-Transformers模型: {model_path}")
        self.nlp = spacy.load(model_path)
        print(f"✓ 模型加载完成")
        print(f"  Pipeline组件: {self.nlp.pipe_names}")
        print(f"  支持的实体类型: {self.nlp.get_pipe('ner').labels}")
    
    def predict(self, text):
        """对单个文本进行预测"""
        doc = self.nlp(text)
        
        result = {
            'text': text,
            'entities': [],
            'tokens': [token.text for token in doc],
            'labels': []
        }
        
        # 提取实体
        for ent in doc.ents:
            result['entities'].append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
        
        # 生成BIO标签
        labels = ['O'] * len(doc)
        for ent in doc.ents:
            labels[ent.start] = f'B-{ent.label_}'
            for i in range(ent.start + 1, ent.end):
                labels[i] = f'I-{ent.label_}'
        result['labels'] = labels
        
        return result
    
    def predict_batch(self, texts, batch_size=32):
        """批量预测"""
        results = []
        
        # 使用nlp.pipe进行批量处理（更高效）
        docs = list(self.nlp.pipe(texts, batch_size=batch_size))
        
        for text, doc in zip(texts, docs):
            result = {
                'text': text,
                'entities': [],
                'tokens': [token.text for token in doc]
            }
            
            for ent in doc.ents:
                result['entities'].append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char
                })
            
            results.append(result)
        
        return results
    
    def analyze_performance(self, texts):
        """分析推理性能"""
        print("\n性能分析:")
        print(f"测试样本数: {len(texts)}")
        
        # 单条推理
        start_time = time.time()
        for text in texts:
            _ = self.nlp(text)
        single_time = time.time() - start_time
        print(f"单条推理总时间: {single_time:.2f}秒")
        print(f"平均每条: {single_time/len(texts)*1000:.2f}毫秒")
        
        # 批量推理
        start_time = time.time()
        _ = list(self.nlp.pipe(texts, batch_size=32))
        batch_time = time.time() - start_time
        print(f"批量推理总时间: {batch_time:.2f}秒")
        print(f"平均每条: {batch_time/len(texts)*1000:.2f}毫秒")
        print(f"加速比: {single_time/batch_time:.2f}x")

def demo():
    """演示推理功能"""
    print("="*60)
    print("SpaCy-Transformers 推理演示")
    print("="*60)
    
    # 初始化推理器
    try:
        inferencer = SpacyTransformersInference()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("请先训练模型: python train_spacy_transformers.py")
        return
    
    # 测试文本
    test_texts = [
        "今天温度是28摄氏度，湿度为65%",
        "使用HPLC分析，溶解度为5.5 mg/mL，pH值为7.4",
        "FTIR在3000 cm⁻¹处有吸收峰，强度为0.85",
        "反应时间30分钟，温度控制在100°C",
        "CPU温度达到85度，风扇转速2500 rpm",
        "The melting point is 252°C as measured by DSC",
        "XRPD analysis shows crystalline form at 25.3°",
        "储存条件：2-8°C，避光保存，有效期36个月"
    ]
    
    print("\n" + "="*60)
    print("单条推理测试")
    print("="*60)
    
    # 展示前3个例子的详细结果
    for i, text in enumerate(test_texts[:3], 1):
        print(f"\n示例 {i}:")
        result = inferencer.predict(text)
        print(f"原文: {result['text']}")
        
        if result['entities']:
            print("识别的实体:")
            for ent in result['entities']:
                print(f"  - {ent['text']:<10} ({ent['label']}) [{ent['start']}:{ent['end']}]")
        else:
            print("  未识别到实体")
        
        # 显示token级别的标注
        print("Token标注:")
        for token, label in zip(result['tokens'][:20], result['labels'][:20]):
            if label != 'O':
                print(f"  {token}: {label}")
    
    print("\n" + "="*60)
    print("批量推理测试")
    print("="*60)
    
    # 批量推理
    batch_results = inferencer.predict_batch(test_texts)
    
    print(f"\n处理了 {len(batch_results)} 条文本")
    
    # 统计实体
    entity_stats = {}
    for result in batch_results:
        for ent in result['entities']:
            label = ent['label']
            entity_stats[label] = entity_stats.get(label, 0) + 1
    
    print("\n实体统计:")
    for label, count in sorted(entity_stats.items()):
        print(f"  {label}: {count}个")
    
    # 性能分析
    print("\n" + "="*60)
    print("性能分析")
    print("="*60)
    inferencer.analyze_performance(test_texts * 10)  # 重复10次以获得更准确的时间
    
    # 保存结果
    output_file = "spacy_transformers_inference_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(batch_results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已保存到: {output_file}")

def interactive_mode():
    """交互式推理模式"""
    print("\n" + "="*60)
    print("SpaCy-Transformers 交互式推理")
    print("="*60)
    print("输入文本进行实体识别（输入'quit'退出）")
    
    try:
        inferencer = SpacyTransformersInference()
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    while True:
        print("\n" + "-"*40)
        text = input("请输入文本: ").strip()
        
        if text.lower() == 'quit':
            print("退出交互模式")
            break
        
        if not text:
            continue
        
        result = inferencer.predict(text)
        
        if result['entities']:
            print("\n识别的实体:")
            for ent in result['entities']:
                print(f"  {ent['text']:<15} {ent['label']:<10} [{ent['start']}:{ent['end']}]")
        else:
            print("未识别到实体")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SpaCy-Transformers推理')
    parser.add_argument('--mode', choices=['demo', 'interactive'], 
                       default='demo', help='运行模式')
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        demo()
    else:
        interactive_mode()