#!/usr/bin/env python3
"""
使用训练好的模型进行推理
"""

import os
import torch
import json
import argparse
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformersInference:
    """Transformers模型推理"""
    
    def __init__(self, model_path):
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 标签映射
        self.id2label = {
            0: 'O', 1: 'B-TECH', 2: 'I-TECH',
            3: 'B-NUM', 4: 'I-NUM',
            5: 'B-UNIT', 6: 'I-UNIT'
        }
        
        logger.info(f"Transformers模型加载完成: {model_path}")
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """对单个文本进行预测"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        # 获取tokens和标签
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = [self.id2label[p.item()] for p in predictions[0]]
        
        # 过滤特殊tokens
        result_tokens = []
        result_labels = []
        for token, label in zip(tokens, labels):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                result_tokens.append(token)
                result_labels.append(label)
        
        # 提取实体
        entities = self._extract_entities(result_tokens, result_labels)
        
        return {
            'text': text,
            'tokens': result_tokens,
            'labels': result_labels,
            'entities': entities
        }
    
    def _extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict[str, str]]:
        """从预测结果中提取实体"""
        entities = []
        current_entity = []
        current_type = None
        
        for token, label in zip(tokens, labels):
            if label.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type
                    })
                current_entity = [token]
                current_type = label[2:]
            elif label.startswith('I-') and current_type:
                # 继续当前实体
                if label[2:] == current_type:
                    current_entity.append(token)
                else:
                    # 类型不匹配，结束当前实体
                    if current_entity:
                        entities.append({
                            'text': ''.join(current_entity),
                            'type': current_type
                        })
                    current_entity = []
                    current_type = None
            else:
                # O标签，结束当前实体
                if current_entity:
                    entities.append({
                        'text': ''.join(current_entity),
                        'type': current_type
                    })
                current_entity = []
                current_type = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append({
                'text': ''.join(current_entity),
                'type': current_type
            })
        
        return entities

class SpacyInference:
    """spaCy模型推理"""
    
    def __init__(self, model_path):
        import spacy
        
        self.nlp = spacy.load(model_path)
        logger.info(f"spaCy模型加载完成: {model_path}")
    
    def predict_text(self, text: str) -> Dict[str, Any]:
        """对单个文本进行预测"""
        doc = self.nlp(text)
        
        # 获取tokens和标签
        tokens = [token.text for token in doc]
        labels = []
        for token in doc:
            if token.ent_type_:
                labels.append(f"{token.ent_iob_}-{token.ent_type_}")
            else:
                labels.append('O')
        
        # 提取实体
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'type': ent.label_
            })
        
        return {
            'text': text,
            'tokens': tokens,
            'labels': labels,
            'entities': entities
        }

def load_test_data(file_path: str) -> List[str]:
    """加载测试数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 过滤空行和去除换行符
    texts = [line.strip() for line in lines if line.strip()]
    return texts

def run_inference(framework: str, model_path: str, test_files: List[str]):
    """运行推理"""
    print("="*80)
    print(f"{framework.upper()} 模型推理")
    print("="*80)
    
    # 初始化模型
    if framework == "transformers":
        model = TransformersInference(model_path)
    elif framework == "spacy":
        model = SpacyInference(model_path)
    else:
        raise ValueError(f"不支持的框架: {framework}")
    
    all_results = {}
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"! 测试文件不存在: {test_file}")
            continue
            
        print(f"\n📄 处理文件: {test_file}")
        print("-" * 60)
        
        # 加载测试数据
        test_texts = load_test_data(test_file)
        print(f"加载了 {len(test_texts)} 行文本")
        
        results = []
        
        # 处理每行文本
        for i, text in enumerate(test_texts[:10], 1):  # 只处理前10行作为示例
            if len(text) > 200:  # 截断过长的文本
                text = text[:200] + "..."
            
            print(f"\n[{i}] 输入: {text}")
            
            try:
                result = model.predict_text(text)
                results.append(result)
                
                # 显示预测结果
                if result['entities']:
                    print("🎯 识别的实体:")
                    for entity in result['entities']:
                        print(f"   • {entity['text']} ({entity['type']})")
                else:
                    print("   无实体识别")
                    
                # 显示详细的token标签（可选，太多时跳过）
                if len(result['tokens']) < 20:
                    print("🏷️  详细标注:")
                    for token, label in zip(result['tokens'], result['labels']):
                        if label != 'O':
                            print(f"   {token} -> {label}")
                
            except Exception as e:
                print(f"   ❌ 预测失败: {e}")
                continue
        
        all_results[test_file] = results
        
        # 统计结果
        total_entities = sum(len(r['entities']) for r in results)
        entity_types = {}
        for result in results:
            for entity in result['entities']:
                entity_type = entity['type']
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print(f"\n📊 统计结果:")
        print(f"   总实体数: {total_entities}")
        if entity_types:
            print("   实体类型分布:")
            for entity_type, count in entity_types.items():
                print(f"     {entity_type}: {count}")
    
    # 保存结果
    output_file = f"inference_results_{framework}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 推理结果已保存到: {output_file}")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="模型推理脚本")
    parser.add_argument(
        "--framework",
        type=str,
        choices=["transformers", "spacy"],
        default="transformers",
        help="选择模型框架"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="模型路径（如果不指定，使用默认路径）"
    )
    parser.add_argument(
        "--test-files",
        nargs="+",
        default=["../data/zh.txt", "../data/en.txt"],
        help="测试文件列表"
    )
    
    args = parser.parse_args()
    
    # 设置默认模型路径
    if not args.model_path:
        if args.framework == "transformers":
            args.model_path = "./trained_models/transformers_offline"
        elif args.framework == "spacy":
            args.model_path = "./trained_models/spacy_offline"
    
    # 检查模型是否存在
    if not os.path.exists(args.model_path):
        print(f"❌ 模型路径不存在: {args.model_path}")
        print("\n请先训练模型:")
        print(f"python train_offline_simple.py --framework {args.framework} --epochs 5")
        return
    
    print(f"🚀 开始推理...")
    print(f"   框架: {args.framework}")
    print(f"   模型: {args.model_path}")
    print(f"   测试文件: {args.test_files}")
    
    # 运行推理
    results = run_inference(args.framework, args.model_path, args.test_files)
    
    print("\n✅ 推理完成!")

if __name__ == "__main__":
    main()