#!/usr/bin/env python3
"""
简化的离线训练脚本 - 支持Transformers和spaCy
先下载预训练模型，然后离线训练
"""

import os
import json
import random
import torch
import argparse
import logging

# 设置离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_models():
    """下载预训练模型到本地"""
    print("="*60)
    print("下载预训练模型")
    print("="*60)
    
    # 临时取消离线模式进行下载
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_DATASETS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)
    
    try:
        # 下载BERT模型
        print("\n1. 下载BERT中文模型...")
        from transformers import AutoTokenizer, AutoModelForTokenClassification
        
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        model = AutoModelForTokenClassification.from_pretrained(
            "bert-base-chinese", 
            num_labels=7
        )
        print("✓ BERT模型下载完成")
        
        # 下载spaCy模型
        print("\n2. 下载spaCy中文模型...")
        import spacy
        os.system("python -m spacy download zh_core_web_sm")
        print("✓ spaCy模型下载完成")
        
        print("\n模型下载完成! 现在可以离线训练了。")
        return True
        
    except Exception as e:
        print(f"下载失败: {e}")
        return False
    finally:
        # 重新设置离线模式
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

def train_transformers(data_path="synthetic_training_data.json", epochs=3):
    """Transformers离线训练"""
    print("="*60)
    print("TRANSFORMERS 离线训练")
    print("="*60)
    
    try:
        from transformers import (
            AutoTokenizer, AutoModelForTokenClassification,
            TrainingArguments, Trainer, DataCollatorForTokenClassification
        )
        from datasets import Dataset
        import numpy as np
    except ImportError as e:
        print(f"导入错误: {e}")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载了 {len(data)} 个样本")
    
    # 2. 加载模型（离线模式）
    print("\n2. 加载预训练模型...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-chinese", 
        num_labels=7,
        local_files_only=True
    )
    print("✓ 模型加载成功")
    
    # 3. 数据预处理
    print("\n3. 处理数据...")
    label_map = {'O': 0, 'B-TECH': 1, 'I-TECH': 2, 'B-NUM': 3, 'I-NUM': 4, 'B-UNIT': 5, 'I-UNIT': 6}
    
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples['tokens'], truncation=True, padding=True, 
            max_length=256, is_split_into_words=True
        )
        labels = []
        for i, label in enumerate(examples['labels']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = []
            previous_word_idx = None
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label_map.get(label[word_idx], 0))
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    dataset = Dataset.from_list(data)
    tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True, remove_columns=dataset.column_names)
    train_split = tokenized_dataset.train_test_split(test_size=0.2, seed=42)
    
    # 4. 训练
    print(f"\n4. 开始训练 ({epochs} epochs)...")
    training_args = TrainingArguments(
        output_dir="./trained_models/transformers_offline",
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        logging_steps=5,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        push_to_hub=False,
        disable_tqdm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_split['train'],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer, padding=True)
    )
    
    train_result = trainer.train()
    
    # 5. 保存模型
    print("\n5. 保存模型...")
    output_dir = "./trained_models/transformers_offline"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\n✓ 训练完成! 损失: {train_result.training_loss:.4f}")
    print(f"✓ 模型保存在: {output_dir}")
    
    return train_result

def train_spacy(data_path="synthetic_training_data.json", epochs=10):
    """spaCy离线训练"""
    print("="*60) 
    print("SPACY 离线训练")
    print("="*60)
    
    try:
        import spacy
        from spacy.training import Example
        from spacy.tokens import DocBin
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请安装spaCy: pip install spacy")
        return None
    
    # 1. 加载数据
    print("\n1. 加载数据...")
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"加载了 {len(data)} 个样本")
    
    # 2. 初始化spaCy模型
    print("\n2. 初始化spaCy模型...")
    
    # 优先使用中文预训练模型 + pkuseg
    try:
        print("尝试使用中文预训练模型 + pkuseg...")
        nlp = spacy.load("zh_core_web_sm")
        print("✓ 使用中文预训练模型 (zh_core_web_sm)")
        
    except Exception as e:
        print(f"中文预训练模型加载失败: {e}")
        print("回退到英文预训练模型 + jieba分词器...")
        
        try:
            # 使用英文预训练模型但配置jieba分词器处理中文
            nlp = spacy.load("en_core_web_sm")
            
            # 安装并配置jieba分词器
            try:
                import jieba
                
                def jieba_tokenizer(text):
                    words = list(jieba.cut(text))
                    spaces = [True] * len(words)
                    return spacy.tokens.Doc(nlp.vocab, words=words, spaces=spaces)
                
                nlp.tokenizer = jieba_tokenizer
                print("✓ 使用英文预训练模型 + jieba中文分词器 (保持预训练效果)")
                
            except ImportError:
                print("! jieba未安装，请运行: pip install jieba")
                print("✓ 暂时使用英文预训练模型")
            
        except Exception as e2:
            print(f"英文预训练模型也加载失败: {e2}")
            raise RuntimeError("无法加载任何预训练模型，请确保已安装 en_core_web_sm")
    
    # 添加NER组件
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")
    
    # 添加标签
    for label in ['TECH', 'NUM', 'UNIT']:
        ner.add_label(label)
    
    # 3. 准备训练数据
    print("\n3. 准备训练数据...")
    examples = []
    for sample in data:
        text = ' '.join(sample['tokens'])
        entities = []
        
        # 提取实体
        current_start = 0
        current_entity = None
        current_label = None
        
        for i, (token, label) in enumerate(zip(sample['tokens'], sample['labels'])):
            if label.startswith('B-'):
                if current_entity:
                    entities.append((current_start, current_start + len(current_entity), current_label))
                current_entity = token
                current_label = label[2:]
                current_start = text.find(token, current_start)
            elif label.startswith('I-') and current_entity:
                current_entity += ' ' + token
            else:
                if current_entity:
                    entities.append((current_start, current_start + len(current_entity), current_label))
                current_entity = None
                current_label = None
                current_start = text.find(token, current_start) + len(token) + 1
        
        if current_entity:
            entities.append((current_start, current_start + len(current_entity), current_label))
        
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, {"entities": entities})
        examples.append(example)
    
    # 4. 训练
    print(f"\n4. 开始训练 ({epochs} epochs)...")
    
    # 使用更安全的初始化方式，避免tokenizer的initialize问题
    try:
        nlp.initialize(get_examples=lambda: examples[:10])  # 提供示例数据
    except Exception as e:
        print(f"! 标准初始化失败: {e}")
        print("! 尝试手动初始化...")
        # 手动初始化NER组件
        ner = nlp.get_pipe("ner")
        ner.initialize(lambda: examples[:10])
    
    for epoch in range(epochs):
        losses = {}
        random.shuffle(examples)
        
        for batch in spacy.util.minibatch(examples, size=8):
            nlp.update(batch, losses=losses)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{epochs}, 损失: {losses.get('ner', 0):.4f}")
    
    # 5. 保存模型
    print("\n5. 保存模型...")
    output_dir = "./trained_models/spacy_offline"
    nlp.to_disk(output_dir)
    
    print(f"✓ 训练完成!")
    print(f"✓ 模型保存在: {output_dir}")
    
    return losses

def test_model(framework, text="今天温度是25摄氏度，CPU有8个核心"):
    """测试训练好的模型"""
    print("="*60)
    print(f"测试 {framework.upper()} 模型")
    print("="*60)
    
    if framework == "transformers":
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            import torch
            
            model_path = "./trained_models/transformers_offline"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForTokenClassification.from_pretrained(model_path)
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            id2label = {0: 'O', 1: 'B-TECH', 2: 'I-TECH', 3: 'B-NUM', 4: 'I-NUM', 5: 'B-UNIT', 6: 'I-UNIT'}
            labels = [id2label[p.item()] for p in predictions[0]]
            
            print(f"输入: {text}")
            print("预测结果:")
            for token, label in zip(tokens, labels):
                if token not in ['[CLS]', '[SEP]', '[PAD]']:
                    print(f"  {token} -> {label}")
                    
        except Exception as e:
            print(f"测试失败: {e}")
    
    elif framework == "spacy":
        try:
            import spacy
            
            model_path = "./trained_models/spacy_offline" 
            nlp = spacy.load(model_path)
            
            doc = nlp(text)
            print(f"输入: {text}")
            print("预测结果:")
            for token in doc:
                print(f"  {token.text} -> {token.ent_iob_}-{token.ent_type_}" if token.ent_type_ else f"  {token.text} -> O")
            
            if doc.ents:
                print("识别的实体:")
                for ent in doc.ents:
                    print(f"  {ent.text} ({ent.label_})")
                    
        except Exception as e:
            print(f"测试失败: {e}")

def main():
    parser = argparse.ArgumentParser(description="简化的离线训练系统")
    parser.add_argument("--action", type=str, choices=["download", "train", "test"], 
                       default="train", help="执行的动作")
    parser.add_argument("--framework", type=str, choices=["transformers", "spacy"], 
                       default="transformers", help="训练框架")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--data", type=str, default="synthetic_training_data.json", help="训练数据")
    
    args = parser.parse_args()
    
    if args.action == "download":
        download_models()
        
    elif args.action == "train":
        if args.framework == "transformers":
            train_transformers(args.data, args.epochs)
        elif args.framework == "spacy":
            train_spacy(args.data, args.epochs)
            
    elif args.action == "test":
        test_model(args.framework)

if __name__ == "__main__":
    main()